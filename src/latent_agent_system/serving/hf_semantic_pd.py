"""Keep upstream HF LatentMAS semantics; move decode forwards to another process.

The upstream method, HF generate loop, logits processors, sampler and RNG stay
unchanged. The P process executes latent agents and the judge prefill. D runs
subsequent judge model forwards using the exact original KV via CUDA IPC.
This is a functional HF P/D reference, not a native-vLLM or network benchmark.
"""
import argparse
from functools import wraps
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace


def tensor_digest(t):
    import torch
    a = t.detach().cpu().contiguous()
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha256": hashlib.sha256(a.view(torch.uint8).numpy().tobytes()).hexdigest()}


def cache_digest(cache):
    tensors = cache.to_legacy_cache()
    rows = [[tensor_digest(t) for t in layer] for layer in tensors]
    return {"positions": cache.get_seq_length(), "layers": len(rows), "tensors": rows,
            "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}


def upstream_config(max_tokens):
    return SimpleNamespace(model_name=os.environ.get("LATENT_SERVING_MODEL", "Qwen/Qwen3-4B"), device="cuda:0", device2="cuda:0",
                           task="gsm8k", prompt="sequential", think=False,
                           latent_space_realign=True, latent_only=False, sequential_info_only=False,
                           max_new_tokens=max_tokens, method="latent_mas",
                           tensor_parallel_size=1, gpu_memory_utilization=.22,
                           enable_prefix_caching=False, use_second_HF_model=False)


def decode_worker(conn, root, suite, max_tokens):
    os.setsid()
    log = open(Path(root) / f"logs/{suite}-D.log", "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    cache = outputs = kwargs = request = None
    try:
        import torch
        sys.path.insert(0, str(Path(root) / "sources/LatentMAS"))
        from models import ModelWrapper
        from utils import set_seed
        set_seed(17)
        torch.set_num_threads(4)
        # Use exactly upstream initialization, including its tokenizer-length
        # resize. Loading AutoModel directly retains 151936 output rows instead
        # of the upstream wrapper's 151669, changing the sampling distribution.
        cfg = upstream_config(max_tokens)
        wrapper = ModelWrapper(cfg.model_name, torch.device("cuda:0"), use_vllm=False, args=cfg)
        model = wrapper.model
        torch.cuda.synchronize()
        conn.send({"ready": True, "pid": os.getpid(), "tokenizer_length": len(wrapper.tokenizer),
                   "input_vocab": model.get_input_embeddings().weight.shape[0],
                   "output_vocab": model.get_output_embeddings().weight.shape[0]})
        with torch.inference_mode():
            while True:
                request = conn.recv()
                command = request["command"]
                if command == "stop":
                    break
                if command == "reset":
                    cache = outputs = kwargs = request = None
                    gc.collect()
                    conn.send({"ok": True})
                    continue
                assert command == "forward"
                kwargs = request["kwargs"]
                if request["first"]:
                    assert cache is None
                    cache = kwargs.pop("past_key_values")
                    received = cache_digest(cache)
                    assert received == request["expected_cache"], "Remote KV bytes changed"
                else:
                    assert "past_key_values" not in kwargs and cache is not None
                    received = None
                kwargs["past_key_values"] = cache
                before = cache.get_seq_length()
                start = time.perf_counter()
                outputs = model(**kwargs)
                torch.cuda.synchronize()
                cache = outputs.past_key_values
                assert cache.get_seq_length() == before + kwargs["input_ids"].shape[1]
                conn.send({"ok": True, "outputs": outputs, "received_cache": received,
                           "forward_wall_s": time.perf_counter()-start,
                           "positions_after": cache.get_seq_length()})
    except Exception as exc:
        try:
            conn.send({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
        except Exception:
            pass
        raise
    finally:
        cache = outputs = kwargs = request = None
        gc.collect()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["reference", "pd"], required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--verify-observer", action="store_true")
    parser.add_argument("--shadow-decode", action="store_true", help="Diagnostic only: compare each D forward with a local teacher-forced forward")
    return parser


def run(args):
    import torch
    import torch.multiprocessing as mp
    root = Path(os.environ["LATENT_SERVING_ROOT"])
    (root / "logs").mkdir(parents=True, exist_ok=True)
    outdir = root / "results" / args.suite
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{args.mode}.jsonl"
    if outfile.exists():
        raise RuntimeError("Refusing to overwrite results")
    sys.path.insert(0, str(root / "sources/LatentMAS"))
    from models import ModelWrapper
    from methods.latent_mas import LatentMASMethod
    from utils import extract_gold, normalize_answer, set_seed
    from latent_agent_system.data.answer_parsing import final_number, numeric
    samples = root / "samples.json"
    if not samples.is_file():
        samples = root / "results/compare100_v1/samples.json"
    items = json.loads(samples.read_text())
    for item in items:
        item["numeric_gold"] = item["gold"]
        item["gold"] = normalize_answer(extract_gold(item["solution"]))
        item["question"] = item["question"].strip()
    batches = args.batches if args.batches is not None else list(range(50))
    config = {"mode": args.mode, "backend": "HF", "batch_size": 2, "latent_steps": 40,
              "max_new_tokens": args.max_tokens, "temperature": .6, "top_p": .95,
              "do_sample": True, "seed": 17, "batches": batches,
              "visible_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "dtype": "bfloat16",
              "semantic_reference": "Unchanged upstream HF LatentMASMethod.run_batch",
              "transport": "CUDA IPC for true KV and model outputs; sampler remains in P",
              "scope": "functional HF P/D correctness reference; not native vLLM"}
    (outdir / f"{args.mode}_config.json").write_text(json.dumps(config, indent=2))
    samplesfile = outdir / "samples.json"
    if samplesfile.exists():
        assert json.loads(samplesfile.read_text()) == items
    else:
        samplesfile.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    statusfile = outdir / f"{args.mode}_status.json"
    def status(stage, **extra):
        statusfile.write_text(json.dumps({"stage": stage, "mode": args.mode, **extra}, indent=2))
    child = conn = None
    def ask(request):
        conn.send(request)
        if not conn.poll(180):
            raise TimeoutError("HF decode RPC timed out")
        response = conn.recv()
        if not response.get("ok"):
            raise RuntimeError(json.dumps(response))
        return response
    try:
        status("starting", completed=0)
        if args.mode == "pd":
            conn, endpoint = mp.get_context("spawn").Pipe()
            child = mp.get_context("spawn").Process(target=decode_worker,
                                                    args=(endpoint, str(root), args.suite, args.max_tokens))
            child.start()
            if not conn.poll(180):
                raise TimeoutError("HF D model startup timed out")
            ready = conn.recv()
            assert ready.get("ready"), ready
            (outdir / "decode_worker.json").write_text(json.dumps(ready, indent=2))
        set_seed(17)
        torch.set_num_threads(4)
        cfg = upstream_config(args.max_tokens)
        wrapper = ModelWrapper(cfg.model_name, torch.device("cuda:0"), use_vllm=False, args=cfg)
        method = LatentMASMethod(wrapper, latent_steps=40, judger_max_new_tokens=args.max_tokens,
                                 temperature=.6, top_p=.95, generate_bs=2, args=cfg)
        model = wrapper.model
        actual_vocab = {"tokenizer_length": len(wrapper.tokenizer),
                        "input_vocab": model.get_input_embeddings().weight.shape[0],
                        "output_vocab": model.get_output_embeddings().weight.shape[0]}
        if args.mode == "pd":
            assert all(ready[k] == v for k,v in actual_vocab.items()), (actual_vocab, ready)
        (outdir / f"{args.mode}_vocabulary.json").write_text(json.dumps(actual_vocab, indent=2))
        native_forward, native_generate = model.forward, model.generate
        current = {}

        @wraps(native_forward)
        def dispatch_forward(*positional, **kwargs):
            if not current.get("in_judge"):
                return native_forward(*positional, **kwargs)
            assert not positional
            step = current["forward_count"]
            current["forward_count"] += 1
            if step == 0:
                current["prefill_inputs"] = {k: tensor_digest(v) for k,v in kwargs.items()
                                              if isinstance(v, torch.Tensor)}
                current["latent_cache"] = cache_digest(kwargs["past_key_values"])
                result = native_forward(**kwargs)
                current["judge_prefill_cache"] = cache_digest(result.past_key_values)
                return result
            if args.mode == "reference":
                return native_forward(**kwargs)
            payload = dict(kwargs)
            first = step == 1
            shadow = None
            if args.shadow_decode:
                from transformers import DynamicCache
                if first:
                    current["shadow_cache"] = DynamicCache.from_legacy_cache(tuple(
                        tuple(t.clone() for t in layer) for layer in kwargs["past_key_values"].to_legacy_cache()))
                shadow_kwargs = dict(kwargs)
                shadow_kwargs["past_key_values"] = current["shadow_cache"]
                shadow = native_forward(**shadow_kwargs)
                current["shadow_cache"] = shadow.past_key_values
            if not first:
                payload.pop("past_key_values")
            start = time.perf_counter()
            rng_before = torch.cuda.get_rng_state() if args.shadow_decode else None
            response = ask({"command": "forward", "first": first, "kwargs": payload,
                            "expected_cache": current["judge_prefill_cache"] if first else None})
            current["rpc_wall_s"] += time.perf_counter()-start
            current["D_forward_wall_s"] += response["forward_wall_s"]
            if first:
                current["received_cache"] = response["received_cache"]
            if shadow is not None:
                remote = response["outputs"]
                diff = (shadow.logits.float() - remote.logits.float()).abs()
                details = {"step": step, "equal": torch.equal(shadow.logits, remote.logits),
                           "max_abs": diff.max().item(), "unequal": int((diff != 0).sum().item()),
                           "rng_unchanged_by_rpc": torch.equal(rng_before, torch.cuda.get_rng_state()),
                           "local_logits": tensor_digest(shadow.logits), "remote_logits": tensor_digest(remote.logits)}
                if not details["equal"] and not current.get("shadow_mismatch_seen"):
                    details["local_cache"] = cache_digest(shadow.past_key_values)
                    details["remote_cache"] = cache_digest(remote.past_key_values)
                    current["shadow_mismatch_seen"] = True
                with (outdir / "shadow_decode.jsonl").open("a") as f:
                    f.write(json.dumps(details)+"\n")
            return response["outputs"]

        @wraps(native_generate)
        def observed_generate(*positional, **kwargs):
            assert not positional
            current.update(in_judge=True, forward_count=0, rpc_wall_s=0., D_forward_wall_s=0.)
            length = kwargs["input_ids"].shape[1]
            try:
                result = native_generate(**kwargs)
                current["sequences"] = result.sequences.detach().cpu().tolist()
                current["prompt_length"] = length
                return result
            finally:
                current["in_judge"] = False

        if args.verify_observer:
            assert args.mode == "reference"
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            with torch.inference_mode():
                native = method.run_batch(items[batches[0]*2:batches[0]*2+2])
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            model.forward, model.generate = dispatch_forward, observed_generate
            with torch.inference_mode():
                observed = method.run_batch(items[batches[0]*2:batches[0]*2+2])
            assert native == observed, "Observation altered upstream HF results"
            (outdir / "observer_check.json").write_text(json.dumps({"equal": True, "cases": 2}, indent=2))
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            current.clear()
        model.forward, model.generate = dispatch_forward, observed_generate
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        torch.cuda.synchronize()
        start_all = time.perf_counter()
        with torch.inference_mode():
            for done, batch in enumerate(batches):
                current.clear()
                status("running", completed=done*2, batch=batch)
                start = time.perf_counter()
                results = method.run_batch(items[batch*2:batch*2+2])
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                cases = []
                for item, result, sequence in zip(items[batch*2:batch*2+2], results, current["sequences"]):
                    continuation = sequence[current["prompt_length"]:]
                    end = next((i+1 for i,t in enumerate(continuation) if t in eos), len(continuation))
                    finish = "stop" if any(t in eos for t in continuation) else "length"
                    prediction = final_number(result["raw_prediction"])
                    matched = prediction == numeric(item["numeric_gold"])
                    cases.append({"dataset_index": item["dataset_index"], "gold": item["gold"],
                                  "numeric_gold": item["numeric_gold"], "upstream_prediction": result["prediction"],
                                  "upstream_correct": result["correct"], "prediction_numeric": prediction,
                                  "answer_match": matched, "completed_correct": matched and finish == "stop",
                                  "finish_reason": finish, "token_ids": continuation[:end],
                                  "padded_token_ids": continuation, "text": result["raw_prediction"]})
                trace = {k:v for k,v in current.items() if k not in ("sequences", "in_judge", "shadow_cache")}
                if args.mode == "pd":
                    assert trace["received_cache"] == trace["judge_prefill_cache"]
                    ask({"command": "reset"})
                row = {"mode": args.mode, "batch": batch, "end_to_end_s_with_observer": elapsed,
                       "generation": trace, "cases": cases}
                with outfile.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False)+"\n")
                print(json.dumps({"mode": args.mode, "completed": (done+1)*2,
                                  "upstream_correct": sum(c["upstream_correct"] for c in cases),
                                  "completed_correct": sum(c["completed_correct"] for c in cases)}), flush=True)
        status("complete", completed=len(batches)*2, measured_wall_s=time.perf_counter()-start_all)
    except Exception as exc:
        status("failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if child:
            try:
                if child.is_alive():
                    conn.send({"command": "stop"})
                    child.join(20)
            except (BrokenPipeError, EOFError):
                pass
            if child.is_alive():
                child.terminate()
                child.join(10)
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main():
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
