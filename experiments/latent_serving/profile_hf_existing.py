"""Profile the previously validated HF deployments without changing their method.

Run the frozen hf_semantic_pd.main with timing hooks. CPU intervals are nested;
CUDA-event spans include stream idle gaps, so they are not called kernel time.
Bounded Kineto windows are collected in a separate pass from timing runs.
"""
import argparse
import atexit
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REC = None
BASE = None
HASH_NATIVE = None
DECODER_NATIVE = None


def kv_meta(cache):
    if cache is None:
        return {"positions": 0, "bytes": 0, "layers": 0}
    layers = cache.to_legacy_cache()
    return {"positions": int(cache.get_seq_length()), "layers": len(layers),
            "bytes": sum(t.numel()*t.element_size() for layer in layers for t in layer)}


class Recorder:
    def __init__(self, root, suite, side, trace_windows):
        import torch
        self.torch = torch
        self.side = side
        self.out = Path(root)/"results"/suite/"profile"
        self.out.mkdir(parents=True, exist_ok=True)
        self.batch = -1
        self.role = "judger" if side == "D" else "outside"
        self.phase = "outside"
        self.events = []
        self.stack = []
        self.counter = 0
        self.agent_index = -1
        self.trace_windows = trace_windows
        self.captured = set()

    @contextmanager
    def scope(self, name, cuda=False, **meta):
        self.counter += 1
        item = {"id": self.counter, "parent": self.stack[-1] if self.stack else None,
                "name": name, "role": self.role, "phase": self.phase, "batch": self.batch,
                "pid": os.getpid(), "side": self.side, "start_ns": time.perf_counter_ns(), **meta}
        pair = None
        if cuda:
            pair = (self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True))
            pair[0].record()
        self.stack.append(item["id"])
        try:
            yield item
        finally:
            if pair:
                pair[1].record()
                item["_cuda_pair"] = pair
            item["end_ns"] = time.perf_counter_ns()
            self.stack.pop()
            self.events.append(item)

    def memory(self):
        return {"allocated": self.torch.cuda.memory_allocated(),
                "reserved": self.torch.cuda.memory_reserved(),
                "peak_allocated": self.torch.cuda.max_memory_allocated()}

    def flush(self):
        if not self.events:
            return
        # Flush outside measured model stages. Existing batch synchronization
        # already completes the model work before the next batch begins.
        self.torch.cuda.synchronize()
        pending, self.events = self.events, []
        for item in pending:
            pair = item.pop("_cuda_pair", None)
            if pair:
                item["cuda_stream_span_ms"] = pair[0].elapsed_time(pair[1])
        with (self.out/f"events-{self.side}-{os.getpid()}.jsonl").open("a") as f:
            for item in pending:
                f.write(json.dumps(item)+"\n")

    def capture(self, key, fn, *args, **kwargs):
        if not self.trace_windows or key in self.captured:
            return fn(*args, **kwargs)
        self.captured.add(key)
        from torch.profiler import profile, ProfilerActivity, record_function
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True) as p:
            with record_function("profile/"+key):
                result = fn(*args, **kwargs)
        stem = f"kineto-{self.side}-{key}"
        p.export_chrome_trace(str(self.out/(stem+".json")))
        summaries = []
        for e in p.key_averages(group_by_input_shape=True):
            summaries.append({"name": e.key, "count": e.count,
                              "cpu_total_us": e.cpu_time_total,
                              "cpu_self_us": e.self_cpu_time_total,
                              "device_total_us": getattr(e, "device_time_total", 0),
                              "device_self_us": getattr(e, "self_device_time_total", 0),
                              "shapes": str(e.input_shapes)})
        (self.out/(stem+"-ops.json")).write_text(json.dumps(summaries, indent=2))
        return result


class TimedPipe:
    def __init__(self, conn):
        self.conn = conn

    def send(self, value):
        if REC is None:
            return self.conn.send(value)
        command = value.get("command", "response") if isinstance(value, dict) else "other"
        with REC.scope("ipc.send", command=command):
            return self.conn.send(value)

    def poll(self, timeout=0):
        if REC is None:
            return self.conn.poll(timeout)
        with REC.scope("ipc.poll"):
            return self.conn.poll(timeout)

    def recv(self):
        if REC is None:
            return self.conn.recv()
        with REC.scope("ipc.recv"):
            value = self.conn.recv()
        if REC.side == "D" and isinstance(value, dict):
            if value.get("first"):
                REC.batch += 1
            if value.get("command") in ("reset", "stop"):
                REC.flush()
        return value

    def __getattr__(self, name):
        # The unpickler must not recurse before conn has been restored.
        conn = self.__dict__.get("conn")
        if conn is None:
            raise AttributeError(name)
        return getattr(conn, name)


def install(root, suite, side, trace_windows):
    global REC, BASE, HASH_NATIVE, DECODER_NATIVE
    import torch
    sys.path.insert(0, str(Path(root)/"sources/LatentMAS"))
    import models
    import hf_semantic_pd as base
    BASE = base
    DECODER_NATIVE = base.decode_worker
    REC = Recorder(root, suite, side, trace_windows)
    native_init = models.ModelWrapper.__init__

    @wraps(native_init)
    def initialized(wrapper, *args, **kwargs):
        native_init(wrapper, *args, **kwargs)
        m = wrapper.model
        model_forward = m.forward
        base_forward = m.model.forward
        head_forward = m.lm_head.forward

        @wraps(base_forward)
        def base_fwd(*a, **kw):
            with REC.scope("transformer", cuda=True):
                return base_forward(*a, **kw)

        @wraps(head_forward)
        def head_fwd(*a, **kw):
            with REC.scope("lm_head", cuda=True) as event:
                result = head_forward(*a, **kw)
                event["output_bytes"] = result.numel()*result.element_size()
                event["output_shape"] = list(result.shape)
                return result

        @wraps(model_forward)
        def fwd(*a, **kw):
            previous = REC.phase
            if kw.get("inputs_embeds") is not None:
                phase = "latent_decode"
                tokens = kw["inputs_embeds"].shape[1]
            else:
                ids = kw.get("input_ids", a[0] if a else None)
                tokens = ids.shape[1] if ids is not None else -1
                phase = "prefill" if tokens > 1 else "text_decode"
            REC.phase = phase
            key = REC.role+"-"+phase
            try:
                with REC.scope("model.forward", cuda=True, tokens=int(tokens)) as event:
                    if os.environ.get("PROFILE_SKIP_LATENT_HEAD")=="1" and REC.role!="judger":
                        # Original latent stages use hidden_states and cache,
                        # not vocabulary logits. Keep the original transformer
                        # operations, input state and alignment unchanged.
                        from transformers.modeling_outputs import CausalLMOutputWithPast
                        assert "labels" not in kw and "logits_to_keep" not in kw
                        inner=m.model(*a,**kw)
                        result=CausalLMOutputWithPast(past_key_values=inner.past_key_values,
                                                     hidden_states=inner.hidden_states,
                                                     attentions=inner.attentions)
                        event["unused_latent_head_skipped"]=True
                    else:
                        result = REC.capture(key, model_forward, *a, **kw)
                    event["kv_after"] = kv_meta(result.past_key_values)
                    return result
            finally:
                REC.phase = previous

        m.model.forward, m.lm_head.forward, m.forward = base_fwd, head_fwd, fwd
        prepare = wrapper.prepare_chat_batch

        @wraps(prepare)
        def prep(*a, **kw):
            REC.agent_index += 1
            idx = REC.agent_index % 4
            if idx == 0:
                REC.flush()
                REC.batch += 1
            REC.role = ["planner", "critic", "refiner", "judger"][idx]
            with REC.scope("prepare_chat"):
                return prepare(*a, **kw)
        wrapper.prepare_chat_batch = prep

        latent = wrapper.generate_latent_batch
        @wraps(latent)
        def lat(*a, **kw):
            with REC.scope("latent.stage", input_kv=kv_meta(kw.get("past_key_values")),
                           memory_before=REC.memory()) as event:
                result = latent(*a, **kw)
                event["output_kv"] = kv_meta(result)
                event["memory_after"] = REC.memory()
                return result
        wrapper.generate_latent_batch = lat

        align = wrapper._apply_latent_realignment
        @wraps(align)
        def ali(*a, **kw):
            with REC.scope("latent.alignment", cuda=True):
                return align(*a, **kw)
        wrapper._apply_latent_realignment = ali

        text = wrapper.generate_text_batch
        @wraps(text)
        def txt(*a, **kw):
            with REC.scope("judge.stage", input_kv=kv_meta(kw.get("past_key_values")),
                           memory_before=REC.memory()) as event:
                result = text(*a, **kw)
                event["memory_after"] = REC.memory()
                return result
        wrapper.generate_text_batch = txt
    models.ModelWrapper.__init__ = initialized

    HASH_NATIVE = base.cache_digest
    @wraps(HASH_NATIVE)
    def digest(cache):
        with REC.scope("validation.cache_hash"):
            if os.environ.get("PROFILE_HASHES", "1") == "0":
                return {**kv_meta(cache), "validation_disabled": True}
            return HASH_NATIVE(cache)
    base.cache_digest = digest

    multinomial = torch.multinomial
    @wraps(multinomial)
    def sample(*a, **kw):
        if REC.role != "judger":
            return multinomial(*a, **kw)
        with REC.scope("generation.sample", cuda=True):
            return multinomial(*a, **kw)
    torch.multinomial = sample


def profiled_decoder(conn, root, suite, max_tokens):
    install(root, suite, "D", os.environ.get("PROFILE_KINETO") == "1")
    try:
        return DECODER_NATIVE(conn, root, suite, max_tokens)
    finally:
        REC.flush()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite", required=True)
    known, _ = parser.parse_known_args()
    root = os.environ["LATENT_SERVING_ROOT"]
    install(root, known.suite, "P", os.environ.get("PROFILE_KINETO") == "1")
    import torch.multiprocessing as mp
    native_context = mp.get_context
    class ContextProxy:
        def __init__(self, actual): self.actual = actual
        def Pipe(self, *args, **kwargs):
            a,b = self.actual.Pipe(*args, **kwargs)
            return TimedPipe(a), TimedPipe(b)
        def __getattr__(self, name): return getattr(self.actual, name)
    mp.get_context = lambda *a, **kw: ContextProxy(native_context(*a, **kw))
    BASE.decode_worker = profiled_decoder
    # Keep a stable reference: BASE.decode_worker now points at this wrapper.
    try:
        BASE.main()
    finally:
        REC.flush()


if __name__ == "__main__":
    main()
