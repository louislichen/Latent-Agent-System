"""Whole-agent profiling controls for fixed P/D versus P/D-colocated agents.

Both use two resident copies of the unchanged HF model on the same GPU. Split
executes every prompt prefill in P and each latent/decode continuation in D.
Mixed executes whole agents on P,D,P,D. Affinity keeps all work in P. These
supplement the original judge-only deployment profile; they are not network
benchmarks or production scheduler implementations.
"""
import argparse
from functools import wraps
import gc
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace
import traceback

from latent_agent_system.profiling import profile_hf_existing as prof


def cpu_kwargs(kwargs):
    import torch
    return {k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v)
            for k,v in kwargs.items() if k!="past_key_values"}


def to_cuda(kwargs):
    import torch
    return {k:(v.to("cuda:0") if isinstance(v,torch.Tensor) else v) for k,v in kwargs.items()}


def worker(conn,root,suite,mode,max_tokens,hashes):
    os.setsid()
    log=open(Path(root)/f"logs/{suite}-D.log","w",buffering=1)
    os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    prof.install(root,suite,"D",False)
    import torch
    from models import ModelWrapper
    from utils import set_seed
    from latent_agent_system.serving.hf_semantic_pd import upstream_config
    set_seed(17);torch.set_num_threads(4)
    cache=None
    try:
        cfg=upstream_config(max_tokens)
        wrapper=ModelWrapper(cfg.model_name,torch.device("cuda:0"),use_vllm=False,args=cfg)
        native_forward=wrapper.model.forward
        native_generate=wrapper.model.generate
        first=False
        captured={}
        @wraps(native_forward)
        def forward(*a,**kw):
            nonlocal first
            if mode=="split" and first:
                first=False
                with prof.REC.scope("rpc.prefill_callback"):
                    conn.send({"callback":"prefill","kwargs":cpu_kwargs(kw)})
                    reply=conn.recv()
                    assert reply.get("ok"),reply
                    return reply["output"]
            return native_forward(*a,**kw)
        wrapper.model.forward=forward
        @wraps(native_generate)
        def generate(*a,**kw):
            out=native_generate(*a,**kw)
            captured["sequences"]=out.sequences.detach().cpu().tolist()
            captured["prompt_length"]=kw["input_ids"].shape[1]
            return out
        wrapper.model.generate=generate
        conn.send({"ready":True,"pid":os.getpid(),"vocab":wrapper.model.get_output_embeddings().weight.shape[0]})
        with torch.inference_mode():
            while True:
                request=conn.recv()
                cmd=request["command"]
                if cmd=="stop":break
                if cmd=="reset":
                    cache=None;request=None;captured.clear();gc.collect();prof.REC.flush()
                    conn.send({"ok":True});continue
                prof.REC.role=request["role"];prof.REC.batch=request["batch"]
                first=mode=="split"
                kw=request["kwargs"]
                if mode=="mixed":
                    cache=request.get("cache")
                elif mode=="split":
                    assert prof.kv_meta(cache)==request["incoming_meta"]
                if hashes and mode=="mixed":
                    assert prof.BASE.cache_digest(cache)==request["cache_hash"]
                kw["past_key_values"]=cache
                start=time.perf_counter()
                if cmd=="latent":
                    cache=wrapper.generate_latent_batch(**kw)
                    torch.cuda.synchronize()
                    reply={"ok":True,"cache":cache,"cache_meta":prof.kv_meta(cache),
                           "wall_s":time.perf_counter()-start}
                    if hashes:reply["cache_hash"]=prof.BASE.cache_digest(cache)
                    conn.send(reply)
                elif cmd=="text":
                    torch.set_rng_state(request["cpu_rng"])
                    torch.cuda.set_rng_state(request["cuda_rng"])
                    texts,cache=wrapper.generate_text_batch(**kw)
                    torch.cuda.synchronize()
                    conn.send({"ok":True,"texts":texts,"captured":dict(captured),
                               "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),
                               "cache_meta":prof.kv_meta(cache),"wall_s":time.perf_counter()-start})
                else:raise ValueError(cmd)
    except Exception as exc:
        try:conn.send({"ok":False,"error":str(exc),"traceback":traceback.format_exc()})
        except Exception:pass
        raise
    finally:
        cache=None;gc.collect();prof.REC.flush()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["split","mixed","affinity"],required=True)
    ap.add_argument("--suite",required=True)
    ap.add_argument("--batches",type=int,nargs="+",default=[0,24,49])
    ap.add_argument("--max-tokens",type=int,default=1024)
    ap.add_argument("--hashes",action="store_true")
    args=ap.parse_args()
    import torch
    import torch.multiprocessing as mp
    root=Path(os.environ["LATENT_SERVING_ROOT"])
    (root / "logs").mkdir(parents=True, exist_ok=True)
    out=root/"results"/args.suite;out.mkdir(parents=True,exist_ok=True)
    (root/"logs"/str(Path(args.suite).parent)).mkdir(parents=True,exist_ok=True)
    output=out/(args.mode+".jsonl")
    if output.exists():raise RuntimeError("Refusing to overwrite route profile")
    os.environ["PROFILE_HASHES"]="1" if args.hashes else "0"
    prof.install(str(root),args.suite,"P",False)
    from models import ModelWrapper
    from methods.latent_mas import LatentMASMethod
    from utils import set_seed,extract_gold,normalize_answer
    from latent_agent_system.serving.hf_semantic_pd import upstream_config
    from latent_agent_system.data.answer_parsing import final_number,numeric
    samples = root / "samples.json"
    if not samples.is_file():
        samples = root / "results/compare100_v1/samples.json"
    items = json.loads(samples.read_text())
    for item in items:
        item["numeric_gold"]=item["gold"]
        item["gold"]=normalize_answer(extract_gold(item["solution"]))
        item["question"]=item["question"].strip()
    config={**vars(args),"visible_gpu":os.environ["CUDA_VISIBLE_DEVICES"],"batch_size":2,"latent_steps":40,
            "seed":17,"temperature":.6,"top_p":.95,"resident_models":2,
            "scope":"same-GPU CUDA-IPC whole-agent route control; logical bytes are not network traffic"}
    (out/(args.mode+"_config.json")).write_text(json.dumps(config,indent=2))
    def status(stage,**extra):
        (out/(args.mode+"_status.json")).write_text(json.dumps({"stage":stage,**extra},indent=2))
    ctx=mp.get_context("spawn");conn,endpoint=ctx.Pipe()
    child=ctx.Process(target=worker,args=(endpoint,str(root),args.suite,args.mode,args.max_tokens,args.hashes))
    current={"cache":None,"handoffs":[],"captured":{},"stage_states":[]}
    try:
        status("starting",completed=0);child.start()
        if not conn.poll(240):raise TimeoutError("route D startup")
        ready=conn.recv();assert ready.get("ready"),ready
        (out/(args.mode+"_worker.json")).write_text(json.dumps(ready,indent=2))
        set_seed(17);torch.set_num_threads(4)
        cfg=upstream_config(args.max_tokens)
        wrapper=ModelWrapper(cfg.model_name,torch.device("cuda:0"),use_vllm=False,args=cfg)
        method=LatentMASMethod(wrapper,latent_steps=40,judger_max_new_tokens=args.max_tokens,
                               temperature=.6,top_p=.95,generate_bs=2,args=cfg)
        native_latent=wrapper.generate_latent_batch
        native_text=wrapper.generate_text_batch
        native_forward=wrapper.model.forward
        native_generate=wrapper.model.generate
        def handoff(direction,cache,reason):
            current["handoffs"].append({"direction":direction,"reason":reason,
                                        "role":prof.REC.role,**prof.kv_meta(cache)})
        def ask(request):
            with prof.REC.scope("rpc.agent_stage",command=request["command"]):
                with prof.REC.scope("ipc.send"):conn.send(request)
                while True:
                    with prof.REC.scope("ipc.wait_recv"):
                        if not conn.poll(180):raise TimeoutError("route stage timed out")
                        response=conn.recv()
                    if response.get("callback")=="prefill":
                        kw=to_cuda(response["kwargs"])
                        kw["past_key_values"]=current["cache"]
                        with prof.REC.scope("route.prefill_in_P"):
                            result=native_forward(**kw)
                        handoff("P_to_D",result.past_key_values,"prefill_result")
                        from transformers.modeling_outputs import CausalLMOutputWithPast
                        if kw.get("output_hidden_states"):
                            # Upstream latent generator reads only last positions
                            # of hidden_states[0] and [-1]; transport that exact data.
                            hs=(result.hidden_states[0][:,-1:,:].contiguous(),
                                result.hidden_states[-1][:,-1:,:].contiguous())
                            result=CausalLMOutputWithPast(past_key_values=result.past_key_values,hidden_states=hs)
                        with prof.REC.scope("ipc.send_prefill"):conn.send({"ok":True,"output":result})
                        continue
                    if not response.get("ok"):raise RuntimeError(json.dumps(response))
                    return response
        @wraps(native_latent)
        def latent(input_ids,**kw):
            role=prof.REC.role
            remote=args.mode=="split" or (args.mode=="mixed" and role=="critic")
            if not remote:
                cache=native_latent(input_ids,**kw)
                current["stage_states"].append({"role":role,"state":prof.BASE.cache_digest(cache) if args.hashes else prof.kv_meta(cache)})
                return cache
            current["cache"]=kw.pop("past_key_values",None)
            request={"command":"latent","role":role,"batch":prof.REC.batch,
                     "kwargs":{"input_ids":input_ids,**kw},"incoming_meta":prof.kv_meta(current["cache"])}
            if args.mode=="mixed":
                handoff("P_to_D",current["cache"],"next_agent")
                request["cache"]=current["cache"]
                if args.hashes:request["cache_hash"]=prof.BASE.cache_digest(current["cache"])
            response=ask(request)
            cache=response["cache"]
            handoff("D_to_P",cache,"latent_complete_next_prefill")
            if args.hashes:assert prof.BASE.cache_digest(cache)==response["cache_hash"]
            current["cache"]=cache
            current["stage_states"].append({"role":role,"state":prof.BASE.cache_digest(cache) if args.hashes else prof.kv_meta(cache)})
            return cache
        wrapper.generate_latent_batch=latent
        @wraps(native_generate)
        def generate(*a,**kw):
            result=native_generate(*a,**kw)
            current["captured"]={"sequences":result.sequences.detach().cpu().tolist(),
                                  "prompt_length":kw["input_ids"].shape[1]}
            return result
        wrapper.model.generate=generate
        @wraps(native_text)
        def text(input_ids,attention_mask=None,**kw):
            if args.mode=="affinity":return native_text(input_ids,attention_mask,**kw)
            current["cache"]=kw.pop("past_key_values",None)
            request={"command":"text","role":"judger","batch":prof.REC.batch,
                     "kwargs":{"input_ids":input_ids,"attention_mask":attention_mask,**kw},
                     "incoming_meta":prof.kv_meta(current["cache"]),
                     "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state()}
            if args.mode=="mixed":
                handoff("P_to_D",current["cache"],"next_agent")
                request["cache"]=current["cache"]
                if args.hashes:request["cache_hash"]=prof.BASE.cache_digest(current["cache"])
            response=ask(request)
            torch.set_rng_state(response["cpu_rng"]);torch.cuda.set_rng_state(response["cuda_rng"])
            current["captured"]=response["captured"]
            # Final cache remains owned by D. run_batch discards this return;
            # preserve its length as an opaque owner handle for observation.
            return response["texts"],SimpleNamespace(remote_owner=ready["pid"],**response["cache_meta"])
        wrapper.generate_text_batch=text
        eos=wrapper.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        with torch.inference_mode():
            for done,batch in enumerate(args.batches):
                status("running",completed=done*2,batch=batch)
                current.update(cache=None,handoffs=[],captured={},stage_states=[])
                start=time.perf_counter()
                results=method.run_batch(items[batch*2:batch*2+2])
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                captured=current["captured"];cases=[]
                for item,result,sequence in zip(items[batch*2:batch*2+2],results,captured["sequences"]):
                    ids=sequence[captured["prompt_length"]:]
                    end=next((i+1 for i,t in enumerate(ids) if t in eos),len(ids))
                    finish="stop" if any(t in eos for t in ids) else "length"
                    pred=final_number(result["raw_prediction"]);correct=pred==numeric(item["numeric_gold"])
                    cases.append({"dataset_index":item["dataset_index"],"gold":item["gold"],
                                  "numeric_gold":item["numeric_gold"],"upstream_prediction":result["prediction"],
                                  "upstream_correct":result["correct"],"prediction_numeric":pred,
                                  "answer_match":correct,"completed_correct":correct and finish=="stop",
                                  "finish_reason":finish,"token_ids":ids[:end],"padded_token_ids":ids,
                                  "text":result["raw_prediction"]})
                row={"mode":args.mode,"batch":batch,"wall_s":elapsed,"cases":cases,
                     "handoffs":list(current["handoffs"]),"latent_stage_states":list(current["stage_states"])}
                with output.open("a") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
                current["cache"]=None;results=None
                conn.send({"command":"reset"});assert conn.recv().get("ok")
                print(json.dumps({"mode":args.mode,"completed":(done+1)*2,"wall_s":elapsed,
                                  "state_handoffs":len(row["handoffs"])}),flush=True)
        status("complete",completed=len(args.batches)*2)
    except Exception as exc:
        status("failed",error=str(exc),traceback=traceback.format_exc());raise
    finally:
        try:
            if child.is_alive():conn.send({"command":"stop"});child.join(20)
        except (EOFError,BrokenPipeError):pass
        if child.is_alive():child.terminate();child.join(10)
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        prof.REC.flush()


if __name__=="__main__":main()
