"""torchrun entry: FSDP2 + torch.compile pretraining with a node-time budget, async DCP checkpoints and HF export."""
import json
import os
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

import export
from configs import ModelCfg, TrainCfg, parse_args
from data import TokenLoader
from model import Qwen3Moe, moe_selftest

PEAK_FLOPS = {"B200": 2.25e15, "H200": 989e12, "H100": 989e12}  # dense BF16


class AppState(Stateful):
    def __init__(self, model, optim, state: dict):
        self.model, self.optim, self.state = model, optim, state

    def state_dict(self):
        m, o = get_state_dict(self.model, self.optim)
        return {"model": m, "optim": o, "state": dict(self.state)}

    def load_state_dict(self, sd):
        set_state_dict(self.model, self.optim, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
        self.state.update(sd["state"])


class Checkpointer:
    """Async DCP saves to fast local disk (<local>/step_N), then a background upload of each finished checkpoint to the
    volume (<run_dir>/step_N + latest.txt) — volume writes are ~100 MB/s per writer, far too slow to block training on."""

    def __init__(self, app: AppState, run_dir: Path, local_dir: Path, rank: int, keep: int):
        self.app, self.run_dir, self.local, self.rank, self.keep = app, run_dir, local_dir, rank, keep
        self.local.mkdir(parents=True, exist_ok=True)
        self.pg = dist.new_group(backend="gloo") if dist.get_world_size() > 1 else None
        self.pending, self.upload, self.newest = None, None, None
        self.pool = ThreadPoolExecutor(1)  # serial uploads; stale ones are skipped

    def save(self, step: int, sync: bool = False):
        self.finalize()
        tmp = self.local / f"step_{step:07d}.tmp"
        self.pending = (dcp.async_save({"app": self.app}, checkpoint_id=str(tmp), process_group=self.pg), tmp)
        if sync:
            self.finalize()

    def finalize(self):
        if self.pending is None:
            return
        fut, tmp = self.pending
        fut.result()
        self.pending = None
        if self.rank != 0:
            return
        final = tmp.with_suffix("")
        os.rename(tmp, final)
        for old in sorted(self.local.glob("step_[0-9]*"))[: -self.keep]:
            shutil.rmtree(old, ignore_errors=True)
        self.newest = final
        self.upload = self.pool.submit(self._upload, final)

    def _upload(self, src: Path):
        if src != self.newest:  # a newer checkpoint is already queued
            return
        t = time.time()
        dst_tmp = self.run_dir / (src.name + ".tmp")
        shutil.rmtree(dst_tmp, ignore_errors=True)
        dst_tmp.mkdir(parents=True)
        files = [f for f in src.iterdir() if f.is_file()]
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda f: shutil.copyfile(f, dst_tmp / f.name), files))
        os.rename(dst_tmp, self.run_dir / src.name)
        (self.run_dir / "latest.txt").write_text(src.name)
        for old in sorted(self.run_dir.glob("step_[0-9]*"))[: -self.keep]:
            if old.suffix != ".tmp":
                shutil.rmtree(old, ignore_errors=True)
        gb = sum(f.stat().st_size for f in files) / 2**30
        print(f"uploaded {src.name}: {gb:.1f} GB in {time.time() - t:.0f}s", flush=True)

    def wait_upload(self, timeout: float):
        if self.rank == 0 and self.upload is not None:
            try:
                self.upload.result(timeout=timeout)
            except Exception as e:  # timeout: the previous uploaded checkpoint stays valid
                print(f"upload not finished: {e}", flush=True)

    def load_latest(self) -> bool:
        latest = self.run_dir / "latest.txt"
        if not latest.exists():
            return False
        dcp.load({"app": self.app}, checkpoint_id=str(self.run_dir / latest.read_text().strip()))
        return True


def setup_attention(tcfg: TrainCfg, device):
    if device.type != "cuda":
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F
    q, k, v = (torch.randn(2, 16, 512, 128, device=device, dtype=torch.bfloat16) for _ in range(3))
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q, k[:, :8], v[:, :8], is_causal=True, enable_gqa=True)
    torch.backends.cuda.enable_cudnn_sdp(tcfg.attn == "cudnn")
    torch.backends.cuda.enable_flash_sdp(tcfg.attn == "flash")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    out = F.scaled_dot_product_attention(q, k[:, :8], v[:, :8], is_causal=True, enable_gqa=True)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 5e-2, f"{tcfg.attn} SDPA mismatch vs math: {err}"


def main(argv=None):
    mcfg, tcfg = parse_args(argv)
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
    t0 = tcfg.t0 or time.time()
    run_dir = Path(tcfg.ckpt_root) / tcfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None

    def log(**kw):
        if rank == 0:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)
            log_f.write(json.dumps(kw) + "\n"), log_f.flush()

    peak = next((v for k, v in PEAK_FLOPS.items() if cuda and k in torch.cuda.get_device_name(device)), 0.0)
    log(event="start", world=world, device=torch.cuda.get_device_name(device) if cuda else "cpu", **{f"params_{k}": v for k, v in mcfg.params.items()},
        gflop_per_token=mcfg.flops_per_token / 1e9, cfg=json.dumps(vars(tcfg)))

    setup_attention(tcfg, device)
    torch.manual_seed(tcfg.seed)  # identical init on all ranks; FSDP shards each rank's copy
    with torch.device(device):
        model = Qwen3Moe(mcfg, tcfg.moe_impl)
    if cuda and tcfg.moe_impl == "grouped":
        log(event="moe_selftest", grouped_fwd_ms=moe_selftest(model.model.layers[0].mlp, mcfg.dim, device))

    if cuda:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        if tcfg.compile:
            torch._logging.set_logs(recompiles=True)
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for i, blk in enumerate(model.model.layers):
            if tcfg.compile:
                blk = torch.compile(blk, fullgraph=True)
                model.model.layers[i] = blk
            fully_shard(blk, mp_policy=mp, reshard_after_forward=False)
        fully_shard(model, mp_policy=mp, reshard_after_forward=False)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "embed_tokens" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "embed_tokens" not in n)]
    optim = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                              lr=tcfg.lr, betas=tuple(tcfg.betas), eps=1e-8, fused=cuda)
    loader = TokenLoader(tcfg.data_dir, mcfg.seq_len, tcfg.micro_batch, tcfg.global_batch, rank, world)
    accum = tcfg.global_batch // (tcfg.micro_batch * world)
    tokens_per_step = tcfg.global_batch * mcfg.seq_len
    log(event="data", shards=len(loader.mm), tokens=loader.n_tokens, windows=loader.n_windows, accum=accum, tokens_per_step=tokens_per_step)

    state = dict(step=0, tokens=0, used_min=0.0)
    app = AppState(model, optim, state)
    ckpt = Checkpointer(app, run_dir, Path(tcfg.local_dir) / tcfg.run, rank, tcfg.keep_ckpts)
    if tcfg.resume == "auto" and ckpt.load_latest():
        budget = run_dir / "budget.json"
        if budget.exists():
            state["used_min"] = max(state["used_min"], json.loads(budget.read_text())["used_min"])
        log(event="resumed", **state)
    used_at_load = state["used_min"]
    used = lambda: used_at_load + (time.time() - t0) / 60

    def lr_at(step, frac):
        warm = min(1.0, (step + 1) / max(1, tcfg.warmup_steps))
        dec = 1.0 if frac < 1 - tcfg.decay_frac else max(tcfg.min_lr_frac, (1 - frac) / tcfg.decay_frac)
        return tcfg.lr * warm * dec

    stop_at = tcfg.time_budget_min - tcfg.final_reserve_min
    last_ckpt, last_budget, t_log, log_tokens, tps_hist, nan_streak = used(), 0.0, time.time(), 0, [], 0
    model.train()
    while not (tcfg.max_steps and state["step"] >= tcfg.max_steps) and used() < stop_at:
        step = state["step"]
        lr = lr_at(step, used() / tcfg.time_budget_min)
        for g in optim.param_groups:
            g["lr"] = lr
        acc = torch.zeros(3, device=device)
        counts_acc = None
        for m in range(accum):
            x, y = loader.batch(step, m)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            ce, aux, z, counts = model(x, y)
            ((ce + mcfg.aux_coef * aux + mcfg.z_coef * z) / accum).backward()
            acc += torch.stack([ce.detach(), aux.detach(), z.detach()]) / accum
            counts_acc = counts if counts_acc is None else counts_acc + counts
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip, foreach=True)
        gnorm = gnorm.full_tensor() if isinstance(gnorm, DTensor) else gnorm
        if not bool(torch.isfinite(acc[0]) & torch.isfinite(gnorm)):
            nan_streak += 1
            optim.zero_grad(set_to_none=True)
            log(event="nonfinite", step=step, ce=acc[0].item(), gnorm=gnorm.item(), streak=nan_streak)
            assert nan_streak < 3, "3 consecutive non-finite steps"
            state["step"] += 1
            continue
        nan_streak = 0
        optim.step()
        optim.zero_grad(set_to_none=True)
        state["step"] += 1
        state["tokens"] += tokens_per_step
        log_tokens += tokens_per_step

        if step % tcfg.log_every == 0 or step < 5:
            if cuda:
                torch.cuda.synchronize()
            dt = time.time() - t_log
            tps = log_tokens / dt
            tps_hist.append(tps)
            load = counts_acc / counts_acc.sum(-1, keepdim=True) * mcfg.n_experts  # 1.0 = perfect balance
            mem = torch.cuda.max_memory_allocated(device) / 2**30 if cuda else 0.0
            log(event="step", step=step, ce=acc[0].item(), aux=acc[1].item() / mcfg.n_layers, z=acc[2].item() / mcfg.n_layers, lr=lr,
                gnorm=gnorm.item(), tok_s=tps, mfu=(tps / world * mcfg.flops_per_token / peak) if peak else 0.0,
                load_max=load.max().item(), load_min=load.min().item(), mem_gb=mem, tokens=state["tokens"], used_min=used(),
                budget_frac=used() / tcfg.time_budget_min)
            if step == 3 and cuda:
                total = torch.cuda.get_device_properties(device).total_memory / 2**30
                assert mem < 0.92 * total, f"peak memory {mem:.1f} GB too close to {total:.1f} GB"
            if len(tps_hist) > 5 and tps < 0.6 * statistics.median(tps_hist[-20:]):
                log(event="slow", step=step, tok_s=tps, median=statistics.median(tps_hist[-20:]))
            t_log, log_tokens = time.time(), 0
        if rank == 0 and time.time() - last_budget > 60:
            state["used_min"] = used()
            (run_dir / "budget.json").write_text(json.dumps(dict(used_min=used(), step=state["step"])))
            last_budget = time.time()
        if used() - last_ckpt >= tcfg.ckpt_every_min:
            state["used_min"] = used()
            ckpt.save(state["step"])
            last_ckpt = used()
            log(event="ckpt", step=state["step"], used_min=used())

    state["used_min"] = used()
    ckpt.save(state["step"], sync=True)
    log(event="final_ckpt", step=state["step"], used_min=used())
    summary = dict(step=state["step"], tokens=state["tokens"], used_min=used(), tok_s_median=statistics.median(tps_hist) if tps_hist else 0)
    if tcfg.export:
        export_dir = run_dir / "export"
        gen = torch.Generator().manual_seed(1234)
        x = torch.randint(0, mcfg.vocab, (tcfg.micro_batch, mcfg.seq_len), generator=gen).to(device)
        logits = model(x).detach()  # same shapes/modes as training -> no recompile
        sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            export_dir.mkdir(exist_ok=True)
            torch.save({"tokens": x[:1, :256].cpu(), "logits": logits[0, :256].float().cpu()}, export_dir / "parity.pt")
            export.write_hf(sd, mcfg, export_dir, tcfg.tokenizer)
            summary["export"] = str(export_dir)
        log(event="exported", dir=str(export_dir))
    ckpt.wait_upload(timeout=max(60.0, 60 * (tcfg.time_budget_min + 5 - used())))
    if rank == 0:
        summary["used_min"] = used()
        (run_dir / "budget.json").write_text(json.dumps(dict(used_min=used(), step=state["step"])))
        (run_dir / "summary.json").write_text(json.dumps(summary))
        log(event="done", **summary)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
