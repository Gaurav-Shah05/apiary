"""Model / training configs and the single argparse shared by train.py, modal_app.py and tests."""
import argparse
import dataclasses
from dataclasses import dataclass


@dataclass
class ModelCfg:
    vocab: int = 49152          # SmolLM2 tokenizer
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    n_experts: int = 64
    top_k: int = 8
    moe_dim: int = 1024         # per-expert SwiGLU hidden size
    seq_len: int = 4096
    rope_theta: float = 1e6
    norm_eps: float = 1e-6
    aux_coef: float = 0.01      # load-balance loss
    z_coef: float = 1e-3        # router z-loss
    init_std: float = 0.02

    @property
    def params(self) -> dict:
        """Parameter counts: total, active (per token), and non-embedding active."""
        attn = self.dim * self.head_dim * (2 * self.n_heads + 2 * self.n_kv_heads) + 2 * self.head_dim
        expert = 3 * self.dim * self.moe_dim
        router = self.n_experts * self.dim
        norms = 2 * self.dim
        layer_total = attn + router + norms + self.n_experts * expert
        layer_active = attn + router + norms + self.top_k * expert
        emb = self.vocab * self.dim
        return dict(
            total=self.n_layers * layer_total + 2 * emb + self.dim,
            active=self.n_layers * layer_active + 2 * emb + self.dim,
            active_nonembed=self.n_layers * layer_active + emb,  # lm_head counts, embed lookup doesn't
        )

    @property
    def flops_per_token(self) -> float:
        """6*N_active + attention-score FLOPs (fwd+bwd, causal not discounted)."""
        attn_scores = 12 * self.n_layers * self.seq_len * self.n_heads * self.head_dim
        return 6 * self.params["active_nonembed"] + attn_scores


@dataclass
class TrainCfg:
    run: str = "main"                  # checkpoint subdir name
    data_dir: str = "/local/tokens"
    ckpt_root: str = "/ckpt"           # volume: uploaded checkpoints, metrics, export
    local_dir: str = "/local/ckpt"     # fast local disk for DCP saves
    seed: int = 0
    micro_batch: int = 8               # sequences per GPU per micro-step (8-way sharding on B200; 4 fits 2xH200)
    global_batch: int = 256            # sequences per optimizer step (all GPUs)
    lr: float = 3e-4
    min_lr_frac: float = 0.0
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 500
    decay_frac: float = 0.2            # last 20% of the time budget decays linearly to min_lr
    time_budget_min: float = 320.0     # node-minutes for the whole run, across restarts (watchdog kills at +10)
    final_reserve_min: float = 8.0     # stop training this many minutes before budget for save+export
    ckpt_every_min: float = 20.0
    keep_ckpts: int = 2
    max_steps: int = 0                 # 0 = unlimited (smoke tests set this)
    log_every: int = 10
    compile: int = 1
    moe_impl: str = "grouped"          # grouped | loop
    attn: str = "cudnn"                # cudnn | flash
    resume: str = "auto"               # auto | none
    export: int = 1                    # write HF checkpoint at the end
    t0: float = 0.0                    # container start epoch (0 = process start)
    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"


PRESETS = {
    "main": ({}, {}),
    "smoke": ({}, dict(run="smoke", micro_batch=4, max_steps=60, time_budget_min=30, final_reserve_min=3, ckpt_every_min=1, global_batch=16, warmup_steps=10)),
    "tiny": (
        dict(vocab=512, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, n_experts=4, top_k=2, moe_dim=32, seq_len=32),
        dict(run="tiny", micro_batch=2, global_batch=4, max_steps=20, warmup_steps=2, compile=0, moe_impl="loop", attn="flash",
             ckpt_every_min=0.05, time_budget_min=5, final_reserve_min=0.5),
    ),
}


def _add(parser: argparse.ArgumentParser, cls, defaults: dict):
    for f in dataclasses.fields(cls):
        default = defaults.get(f.name, f.default)
        typ = type(default) if not isinstance(default, tuple) else lambda s: tuple(float(x) for x in s.split(","))
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=typ, default=default, dest=f.name)


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default="main", choices=list(PRESETS))
    ns, rest = pre.parse_known_args(argv)
    mdef, tdef = PRESETS[ns.preset]
    parser = argparse.ArgumentParser(parents=[pre])
    _add(parser, ModelCfg, mdef)
    _add(parser, TrainCfg, tdef)
    args = vars(parser.parse_args(argv))
    args.pop("preset")
    mnames = {f.name for f in dataclasses.fields(ModelCfg)}
    mcfg = ModelCfg(**{k: v for k, v in args.items() if k in mnames})
    tcfg = TrainCfg(**{k: v for k, v in args.items() if k not in mnames})
    return mcfg, tcfg
