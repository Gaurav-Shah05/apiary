"""Qwen3-MoE-compatible decoder: RMSNorm, HF-style RoPE, GQA with QK-norm, SDPA, fine-grained top-k MoE.
Module/parameter names match HF `Qwen3MoeForCausalLM` (fused expert tensors as in transformers v5)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import ModelCfg


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


def rope_cache(seq_len: int, head_dim: int, theta: float):
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv)
    emb = torch.cat([freqs, freqs], -1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):  # x: (B, T, H, hd); cos/sin: (T, hd)
    x1, x2 = x.chunk(2, -1)
    rot = torch.cat([-x2, x1], -1)
    return x * cos[None, :, None, :].to(x.dtype) + rot * sin[None, :, None, :].to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
        self.q_proj = nn.Linear(c.dim, c.n_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.n_heads * c.head_dim, c.dim, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.norm_eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = apply_rope(self.q_norm(self.q_proj(x).view(B, T, self.nh, self.hd)), cos, sin)
        k = apply_rope(self.k_norm(self.k_proj(x).view(B, T, self.nkv, self.hd)), cos, sin)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, -1))


class Experts(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(c.n_experts, 2 * c.moe_dim, c.dim))  # gate rows then up rows
        self.down_proj = nn.Parameter(torch.empty(c.n_experts, c.dim, c.moe_dim))


class MoE(nn.Module):
    """Softmax router, top-k with renormalization, SwiGLU experts, Switch load-balance loss + router z-loss."""

    def __init__(self, c: ModelCfg, impl: str):
        super().__init__()
        self.E, self.K, self.impl = c.n_experts, c.top_k, impl
        self.gate = nn.Linear(c.dim, c.n_experts, bias=False)
        self.experts = Experts(c)

    def forward(self, x):
        B, T, D = x.shape
        x = x.reshape(-1, D)
        logits = F.linear(x.float(), self.gate.weight.float())
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.K, -1)
        topv = (topv / topv.sum(-1, keepdim=True)).to(x.dtype)
        out = self._grouped(x, topi, topv) if self.impl == "grouped" else self._loop(x, topi, topv)
        counts = F.one_hot(topi, self.E).sum((0, 1)).float()  # static shape (E,), compile-friendly
        aux = self.E * (counts / topi.numel() * probs.mean(0)).sum()  # == 1 at perfect balance
        z = torch.logsumexp(logits, -1).pow(2).mean()
        return out.view(B, T, D), aux, z, counts

    def _grouped(self, x, topi, topv):
        flat = topi.reshape(-1)
        order = torch.argsort(flat, stable=True)  # static (T*K,) — only `offs` is data-dependent
        tok = order // self.K
        offs = F.one_hot(flat, self.E).sum(0).cumsum(0).to(torch.int32)
        w = self.experts
        h = F.grouped_mm(x[tok].to(torch.bfloat16), w.gate_up_proj.to(torch.bfloat16).transpose(-2, -1), offs=offs)
        g, u = h.chunk(2, -1)
        h = F.grouped_mm(F.silu(g) * u, w.down_proj.to(torch.bfloat16).transpose(-2, -1), offs=offs)
        h = h.to(x.dtype) * topv.reshape(-1)[order, None]
        return torch.zeros_like(x).index_add_(0, tok, h)

    def _loop(self, x, topi, topv):  # eager reference / CPU fallback
        out = torch.zeros_like(x)
        for e in range(self.E):
            rows, slot = (topi == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            g, u = (x[rows] @ self.experts.gate_up_proj[e].T).chunk(2, -1)
            h = (F.silu(g) * u) @ self.experts.down_proj[e].T
            out.index_add_(0, rows, h * topv[rows, slot, None])
        return out


class Block(nn.Module):
    def __init__(self, c: ModelCfg, moe_impl: str):
        super().__init__()
        self.input_layernorm = RMSNorm(c.dim, c.norm_eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.dim, c.norm_eps)
        self.mlp = MoE(c, moe_impl)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        h, aux, z, counts = self.mlp(self.post_attention_layernorm(x))
        return x + h, aux, z, counts


class Decoder(nn.Module):
    def __init__(self, c: ModelCfg, moe_impl: str):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab, c.dim)
        self.layers = nn.ModuleList(Block(c, moe_impl) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.dim, c.norm_eps)


class Qwen3Moe(nn.Module):
    def __init__(self, c: ModelCfg, moe_impl: str = "grouped"):
        super().__init__()
        self.cfg = c
        self.model = Decoder(c, moe_impl)
        self.lm_head = nn.Linear(c.dim, c.vocab, bias=False)
        cos, sin = rope_cache(c.seq_len, c.head_dim, c.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        std = self.cfg.init_std
        for name, p in self.named_parameters():
            if p.dim() == 1:  # norm weights stay 1
                continue
            s = std / math.sqrt(2 * self.cfg.n_layers) if name.endswith(("o_proj.weight", "down_proj")) else std
            nn.init.trunc_normal_(p, mean=0.0, std=s, a=-3 * s, b=3 * s)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        cos, sin = self.cos[:T], self.sin[:T]
        x = self.model.embed_tokens(idx)
        auxs, zs, counts = [], [], []
        for blk in self.model.layers:
            x, a, z, c = blk(x, cos, sin)
            auxs.append(a), zs.append(z), counts.append(c)
        logits = self.lm_head(self.model.norm(x))
        if targets is None:
            return logits
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return ce, torch.stack(auxs).sum(), torch.stack(zs).sum(), torch.stack(counts)


@torch.no_grad()
def _grad_check(moe: MoE, x, impl: str):
    moe.impl = impl
    with torch.enable_grad():
        moe.zero_grad(set_to_none=True)
        out = moe(x)[0]
        out.float().pow(2).mean().backward()
    return out, moe.experts.gate_up_proj.grad.clone(), moe.gate.weight.grad.clone()


def moe_selftest(moe: MoE, dim: int, device, n_tokens: int = 4096) -> float:
    """Grouped-GEMM vs eager loop on random tokens (fwd + bwd, incl. empty experts). Returns grouped fwd time (ms)."""
    import copy, time
    m = copy.deepcopy(moe).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(1, n_tokens, dim, device=device, dtype=torch.bfloat16)
    with torch.no_grad():  # force some empty experts
        m.gate.weight[: m.E // 4].fill_(-1e4)
    ref = _grad_check(m, x, "loop")
    got = _grad_check(m, x, "grouped")
    for a, b, name in zip(got, ref, ("out", "grad_gate_up", "grad_router")):
        err = (a.float() - b.float()).abs().max().item()
        tol = 0.05 * b.float().abs().max().item() + 1e-6
        assert err <= tol, f"moe selftest {name}: max err {err:.3e} > tol {tol:.3e}"
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        m(x)
    torch.cuda.synchronize()
    return (time.time() - t) / 10 * 1e3
