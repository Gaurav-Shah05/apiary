"""Write the trained model as an HF `Qwen3MoeForCausalLM` directory (per-expert keys: loadable by transformers and vLLM),
and check logits parity against transformers."""
import json
import shutil
from pathlib import Path

import torch

from configs import ModelCfg

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def hf_config(c: ModelCfg) -> dict:
    return dict(
        architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe", hidden_size=c.dim, num_hidden_layers=c.n_layers,
        num_attention_heads=c.n_heads, num_key_value_heads=c.n_kv_heads, head_dim=c.head_dim, num_experts=c.n_experts,
        num_experts_per_tok=c.top_k, moe_intermediate_size=c.moe_dim, intermediate_size=c.moe_dim, norm_topk_prob=True,
        decoder_sparse_step=1, mlp_only_layers=[], hidden_act="silu", vocab_size=c.vocab, max_position_embeddings=c.seq_len,
        rope_theta=c.rope_theta, rope_parameters={"rope_type": "default", "rope_theta": c.rope_theta}, rms_norm_eps=c.norm_eps,
        attention_bias=False, attention_dropout=0.0, tie_word_embeddings=False, router_aux_loss_coef=c.aux_coef,
        output_router_logits=False, use_sliding_window=False, initializer_range=c.init_std, use_cache=True,
        bos_token_id=0, eos_token_id=0, dtype="bfloat16", torch_dtype="bfloat16",
    )


def hf_state_dict(sd: dict, c: ModelCfg) -> dict:
    """Our (fused-expert) state dict -> HF per-expert keys, bf16."""
    out = {}
    for k, v in sd.items():
        k = k.replace("._orig_mod", "")
        if k.endswith("experts.gate_up_proj"):
            for e in range(c.n_experts):
                out[f"{k[:-len('gate_up_proj')]}{e}.gate_proj.weight"] = v[e, : c.moe_dim]
                out[f"{k[:-len('gate_up_proj')]}{e}.up_proj.weight"] = v[e, c.moe_dim:]
        elif k.endswith("experts.down_proj"):
            for e in range(c.n_experts):
                out[f"{k[:-len('down_proj')]}{e}.down_proj.weight"] = v[e]
        else:
            out[k] = v
    return {k: v.detach().to(torch.bfloat16).contiguous() for k, v in out.items()}


def write_hf(sd: dict, c: ModelCfg, out_dir: Path, tokenizer_id: str, shard_bytes: int = 4 * 2**30):
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = hf_state_dict(sd, c)
    shards, cur, size = [], {}, 0
    for k, v in tensors.items():
        if cur and size + v.nbytes > shard_bytes:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = v
        size += v.nbytes
    shards.append(cur)
    names = [f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(shards)) as ex:
        list(ex.map(lambda p: save_file(p[0], out_dir / p[1], metadata={"format": "pt"}), zip(shards, names)))
    weight_map = {k: n for shard, n in zip(shards, names) for k in shard}
    total = sum(v.nbytes for v in tensors.values())
    (out_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    (out_dir / "config.json").write_text(json.dumps(hf_config(c), indent=1))
    try:
        from huggingface_hub import hf_hub_download
        for f in TOKENIZER_FILES:
            shutil.copyfile(hf_hub_download(tokenizer_id, f), out_dir / f)
    except Exception as e:  # offline (tests): the model is still loadable without a tokenizer
        print(f"tokenizer files not copied: {e}")
    print(f"wrote HF checkpoint: {len(tensors)} tensors, {total / 2**30:.1f} GB, {len(shards)} shards -> {out_dir}", flush=True)


@torch.no_grad()
def parity(export_dir: str) -> dict:
    """Compare transformers' logits on the parity batch saved at export time with the trainer's own logits."""
    from transformers import AutoModelForCausalLM
    p = torch.load(Path(export_dir) / "parity.pt")
    model = AutoModelForCausalLM.from_pretrained(export_dir, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu")
    logits = model(p["tokens"].to(model.device)).logits[0].float().cpu()
    ref = p["logits"]
    diff = (logits - ref).abs()
    res = dict(max_abs=diff.max().item(), mean_abs=diff.mean().item(), argmax_agree=(logits.argmax(-1) == ref.argmax(-1)).float().mean().item())
    assert res["argmax_agree"] > 0.97 and res["mean_abs"] < 0.1, f"parity failed: {res}"
    return res
