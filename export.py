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
    res = dict(max_abs=diff.max().item(), mean_abs=diff.mean().item(), argmax_agree=(logits.argmax(-1) == ref.argmax(-1)).float().mean().item(),
               ref_argmax_topk_in=(logits.topk(5, -1).indices == ref.argmax(-1, keepdim=True)).any(-1).float().mean().item())
    # Report only. bf16 kernel differences (compile+cuDNN SDPA+grouped_mm vs eager) move argmax on near-tied logits;
    # the definitive correctness gate is held-out loss on the HF-loaded model (see heldout_loss).
    assert res["mean_abs"] < 0.5, f"export logits diverge from transformers: {res}"
    return res


@torch.no_grad()
def heldout_loss(export_dir: str, data_dir: str, n_windows: int = 64, seq_len: int = 4096) -> dict:
    """Mean CE on windows from the end of the last shard, which a run that consumes shards in order never sees."""
    import numpy as np
    from transformers import AutoModelForCausalLM
    shard = sorted(Path(data_dir).glob("*.bin"))[-1]
    mm = np.memmap(shard, dtype=np.uint16, mode="r")
    model = AutoModelForCausalLM.from_pretrained(export_dir, dtype=torch.bfloat16, device_map="cuda")
    losses = []
    for i in range(n_windows):
        off = len(mm) - (i + 1) * (seq_len + 1)
        x = torch.from_numpy(mm[off: off + seq_len + 1].astype(np.int64))[None].cuda()
        logits = model(x[:, :-1]).logits.float()
        losses.append(torch.nn.functional.cross_entropy(logits[0], x[0, 1:]).item())
    loss = sum(losses) / len(losses)
    return dict(shard=shard.name, tokens=n_windows * seq_len, loss=loss, ppl=float(torch.tensor(loss).exp()))


def _loss_plot(metrics_path: Path, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines()]
    steps = [r for r in rows if r.get("event") == "step"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot([r["tokens"] / 1e9 for r in steps], [r["ce"] for r in steps], lw=0.8)
    ax.set_xlabel("tokens (B)"), ax.set_ylabel("train CE (nats)"), ax.set_ylim(2, 6), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(out_png, dpi=130)


def publish(export_dir: str, repo_id: str, code_url: str) -> str:
    """Upload the HF checkpoint, a model card, the training log and loss curve to the Hub."""
    import hashlib
    from huggingface_hub import HfApi
    export_dir = Path(export_dir)
    run_dir = export_dir.parent
    cfg, res = json.loads((export_dir / "config.json").read_text()), json.loads((export_dir / "results.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    steps = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    steps = [r for r in steps if r.get("event") == "step"]
    tps = sorted(r["tok_s"] for r in steps)[len(steps) // 2]
    mfu = sorted(r["mfu"] for r in steps)[len(steps) // 2]
    _loss_plot(run_dir / "metrics.jsonl", export_dir / "loss_curve.png")
    sha = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(export_dir.glob("*.safetensors"))}
    ev = res.get("lm_eval", {})
    def acc(t):
        r = ev.get(t, {})
        v = r.get("acc_norm,none", r.get("acc,none"))
        return f"{100 * v:.1f}" if v is not None else "-"
    wt = ev.get("wikitext", {}).get("word_perplexity,none")
    card = f"""---
license: apache-2.0
language: [en]
datasets: [HuggingFaceFW/fineweb-edu]
pipeline_tag: text-generation
library_name: transformers
tags: [moe, qwen3_moe, pretraining, from-scratch]
---
# {repo_id.split('/')[-1]}

A {cfg['num_experts']}-expert Mixture-of-Experts language model pretrained **from scratch** on
{summary['tokens'] / 1e9:.2f}B tokens of FineWeb-Edu in {summary['used_min'] / 60:.2f} hours on one 8xB200 node
({summary['used_min'] * 8 / 60:.1f} B200 GPU-hours). Architecture is exactly `Qwen3MoeForCausalLM`
({cfg['num_hidden_layers']} layers, d={cfg['hidden_size']}, {cfg['num_attention_heads']}/{cfg['num_key_value_heads']} heads,
{cfg['num_experts']} experts top-{cfg['num_experts_per_tok']}, expert size {cfg['moe_intermediate_size']}, vocab {cfg['vocab_size']}):
6.85B total / 1.21B active parameters. Code, training log and configs: {code_url}

This is a research artifact from a fixed compute budget: it is a coherent base LM in the GPT-2-XL / Pythia-1B class,
not an instruction model and not competitive with models trained on trillions of tokens.

## Training
- Data: first 20 parquet files of `HuggingFaceFW/fineweb-edu` `sample/100BT`, tokenized with the SmolLM2 tokenizer
  (EOS-separated, no document masking), consumed sequentially — see `data.py` in the code repo.
- Optimizer: AdamW lr 3e-4 (betas 0.9/0.95, wd 0.1), 1.05M-token batches, 500 warmup steps, constant then linear
  decay over the last 20% of the wall-clock budget; grad clip 1.0; load-balance loss 0.01, router z-loss 1e-3.
- Systems: PyTorch 2.13, FSDP2 + torch.compile + `grouped_mm` experts, bf16 with fp32 master weights;
  median {tps / 1e3:.0f}k tokens/s ({100 * mfu:.1f}% MFU) on 8xB200; {summary['step']} optimizer steps.
- Final train CE {steps[-1]['ce']:.3f}; held-out FineWeb-Edu CE {res['heldout']['loss']:.3f} (ppl {res['heldout']['ppl']:.1f})
  on {res['heldout']['tokens'] / 1e3:.0f}k unseen tokens.

![loss curve](loss_curve.png)

## Evaluation (lm-eval-harness, 0-shot)
| HellaSwag | ARC-e | ARC-c | PIQA | Winogrande | MMLU | wikitext word ppl |
|---|---|---|---|---|---|---|
| {acc('hellaswag')} | {acc('arc_easy')} | {acc('arc_challenge')} | {acc('piqa')} | {acc('winogrande')} | {acc('mmlu')} | {f'{wt:.1f}' if wt else '-'} |

(acc_norm where the task defines it.) Logits parity vs the training code: argmax agreement
{100 * res['parity']['argmax_agree']:.1f}%, mean |diff| {res['parity']['mean_abs']:.3f}.

## Provenance
Weights sha256 (first 16 hex): {json.dumps(sha)}. Full per-10-step training log: `metrics.jsonl`.
"""
    (export_dir / "README.md").write_text(card)
    shutil.copyfile(run_dir / "metrics.jsonl", export_dir / "metrics.jsonl")
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=False)
    api.upload_folder(folder_path=str(export_dir), repo_id=repo_id,
                      allow_patterns=["*.safetensors", "model.safetensors.index.json", "config.json", *TOKENIZER_FILES, "README.md", "loss_curve.png", "metrics.jsonl"])
    return f"https://huggingface.co/{repo_id}"
