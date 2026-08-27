# apiary

A hive of 64 experts. From-scratch pretraining of a **6.85B-total / 1.21B-active fine-grained MoE** — the exact
Qwen3-MoE architecture (64 experts, top-8, 16 layers, d=2048, GQA 16/8 with QK-norm, 4k context, SmolLM2 49k
tokenizer) — on one Modal `B200:8` node in ~5 hours, in pure PyTorch: FSDP2 (`fully_shard`), `torch.compile`,
`torch.nn.functional.grouped_mm` experts, SDPA. No DeepSpeed, Megatron or torchtitan. ~800 lines.

| file | what |
|---|---|
| `configs.py` | model + training dataclasses, presets (`main`, `smoke`, `tiny`), shared argparse |
| `model.py` | Qwen3-MoE-compatible decoder; grouped-GEMM MoE with eager fallback; load-balance + z-loss; self-test |
| `data.py` | FineWeb-Edu parquet → uint16 shards (Modal CPU fan-out); deterministic, resumable memmap loader |
| `train.py` | torchrun entry: FSDP2 + compile, AdamW, wall-clock WSD schedule, async checkpoints, HF export |
| `export.py` | HF `Qwen3MoeForCausalLM` writer (per-expert keys: loads in transformers and vLLM) + logits-parity check |
| `modal_app.py` | Modal images/volumes/functions: `tokenize`, `smoke` (H200:2), `train` (B200:8), `evaluate` (H100) |

## Run
```bash
pip install modal && modal profile activate <profile>
modal run modal_app.py --stage tokenize --n-files 20                             # 15.3B tokens, ~12 min, ~$2
modal run modal_app.py --stage smoke --args "--preset smoke"                     # H200:2: compile, ckpt/resume, export
modal run --detach modal_app.py --stage train --args "--time-budget-min 320"     # the run; follow with `modal app logs`
modal run modal_app.py --stage evaluate --args "/ckpt/main/export"               # parity check + lm-eval
pytest tests/                                                                    # CPU tests on a tiny config
```

## Results
_(filled in after the run)_
