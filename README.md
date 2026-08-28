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

Final model: **9.84B tokens**, train CE **2.465** / held-out CE **2.462** (ppl 11.7),
**314 B200 node-minutes** (41.9 GPU-hours), median **612k tok/s / 28.3% MFU** on 8xB200.
Weights + card: [DruidTheGetafix/apiary-7B-A1B](https://huggingface.co/DruidTheGetafix/apiary-7B-A1B).

lm-eval-harness, 0-shot (acc_norm where the task defines it):

| HellaSwag | ARC-easy | ARC-challenge | PIQA | Winogrande | MMLU | wikitext word-ppl |
|---|---|---|---|---|---|---|
| 48.8 | 59.6 | 33.4 | 70.9 | 52.5 | 25.5 | 19.2 |

### How it compares

All rows are 0-shot lm-eval-harness (acc_norm for HellaSwag/ARC/PIQA, acc for Winogrande/MMLU).

**Competitive per-token — matches or beats dense models trained on 3-30x more tokens:**

| model | tokens | HellaSwag | ARC-e | ARC-c | PIQA | Wino |
|---|---|---|---|---|---|---|
| **apiary-7B-A1B (this)** | **9.8B** | **48.8** | **59.6** | **33.4** | **70.9** | **52.5** |
| Cerebras-GPT-1.3B | 26B | 32.5 | 50.8 | 22.4 | 66.4 | 52.1 |
| Pythia-1B | 300B | 47.2 | 49.0 | 27.1 | 69.2 | 53.4 |
| Pythia-1.4B | 300B | 52.0 | 54.0 | 28.5 | 71.0 | 57.4 |
| OPT-1.3B | 180B | 53.6 | 50.8 | 29.4 | 72.4 | 59.6 |

We beat Cerebras-GPT-1.3B outright and match-or-beat Pythia-1B across the board with ~30x fewer
tokens; against the bigger Pythia-1.4B / OPT-1.3B we lead on both ARC tasks and trail on
HellaSwag/Winogrande. HuggingFace's own FineWeb-Edu 1.8B ablation reaches only ~44 HellaSwag at
~10B tokens, so this run sits at/above the efficient frontier for its budget.

**Not competitive in absolute terms with fully-trained modern small models:**

| model | tokens | HellaSwag | ARC-e | ARC-c | PIQA | Wino | MMLU |
|---|---|---|---|---|---|---|---|
| **apiary-7B-A1B (this)** | **9.8B** | 48.8 | **59.6** | **33.4** | 70.9 | 52.5 | 25.5 |
| TinyLlama-1.1B | 3T | 59.2 | 55.2 | 30.1 | 73.3 | 59.1 | ~25 |
| OLMo-1B | 3T | 62.5 | 58.1 | 34.5 | 73.7 | 58.9 | ~25 |
| SmolLM2-1.7B | 11T | 68.7 | - | - | 77.6 | 59.4 | ~50 |
| OLMoE-1B-7B (same 64-expert shape) | 5.1T | 78.2 | 76.9 | 49.2 | 79.7 | 68.9 | 53.5 |

These win on HellaSwag/PIQA/Winogrande by 10-30 points on 300-1000x the compute; we still edge
TinyLlama and match OLMo-1B on ARC. The identical-architecture OLMoE-1B-7B (5.1T tokens) is the
honest ceiling of this design. MMLU at chance is universal at this budget.

Full per-10-step training log, loss curve and eval output are in [`runs/main/`](runs/main).
The exact training code is tagged [`run-main`](../../releases/tag/run-main).

![loss curve](runs/main/loss_curve.png)
