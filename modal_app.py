"""Modal entrypoints: tokenize (CPU fan-out), smoke (H100:2), train (B200:8), evaluate (H100).
Usage:  modal run modal_app.py --stage tokenize --n-files 20
        modal run modal_app.py --stage smoke --args "--preset smoke"      # H200:2, ~30 min
        modal run --detach modal_app.py --stage train --args "--time-budget-min 320"
        modal run modal_app.py --stage evaluate --args "/ckpt/main/export"
        modal run modal_app.py --stage publish --args "/ckpt/main/export user/model https://github.com/user/repo"
"""
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

app = modal.App("apiary")
data_vol = modal.Volume.from_name("moe-data", create_if_missing=True, version=2)
ckpt_vol = modal.Volume.from_name("moe-ckpt", create_if_missing=True, version=2)
VOLS = {"/data": data_vol, "/ckpt": ckpt_vol}
MODULES = ["configs", "data", "model", "train", "export"]

base = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("numpy", "safetensors", "tokenizers", "huggingface_hub[hf_transfer]")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
train_image = base.add_local_python_source(*MODULES)
eval_image = base.uv_pip_install("lm_eval", "transformers>=5.0", "accelerate", "matplotlib").add_local_python_source(*MODULES)
tok_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "pyarrow", "tokenizers", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "RAYON_NUM_THREADS": "16"})
    .add_local_python_source("data")
)


@app.function(image=tok_image, cpu=16, memory=32768, timeout=3600, volumes={"/data": data_vol}, max_containers=40)
def tokenize(fname: str) -> dict:
    import data
    info = data.tokenize_file(fname, "/data/tokens", "HuggingFaceTB/SmolLM2-135M")
    data_vol.commit()
    return info


def _stage_tokens(src: str, dst: str):
    """Copy shards from the volume to local disk once per container (memmap-friendly, page-cache shared by ranks)."""
    if src == dst:
        return
    Path(dst).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(src).glob("*.bin"))
    t = time.time()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda f: shutil.copyfile(f, Path(dst) / f.name), files))
    print(f"staged {len(files)} shards ({sum(f.stat().st_size for f in files) / 1e9:.1f} GB) in {time.time() - t:.0f}s", flush=True)


def _run_torchrun(argv: list[str], nproc: int) -> dict:
    """Launch train.py under torchrun; commit the volume periodically and stop torchrun once the run's time is up."""
    import configs
    t0 = time.time()
    _, tcfg = configs.parse_args(argv)
    _stage_tokens("/data/tokens", tcfg.data_dir)
    run_dir = Path(tcfg.ckpt_root) / tcfg.run
    used0 = 0.0
    if tcfg.resume == "auto" and (run_dir / "budget.json").exists():
        used0 = json.loads((run_dir / "budget.json").read_text())["used_min"]
    t_end = t0 + 60 * (tcfg.time_budget_min + 10 - used0)
    env = os.environ | dict(OMP_NUM_THREADS="4", NCCL_DEBUG="WARN", NCCL_IB_DISABLE="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
                            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TORCHINDUCTOR_CACHE_DIR="/local/inductor")
    cmd = ["torchrun", "--standalone", f"--nproc-per-node={nproc}", "-m", "train", *argv, f"--t0={t0}"]
    print("launch:", " ".join(cmd), f"| used0={used0:.1f}min ends_in={(t_end - t0) / 60:.0f}min", flush=True)
    p = subprocess.Popen(cmd, env=env | {"PYTHONPATH": "/root"}, cwd="/root", start_new_session=True)
    last_commit = time.time()
    while p.poll() is None:
        time.sleep(15)
        if time.time() > t_end:
            os.killpg(p.pid, signal.SIGKILL)
            ckpt_vol.commit()
            raise TimeoutError("torchrun exceeded the run's time limit and was killed")
        if time.time() - last_commit > 60:
            ckpt_vol.commit()
            last_commit = time.time()
    ckpt_vol.commit()
    if p.returncode != 0:
        raise RuntimeError(f"torchrun exited with {p.returncode}")
    summary = run_dir / "summary.json"
    return json.loads(summary.read_text()) if summary.exists() else {}


@app.function(image=train_image, gpu="H200:2", cpu=16, memory=131072, timeout=45 * 60, volumes=VOLS, retries=0)
def smoke(argv: list[str]) -> dict:
    return _run_torchrun(argv, nproc=2)


@app.function(image=train_image, gpu="B200:8", cpu=32, memory=262144, timeout=335 * 60, volumes=VOLS, retries=0)
def train(argv: list[str]) -> dict:
    return _run_torchrun(argv, nproc=8)


HF_SECRET = [modal.Secret.from_name("huggingface")]


@app.function(image=eval_image, gpu="H100", cpu=8, memory=65536, timeout=3600, volumes=VOLS, secrets=HF_SECRET)
def evaluate(export_dir: str, tasks: str = "hellaswag,arc_easy,arc_challenge,piqa,winogrande,wikitext,mmlu", fewshot: int = 0) -> dict:
    import export
    res = {"parity": export.parity(export_dir), "heldout": export.heldout_loss(export_dir, "/data/tokens")}
    print(res, flush=True)
    out = Path(export_dir) / "lm_eval"
    subprocess.run(["lm_eval", "--model", "hf", "--model_args", f"pretrained={export_dir},dtype=bfloat16", "--tasks", tasks,
                    "--num_fewshot", str(fewshot), "--batch_size", "auto", "--output_path", str(out)], check=True)
    for f in out.rglob("results_*.json"):
        res["lm_eval"] = {t: {k: v for k, v in r.items() if "acc" in k or "perplexity" in k} for t, r in json.loads(f.read_text())["results"].items()}
    (Path(export_dir) / "results.json").write_text(json.dumps(res, indent=1))
    ckpt_vol.commit()
    return res


@app.function(image=eval_image, cpu=8, memory=32768, timeout=3600, volumes={"/ckpt": ckpt_vol}, secrets=HF_SECRET)
def publish(export_dir: str, repo_id: str, code_url: str) -> str:
    import export
    return export.publish(export_dir, repo_id, code_url)


@app.local_entrypoint()
def main(stage: str = "train", args: str = "", n_files: int = 20):
    argv = shlex.split(args)
    if stage == "tokenize":
        import data
        files = data.list_parquet_files(n_files)
        print(f"tokenizing {len(files)} files: {files[0]} .. {files[-1]}")
        infos = [i for i in tokenize.map(files, return_exceptions=True)]
        ok = [i for i in infos if isinstance(i, dict)]
        print(f"done: {len(ok)}/{len(files)} files, {sum(i['n_tokens'] for i in ok) / 1e9:.2f}B tokens")
        for i in infos:
            if not isinstance(i, dict):
                print("FAILED:", i)
    elif stage == "smoke":
        print(json.dumps(smoke.remote(argv), indent=1))
    elif stage == "train":
        print(json.dumps(train.remote(argv), indent=1))
    elif stage == "evaluate":
        print(json.dumps(evaluate.remote(*argv), indent=1))
    elif stage == "publish":
        print(publish.remote(*argv))
    else:
        raise SystemExit(f"unknown stage {stage}")
