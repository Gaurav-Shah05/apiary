"""FineWeb-Edu -> uint16 token shards (run on Modal CPU workers) and the memmap training loader."""
import json
import os
from pathlib import Path

import numpy as np

REPO = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample/100BT"
EOS_TOKEN = "<|endoftext|>"


def list_parquet_files(n_files: int) -> list[str]:
    from huggingface_hub import HfApi
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith(SUBSET + "/") and f.endswith(".parquet"))
    return files[:n_files]


def tokenize_file(fname: str, out_dir: str, tokenizer_id: str) -> dict:
    """Tokenize one parquet file into <out_dir>/<name>.bin (uint16, EOS after each doc). Idempotent."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    out = Path(out_dir) / (Path(fname).stem + ".bin")
    meta = out.with_suffix(".json")
    if meta.exists():
        return json.loads(meta.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_pretrained(tokenizer_id)
    eos = tok.token_to_id(EOS_TOKEN)
    assert eos is not None and tok.get_vocab_size() < 65536
    path = hf_hub_download(REPO, fname, repo_type="dataset")
    n = 0
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
            encs = tok.encode_batch(batch.column("text").to_pylist(), add_special_tokens=False)
            arr = np.concatenate([np.array(e.ids + [eos], dtype=np.uint16) for e in encs])
            f.write(arr.tobytes())
            n += len(arr)
    os.replace(tmp, out)
    info = dict(file=fname, n_tokens=n, eos=eos, tokenizer=tokenizer_id)
    meta.write_text(json.dumps(info))
    return info


class TokenLoader:
    """Deterministic, resumable batches. Window w = tokens[w*L : w*L+L+1] over the concatenated shards;
    optimizer step s, micro-step m, rank r reads windows s*G + r*(G/W) + m*B + [0, B)."""

    def __init__(self, data_dir: str, seq_len: int, micro_batch: int, global_batch: int, rank: int, world: int):
        files = sorted(Path(data_dir).glob("*.bin"))
        assert files, f"no .bin shards in {data_dir}"
        self.mm = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.L, self.B, self.G, self.rank, self.world = seq_len, micro_batch, global_batch, rank, world
        assert global_batch % (micro_batch * world) == 0
        self.per_rank = global_batch // world
        counts = np.array([(len(m) - 1) // seq_len for m in self.mm])
        self.cum = np.concatenate([[0], np.cumsum(counts)])
        self.n_windows = int(self.cum[-1])
        self.n_tokens = sum(len(m) for m in self.mm)

    def batch(self, step: int, micro: int):
        import torch
        start = step * self.G + self.rank * self.per_rank + micro * self.B
        rows = []
        for w in range(start, start + self.B):
            w %= self.n_windows
            shard = int(np.searchsorted(self.cum, w, side="right") - 1)
            off = (w - self.cum[shard]) * self.L
            rows.append(self.mm[shard][off: off + self.L + 1].astype(np.int64))
        buf = torch.from_numpy(np.stack(rows))
        return buf[:, :-1], buf[:, 1:]
