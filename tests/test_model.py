import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export  # noqa: E402
from configs import parse_args  # noqa: E402
from model import Qwen3Moe, moe_selftest  # noqa: E402

MCFG, TCFG = parse_args(["--preset", "tiny"])


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return Qwen3Moe(MCFG, moe_impl="loop")


def test_forward_and_losses(net):
    x = torch.randint(0, MCFG.vocab, (2, MCFG.seq_len))
    ce, aux, z, counts = net(x, x)
    assert ce.isfinite() and 5 < ce.item() < 8  # ~ln(512)=6.2 at init
    assert counts.shape == (MCFG.n_layers, MCFG.n_experts) and counts.sum() == 2 * MCFG.seq_len * MCFG.top_k * MCFG.n_layers
    assert aux.item() >= MCFG.n_layers * 0.99  # >= 1 per layer, == 1 only at perfect balance
    assert net(x).shape == (2, MCFG.seq_len, MCFG.vocab)


def test_causal(net):
    x = torch.randint(0, MCFG.vocab, (1, MCFG.seq_len))
    y = x.clone()
    y[0, 16:] = (y[0, 16:] + 1) % MCFG.vocab
    torch.testing.assert_close(net(x)[0, :16], net(y)[0, :16], atol=1e-5, rtol=1e-4)


def test_aux_uniform_router(net):
    moe = net.model.layers[0].mlp
    with torch.no_grad():
        moe.gate.weight.zero_()
    _, aux, _, _ = moe(torch.randn(1, 64, MCFG.dim))
    assert abs(aux.item() - 1.0) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm needs CUDA")
def test_grouped_vs_loop(net):
    moe_selftest(net.model.layers[0].mlp, MCFG.dim, torch.device("cuda"), n_tokens=512)


def test_hf_export_keys(net, tmp_path):
    sd = export.hf_state_dict(net.state_dict(), MCFG)
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for l in range(MCFG.n_layers):
        p = f"model.layers.{l}."
        expected |= {p + k for k in ("input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate.weight",
                                     *(f"self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")))}
        for e in range(MCFG.n_experts):
            expected |= {f"{p}mlp.experts.{e}.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")}
    assert set(sd) == expected
    assert sd["model.layers.0.mlp.experts.1.gate_proj.weight"].shape == (MCFG.moe_dim, MCFG.dim)
    assert sd["model.layers.0.mlp.experts.1.down_proj.weight"].shape == (MCFG.dim, MCFG.moe_dim)
    torch.testing.assert_close(sd["model.layers.0.mlp.experts.1.up_proj.weight"].float(),
                               net.model.layers[0].mlp.experts.gate_up_proj[1, MCFG.moe_dim:].to(torch.bfloat16).float())
    export.write_hf(net.state_dict(), MCFG, tmp_path / "hf", "unused")
    assert (tmp_path / "hf" / "config.json").exists() and (tmp_path / "hf" / "model.safetensors.index.json").exists()


def test_train_resume(tmp_path):
    """Single-process gloo run on fake shards: train 4 steps w/ checkpoint, resume, finish at 7; export dir produced."""
    import train
    data = tmp_path / "tokens"
    data.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        rng.integers(0, MCFG.vocab, 5000, dtype=np.uint16).tofile(data / f"s{i}.bin")
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29571", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
    common = ["--preset", "tiny", "--data-dir", str(data), "--ckpt-root", str(tmp_path), "--local-dir", str(tmp_path / "local"), "--export", "0"]
    train.main(common + ["--max-steps", "4"])
    run = tmp_path / "tiny"
    assert (run / "latest.txt").exists() and json.loads((run / "summary.json").read_text())["step"] == 4
    assert (run / (run / "latest.txt").read_text() / ".metadata").exists() and (tmp_path / "local" / "tiny" / (run / "latest.txt").read_text()).exists()
    train.main(common + ["--max-steps", "7", "--export", "1"])
    events = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert any(e["event"] == "resumed" and e["step"] == 4 for e in events)
    assert json.loads((run / "summary.json").read_text())["step"] == 7
    assert (run / "export" / "parity.pt").exists() and (run / "export" / "config.json").exists()
