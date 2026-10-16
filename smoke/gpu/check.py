"""CPU 必测，CUDA 有卡才实测；检查两模型的设备迁移和确定性断点。无正式训练。"""

import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from config import CONFIG, ROOT
from evaluate import evaluate, write_json
from fb.config import CONFIG as FB_CONFIG
from fb.model import FBSSVEPformer
from fb.train import save_checkpoint as save_fb_checkpoint
from model import SSVEPformer
from runtime import configure_device, optimizer_to, restore_rng, timing_message
from train import save_checkpoint, seed_everything, train_epoch


def check_model(cfg, factory, fb, directory):
    device = seed_everything(cfg)
    model = factory(cfg).to(device)
    shape = (4, cfg.subbands, len(cfg.channels), cfg.features) if fb else (4, len(cfg.channels), cfg.features)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(37))
    y = torch.tensor([0, 1, 2, 3])
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    train_epoch(model, optimizer, x, y, cfg, 1)
    path = directory / "checkpoint.pt"
    if fb:
        save_fb_checkpoint(path, model, optimizer, cfg, "test", "test", cfg.subbands, 1, [])
    else:
        save_checkpoint(path, model, optimizer, cfg, "test", 1, [])
    expected_loss, _ = train_epoch(model, optimizer, x, y, cfg, 2)
    expected = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    expected_metrics, expected_predictions = evaluate(model, x, y, cfg)
    assert len(expected_predictions) == len(y)
    del model, optimizer

    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["execution_device"] == device.type
    assert (saved["cuda_rng"] is not None) == (device.type == "cuda")
    resumed = factory(cfg).to(device)
    resumed.load_state_dict(saved["model"])
    optimizer = torch.optim.SGD(resumed.parameters(), lr=cfg.learning_rate,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(saved["optimizer"])
    optimizer_to(optimizer, device)
    restore_rng(saved, resumed)
    actual_loss, _ = train_epoch(resumed, optimizer, x, y, cfg, 2)
    assert actual_loss == expected_loss
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
    actual_metrics, actual_predictions = evaluate(resumed, x, y, cfg)
    assert actual_metrics == expected_metrics
    assert (actual_predictions == expected_predictions).all()
    assert all(value.device == device for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor))
    if device.type == "cuda":
        # GPU 保存的模型可单独在 CPU 推理；不将其冒充 CPU 上连续训练的断点。
        cpu_model = factory(cfg)
        cpu_model.load_state_dict(saved["model"])
        metrics, _ = evaluate(cpu_model, x, y, replace(cfg, device="cpu"))
        assert metrics["total"] == 4
        try:
            restore_rng(saved, cpu_model)
        except ValueError:
            pass
        else:
            raise AssertionError("不应无提示地跨 CPU/CUDA 恢复训练随机状态")


def main():
    output = ROOT / "smoke" / "gpu" / "results"
    output.mkdir(parents=True, exist_ok=True)
    cpu_cfg = replace(CONFIG, device="cpu", cpu_threads=1)
    with patch("torch.cuda.is_available", return_value=False):
        try:
            configure_device(replace(cpu_cfg, device="cuda:0"))
        except RuntimeError:
            pass
        else:
            raise AssertionError("CUDA 不可用时必须报错，不得静默退回 CPU")
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
        try:
            configure_device(replace(cpu_cfg, device="cuda:2"))
        except ValueError:
            pass
        else:
            raise AssertionError("应拒绝不存在的 GPU 编号")
    message = timing_message([{"epoch": i, "seconds": 2.0} for i in range(1, 6)], 100, 35)
    assert "1.94" in message
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    checks = []
    for device in devices:
        for name, template, factory, fb in (("SSVEPformer", CONFIG, SSVEPformer, False),
                                            ("FB-SSVEPformer", FB_CONFIG, FBSSVEPformer, True)):
            cfg = replace(template, device=device, cpu_threads=1)
            with tempfile.TemporaryDirectory(dir=output) as temporary:
                check_model(cfg, factory, fb, Path(temporary))
            checks.append(f"{name} {device}: forward/backward, evaluation, optimizer/RNG resume bit-exact")
            print("PASS:", checks[-1], flush=True)
    result = {"passed": checks, "guards": "Unavailable CUDA and invalid GPU index fail explicitly",
              "cuda_tested": len(devices) > 1,
              "cuda_status": "tested on available GPUs" if len(devices) > 1 else "SKIPPED: this machine has no CUDA GPU",
              "timing": "2 seconds/epoch × 100 epochs × 35 folds = 1.94 training hours (conditional arithmetic only)"}
    write_json(output / "checks.json", result)
    print(result["cuda_status"], flush=True)


if __name__ == "__main__":
    main()
