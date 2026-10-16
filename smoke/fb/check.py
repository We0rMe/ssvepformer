"""FB 单线程轻量检查。合成训练产生的文件均在 smoke/fb/results，绝不启动正式实验。"""

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy import signal

from config import ROOT
from data import complex_spectrum, read_subject, split_subjects
from evaluate import evaluate, write_json
from fb.config import CONFIG
from fb.data import filter_bank, prepare_data, preprocess
from fb.evaluate import evaluate_checkpoint, save_results, summarize
from fb.model import FBSSVEPformer
from fb.train import check_manifest, run, train_stages, training_plan
from train import seed_everything, train_epoch


def protected_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("config.py", "data.py", "model.py", "train.py", "evaluate.py")}


def check_filters(cfg):
    t = np.arange(1500) / cfg.sample_rate
    wave = np.cos(2 * np.pi * 10 * t) + np.cos(2 * np.pi * 30 * t)
    trials = np.broadcast_to(wave, (2, len(cfg.channels), len(t))).copy()
    designs = filter_bank(cfg)
    actual = preprocess(trials, cfg, designs)
    assert actual.shape == (2, 3, 9, 562) and actual.dtype == np.float32
    for band, design in enumerate(designs):
        sos = np.array(design["sos"])
        assert design["passband_hz"] == [(band + 1) * 9, 80]
        assert design["stopband_hz"] == [(band + 1) * 9 - 2, 90]
        pass_freqs = np.linspace(*design["passband_hz"], 1000)
        _, response = signal.sosfreqz(sos, worN=pass_freqs, fs=cfg.sample_rate)
        assert np.min(20 * np.log10(np.abs(response))) >= -cfg.passband_ripple - 1e-7
        _, response = signal.sosfreqz(sos, worN=np.array(design["stopband_hz"]), fs=cfg.sample_rate)
        assert np.max(20 * np.log10(np.abs(response))) <= -cfg.stopband_attenuation
        filtered = signal.sosfiltfilt(sos, trials, axis=-1)
        expected = complex_spectrum(filtered[..., cfg.start:cfg.start + cfg.samples], cfg)
        np.testing.assert_array_equal(actual[:, band], expected)
    # 后两个子带应抑制 10 Hz，而保留 30 Hz；不把子带当成互不重叠的频谱切片。
    ten, thirty = round((10 - 8) / .2), round((30 - 8) / .2)
    magnitude = np.hypot(actual[..., :cfg.features // 2], actual[..., cfg.features // 2:])
    assert np.max(magnitude[0, 1:, 0, ten]) < magnitude[0, 0, 0, ten] * .1
    assert np.min(magnitude[0, :, 0, thirty]) > .3
    raw = read_subject(2, cfg)[:2]  # 仅取训练侧两个试次；不生成全量缓存、不使用测试指标。
    real = preprocess(raw, cfg)
    np.testing.assert_array_equal(real, np.concatenate([preprocess(raw[i:i + 1], cfg) for i in range(2)]))
    assert np.isfinite(real).all()
    return designs, real


def check_isolation_and_fusion(cfg, real):
    seed_everything(cfg)
    model = FBSSVEPformer(cfg)
    parameter_sets = [set(id(p) for p in net.parameters()) for net in model.subnetworks]
    assert all(not parameter_sets[i] & parameter_sets[j] for i in range(3) for j in range(i))
    x = torch.from_numpy(real)
    y = torch.tensor([0, 1])
    model.eval()
    with torch.no_grad():
        scores = torch.stack([net(x[:, i]) for i, net in enumerate(model.subnetworks)], dim=1)
        expected = (scores * model.fusion.weight.view(1, 3, 1)).sum(1)
        torch.testing.assert_close(model(x), expected)
    untouched = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith("subnetworks.0.")}
    optimizer = torch.optim.SGD(model.subnetworks[0].parameters(), lr=cfg.learning_rate)
    train_epoch(model.subnetworks[0], optimizer, x[:, 0], y, cfg, 1)
    for key, value in untouched.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    model.train()
    torch.nn.functional.cross_entropy(model(x), y).backward()
    assert torch.isfinite(model.fusion.weight.grad).all() and model.fusion.weight.grad.abs().sum() > 0
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()) for net in model.subnetworks)


def check_cache_and_splits(cfg, output):
    subject_data = np.broadcast_to(np.arange(1, 36)[:, None, None, None, None], (35, 240, 3, 1, 1)).copy()
    for single in replace(cfg, N=35).runs():
        x, y, test_x, test_y = split_subjects(subject_data, single)
        assert len(y) == 8160 and len(test_y) == 240
        assert single.test_subject not in torch.unique(x).tolist()
        assert torch.unique(test_x).tolist() == [single.test_subject]
    # 小型假数据覆盖缓存写入/读取路径；文件位置也在 smoke，不触碰正在使用的缓存。
    tiny = replace(cfg, channels=("Oz",), frequency_range=(8., 8.4))
    with tempfile.TemporaryDirectory(dir=output) as temp, patch("fb.data.ROOT", ROOT):
        # ROOT 同时用于代码摘要；只改缓存路径需提供相同的源文件快照。
        sandbox = Path(temp)
        (sandbox / "fb").mkdir()
        for name in ("data.py", "fb/data.py"):
            (sandbox / name).write_bytes((ROOT / name).read_bytes())
        with patch("fb.data.ROOT", sandbox), patch("fb.data.read_subject", return_value=np.zeros((240, 1, 1500))) as read, \
                patch("fb.data.preprocess", return_value=np.zeros((240, 3, 1, 6), dtype=np.float32)):
            features, audit = prepare_data(tiny)
            assert read.call_count == 35
            read.reset_mock()
            features_again, audit_again = prepare_data(replace(tiny, N=35, M=2, seed=2027))
            read.assert_not_called()
            assert audit["cache_id"] == audit_again["cache_id"]
            np.testing.assert_array_equal(features, features_again)
            del features, features_again


def check_stages(cfg, output):
    # 缩小特征/通道以覆盖完整控制流程；完整 3×9×562 架构已在前一项检查反向传播。
    tiny = replace(cfg, channels=("Oz",), frequency_range=(8., 8.4), encoder_blocks=1,
                   kernel_size=3, head_multiplier=1, spatial_multiplier=1,
                   epochs=2, joint_epochs=2, batch_size=2, checkpoint_every=1, monitor_every=1)
    tiny.validate()
    x = torch.tensor(np.random.default_rng(7).normal(size=(4, 3, 1, 6)), dtype=torch.float32)
    y, monitor = torch.tensor([0, 1, 2, 3]), torch.arange(2)
    assert [s[0] for s in training_plan(tiny)] == ["subband_1", "subband_2", "subband_3", "joint"]
    with tempfile.TemporaryDirectory(dir=output) as temp:
        folder = Path(temp)
        baseline_dir = folder / "baseline" / "reproducibility"
        baseline_dir.mkdir(parents=True)
        seed_everything(tiny)
        expected_model = FBSSVEPformer(tiny)
        with patch("fb.train.plot_training"):
            expected_history = train_stages(expected_model, x, y, monitor, tiny, baseline_dir,
                                             "test", "synthetic", lambda: False, lambda _: None)
        expected = torch.load(baseline_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert expected["training_complete"] and len(expected_history) == 8

        # 在预训练中途、阶段边界、联合训练中途分别中断，恢复后必须逐参数完全一致。
        for calls_before_stop in (2, 4, 14):
            details = folder / f"resume_{calls_before_stop}" / "reproducibility"
            details.mkdir(parents=True)
            seed_everything(tiny)
            interrupted = FBSSVEPformer(tiny)
            calls = 0

            def should_stop():
                nonlocal calls
                calls += 1
                return calls >= calls_before_stop

            with patch("fb.train.plot_training"):
                try:
                    train_stages(interrupted, x, y, monitor, tiny, details, "test", "synthetic", should_stop, lambda _: None)
                except InterruptedError:
                    pass
                else:
                    raise AssertionError("STOP 没有生效")
                seed_everything(tiny)
                resumed = FBSSVEPformer(tiny)
                actual_history = train_stages(resumed, x, y, monitor, tiny, details,
                                               "test", "synthetic", lambda: False, lambda _: None)
            for key, value in expected_model.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            assert [{k: v for k, v in r.items() if k != "seconds"} for r in actual_history] == [
                {k: v for k, v in r.items() if k != "seconds"} for r in expected_history]
        metrics, predictions = evaluate(expected_model, x, y, tiny)
        save_results(tiny, expected_model, expected_history, metrics, metrics, predictions,
                     y.numpy(), list(range(40)), output=folder / "baseline")
        for name in ("report.md", "metrics.csv", "training.png", "confusion.png"):
            assert (folder / "baseline" / name).is_file()
        # 保存一张合成阶段曲线供人工查看，明确不会作为正式指标。
        (output / "synthetic_training.png").write_bytes((folder / "baseline" / "training.png").read_bytes())

        incomplete = expected.copy()
        incomplete["training_complete"] = False
        torch.save(incomplete, folder / "incomplete.pt")
        with patch("fb.evaluate.prepare_data") as prepare:
            assert not evaluate_checkpoint(folder / "incomplete.pt")
            prepare.assert_not_called()
        with patch("fb.evaluate.prepare_data", return_value=(None, {"cache_id": "synthetic", "target_frequencies": list(range(40))})), \
                patch("fb.evaluate.split_subjects", return_value=(x, y, x, y)):
            assert evaluate_checkpoint(baseline_dir / "checkpoint.pt")


def check_scheduler(cfg, output):
    with tempfile.TemporaryDirectory(dir=output) as temp, patch("config.ROOT", Path(temp)):
        plan = replace(cfg, N=2, M=2)
        with patch("fb.train.prepare_data", return_value=(None, {})), \
                patch("fb.train.run_single", side_effect=[True, False]) as execute, patch("fb.train.summarize"):
            run(plan)
            assert [(c.args[0].test_subject, c.args[0].seed) for c in execute.call_args_list] == [(1, 2026), (1, 2027)]
        single = next(plan.runs())
        details = single.output_dir / "reproducibility"
        details.mkdir(parents=True)
        signature = check_manifest(single, {"cache_id": "test"}, details)
        assert signature == check_manifest(replace(single, N=35, M=3), {"cache_id": "test"}, details)
        try:
            check_manifest(replace(single, subbands=2), {"cache_id": "test"}, details)
        except ValueError:
            pass
        else:
            raise AssertionError("更换子带数应拒绝复用原实验")
        summarize(plan)
        assert "# FB-SSVEPformer" in (plan.output_dir / "report.md").read_text(encoding="utf-8")


def main():
    output = ROOT / "smoke" / "fb" / "results"
    output.mkdir(parents=True, exist_ok=True)
    cfg = replace(CONFIG, cpu_threads=1, device="cpu")
    seed_everything(cfg)
    before = protected_hashes()
    designs, real = check_filters(cfg)
    check_isolation_and_fusion(cfg, real)
    check_cache_and_splits(cfg, output)
    check_stages(cfg, output)
    check_scheduler(cfg, output)
    assert before == protected_hashes()
    passed = ["Subband frequency responses and explicit filter/crop/FFT ordering",
              "Two real training-side trials: finite features and independent per-trial processing",
              "Full-size FB: independent subnetworks, exact logits fusion, isolated pretraining, joint gradients",
              "All 35 LOSO splits exclude test subject; cache is shared across folds and seeds",
              "Resume within pretraining, at stage boundary, and within joint training is bit-exact",
              "Final FB checkpoint re-evaluation and reports work; incomplete checkpoint never tests",
              "STOP stops the whole plan; N/M expansion preserves signature; incompatible subbands rejected",
              "FB checks do not modify SSVEPformer core scripts"]
    write_json(output / "checks.json", {"passed": passed, "filter_orders": [d["prototype_order"] for d in designs],
                                        "protected_hashes": before, "formal_training_started": False})
    for item in passed:
        print("PASS:", item)


if __name__ == "__main__":
    main()
