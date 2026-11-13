"""实验三四独立检查；合成数据，不读取 Benchmark，不启动正式训练。"""
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from course import runner as exp
from config import CONFIG, ROOT
from data import preprocess
from model import SSVEPformer
from train import seed_everything


def main():
    cfg = replace(CONFIG, device="cpu", cpu_threads=1, N=1, M=1, channels=("Oz",),
                  duration=.024, frequency_range=(8., 8.4), encoder_blocks=1, kernel_size=3,
                  spatial_multiplier=1, head_multiplier=1, epochs=2, batch_size=512)
    seed_everything(cfg)
    raw = np.random.default_rng(42).normal(size=(240, 1, 1500))
    for snr in (20., 10., 0.):
        a = exp.add_noise(raw, cfg, snr, 1, 0, 711)
        b = exp.add_noise(raw, cfg, snr, 1, 0, 711)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, exp.add_noise(raw, cfg, snr, 1, 1, 711))
        w = slice(cfg.start, cfg.start + cfg.samples)
        actual = 10 * np.log10(np.mean(raw[..., w] ** 2, axis=-1) / np.mean((a - raw)[..., w] ** 2, axis=-1))
        np.testing.assert_allclose(actual, snr, atol=1e-10)
    np.testing.assert_array_equal(exp.add_noise(raw, cfg, None, 1, 0, 711), raw)
    first = SSVEPformer(replace(cfg, input_type="time")).eval()
    second = SSVEPformer(replace(cfg, input_type="complex")).eval()
    model = exp.ScoreFusion(first, second, cfg.samples).eval()
    x = torch.randn(4, 1, 12)
    with torch.no_grad():
        expected = .5 * first(x[..., :6]).softmax(-1) + .5 * second(x[..., 6:]).softmax(-1)
        torch.testing.assert_close(model(x).exp(), expected)
    for kinds in (("time", "complex"), ("time", "time"), ("complex", "complex")):
        net = exp.FeatureFusion(cfg, kinds, 4)
        values = torch.randn(4, 1, sum(net.lengths))
        loss = torch.nn.functional.cross_entropy(net(values), torch.arange(4))
        loss.backward()
        assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters()) for e in net.encoders)
        groups = [set(id(p) for p in e.parameters()) for e in net.encoders]
        assert groups[0].isdisjoint(groups[1])
    output = ROOT / "smoke/experiments/course_results"
    output.mkdir(parents=True, exist_ok=True)
    # 固定合成样本的演示图保留在 smoke 中，不能用作真实实验图。
    exp.plot_signal(raw, exp.add_noise(raw, cfg, 0., 1, 0, 711), cfg, output, 1, 0.)
    with tempfile.TemporaryDirectory(dir=output) as temp:
        folder = Path(temp)
        def fake_prepare(c):
            values = preprocess(raw, c)
            return np.broadcast_to(values, (35, *values.shape)).copy(), {"cache_id": "synthetic"}
        options = dict(name="synthetic_course_check", duration=.024,
                       fusion=replace(exp.FUSION, projection_dim=4),
                       noise=replace(exp.NOISE, snr_db=(10., 0.), repeats=2))
        sources = {"complex_no_bandpass": None}
        for kind in ("time", "complex"):
            c = replace(cfg, input_type=kind)
            checkpoint = folder / f"{kind}_S01_seed2026.pt"
            torch.save({"config": c.to_dict(), "epoch": c.epochs, "model": SSVEPformer(c).state_dict()}, checkpoint)
            sources[kind] = folder / (kind + "_S{subject:02d}_seed{seed}.pt")
        with patch("course.runner.ROOT", folder), patch("course.runner.prepare_data", side_effect=fake_prepare), \
             patch("course.runner.read_subject", return_value=raw), patch.dict(exp.COURSE_CHECKPOINTS, sources):
            # ROOT 也用于源码摘要；提供真实的待测试源码。
            for name in ("course/runner.py", "course/figures.py", "course/__init__.py", "config.py", "data.py", "model.py", "train.py", "evaluate.py", "runtime.py"):
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes((ROOT / name).read_bytes())
            with patch("course.runner.fit_stages", wraps=exp.fit_stages) as fit:
                exp.run_experiments(cfg, ("noise", "fusion"), **options)
                assert [c.kwargs["model_name"] for c in fit.call_args_list] == ["Course-complex", "Course-feature_tf"]
            results = folder / "results/synthetic_course_check/0.024s"
            def records():
                return [row for task in ("fusion", "noise") for row in
                        json.loads((results / task / "records.json").read_text(encoding="utf-8"))]
            before = records()
            assert all(r["forward_ms_batch1"] > 0 and r["timing_trials"] == 240 for r in before)
            assert len(before) == 7
            fusion = [r for r in before if r["task"] == "fusion"]
            assert [r["method"] for r in fusion] == ["score_tf", "feature_tf"]
            assert fusion[0]["checkpoints"] == [str(folder / f"{k}_S01_seed2026.pt") for k in ("time", "complex")]
            noise = [r for r in before if r["task"] == "noise"]
            assert [(r["condition"], r["snr_db"], r["bandpass"]) for r in noise] == [
                ("added_noise", db, True) for db in (10., 10., 0., 0.)] + [("no_bandpass", "clean", False)]
            assert all(r["checkpoints"] == str(folder / "complex_S01_seed2026.pt") for r in noise[:-1])
            assert "complex_no_bandpass" in noise[-1]["checkpoints"]
            for name in ("fusion_paired.png", "fusion_diagnostics.svg", "noise_accuracy.png", "paired_differences.csv"):
                assert (results / ("noise" if name.startswith("noise") else "fusion") / name).exists()
            with patch("course.runner.fit_stages", wraps=exp.fit_stages) as fit:
                exp.run_experiments(cfg, ("noise", "fusion"), **options)
                fit.assert_not_called()
            after = records()
            assert before == after
            from evaluate import evaluate_checkpoint
            fused = results / 'models/S01_seed2026/feature_tf/reproducibility/checkpoint.pt'
            measured = []
            assert evaluate_checkpoint(fused, device='cpu', collected=measured)
            assert measured[0]["model"] == "Course-feature_tf"
            assert measured[0]["forward_ms_batch1"] > 0 and measured[0]["timing_trials"] == 240
            checkpoint = folder / "complex_S01_seed2026.pt"
            try:
                exp.load_baseline(checkpoint, replace(cfg, input_type='complex', duration=.028))
            except ValueError:
                pass
            else:
                raise AssertionError('Mismatched baseline accepted')
    (output / 'checks.json').write_text(json.dumps({'passed': ['Exact per-channel added SNR and shared deterministic noise',
        'Probability averaging matches independent reference', 'Both feature branches receive gradients; weights independent',
        'Two fusion methods, noise-only levels, clean bandpass ablation and plot outputs',
        'Completed-model reuse, result deduplication, external checkpoint guards'], 'formal_training_started': False}, indent=2), encoding='utf-8')
    print('PASS: course experiment checks', flush=True)


if __name__ == '__main__':
    main()
