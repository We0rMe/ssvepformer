"""输入对比的独立检查；合成实验只写 smoke/inputs/results，不启动正式训练。"""

import importlib.util
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy import signal

from config import CONFIG, ROOT
from data import network_input, prepare_data, preprocess, read_subject, split_subjects
from evaluate import evaluate_checkpoint, summarize, write_json
from fb.config import CONFIG as FB_CONFIG
from fb.data import filter_bank, prepare_data as prepare_fb, preprocess as preprocess_fb
from fb.evaluate import evaluate_checkpoint as evaluate_fb_checkpoint
from fb.model import FBSSVEPformer
from fb.train import run_single as run_fb_single
from model import SSVEPformer
from train import run_single, seed_everything, train_epoch

TYPES = ("complex", "time", "complex_channels")


def check_features():
    cfg = replace(CONFIG, device="cpu", cpu_threads=1, input_type="complex")
    raw = read_subject(2, cfg)[:2]
    basic, bank = preprocess(raw, cfg), preprocess_fb(raw, FB_CONFIG)
    counts = []
    for kind in TYPES:
        for fb in (False, True):
            c = replace(FB_CONFIG if fb else cfg, input_type=kind, device="cpu", cpu_threads=1)
            processed = preprocess_fb(raw, c) if fb else preprocess(raw, c)
            if kind != "time":
                np.testing.assert_array_equal(processed, bank if fb else basic)
            else:
                designs = [np.array(d["sos"]) for d in filter_bank(c)] if fb else [
                    signal.butter(c.filter_order, c.frequency_range, btype="bandpass", fs=c.sample_rate, output="sos")]
                expected = [signal.sosfiltfilt(sos, raw, axis=-1)[..., c.start:c.start+c.samples].astype(np.float32)
                            for sos in designs]
                np.testing.assert_array_equal(processed, np.stack(expected, axis=1) if fb else expected[0])
            x = network_input(processed, c)
            assert x.shape[-2:] == (c.input_channels, c.features)
            if kind == "complex_channels":
                n = len(c.channels)
                np.testing.assert_array_equal(np.concatenate((x[..., :n, :], x[..., n:, :]), axis=-1), processed)
                # 通道分组与交错只是输入通道的置换；同样置换首层权重可保持输出。
                seed_everything(c)
                net = SSVEPformer(c).eval()
                values = torch.from_numpy(x[:, 0] if fb else x)
                order = torch.tensor([j for i in range(n) for j in (i, n+i)])
                with torch.no_grad():
                    expected = net(values)
                    conv = net.channel_combination[0]
                    conv.weight.copy_(conv.weight[:, order].clone())
                    torch.testing.assert_close(net(values[:, order]), expected, rtol=1e-5, atol=1e-6)
            seed_everything(c)
            net = FBSSVEPformer(c) if fb else SSVEPformer(c)
            optimizer = torch.optim.SGD(net.parameters(), lr=c.learning_rate, momentum=c.momentum)
            loss, _ = train_epoch(net, optimizer, torch.from_numpy(x), torch.tensor([0, 1]), c, 1)
            assert np.isfinite(loss)
            counts.append({"model": "FB" if fb else "SSVEPformer", "input_type": kind,
                           "shape": list(x.shape[1:]), "parameters": sum(p.numel() for p in net.parameters())})
    return counts


def check_cache(cfg, output, fb):
    module = "fb.data" if fb else "data"
    prepare = prepare_fb if fb else prepare_data
    with tempfile.TemporaryDirectory(dir=output) as temp:
        sandbox = Path(temp)
        if fb:
            (sandbox / "fb").mkdir()
            for name in ("data.py", "fb/data.py"):
                (sandbox / name).write_bytes((ROOT / name).read_bytes())
        shape = (240, 3, 1, 6) if fb else (240, 1, 6)
        with patch(f"{module}.ROOT", sandbox), patch(f"{module}.read_subject", return_value=np.zeros((1, 1, 1500))) as read, \
                patch(f"{module}.preprocess", return_value=np.zeros(shape, np.float32)):
            first, audit = prepare(replace(cfg, input_type="complex"))
            assert read.call_count == 35
            read.reset_mock()
            second, again = prepare(replace(cfg, input_type="complex_channels", seed=2027))
            assert audit["cache_id"] == again["cache_id"] and read.call_count == 0
            third, temporal = prepare(replace(cfg, input_type="time"))
            assert temporal["cache_id"] != audit["cache_id"] and read.call_count == 35
            del first, second, third


def check_end_to_end(output):
    for fb in (False, True):
        cfg = replace(FB_CONFIG if fb else CONFIG, device="cpu", cpu_threads=1, N=1, M=1,
                      channels=("Oz",), frequency_range=(8., 8.4), duration=.024,
                      epochs=1, encoder_blocks=1, kernel_size=3, spatial_multiplier=1,
                      head_multiplier=1, batch_size=512, monitor_every=1, checkpoint_every=1)
        if fb:
            cfg = replace(cfg, joint_epochs=1)
        check_cache(cfg, output, fb)
        shape = (35, 240, 3, 1, 6) if fb else (35, 240, 1, 6)
        features = np.random.default_rng(37).normal(size=shape).astype(np.float32)
        audit = {"cache_id": "synthetic", "target_frequencies": list(range(40))}
        with tempfile.TemporaryDirectory(dir=output) as temp, patch("config.ROOT", Path(temp)):
            for kind in TYPES:
                plan = replace(cfg, input_type=kind)
                single = next(plan.runs())
                single.validate()
                assert single.output_dir.parent == plan.output_dir
                train_x, _, test_x, _ = split_subjects(features, single)
                perturbed = features.copy()
                perturbed[0] += 10
                after_x, _, after_test, _ = split_subjects(perturbed, single)
                torch.testing.assert_close(after_x, train_x, rtol=0, atol=0)
                assert not torch.equal(after_test, test_x)
                runner = run_fb_single if fb else run_single
                assert runner(single, features, audit, plan.output_dir / "STOP")
                checkpoint = single.output_dir / "reproducibility" / "checkpoint.pt"
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert saved["config"]["input_type"] == kind
                settings = json.loads((checkpoint.parent / "settings.json").read_text(encoding="utf-8"))
                assert settings["model"]["parameters"] > 0
                target = "fb.evaluate.prepare_data" if fb else "data.prepare_data"
                with patch(target, return_value=(features, audit)):
                    assert (evaluate_fb_checkpoint if fb else evaluate_checkpoint)(checkpoint)
                assert len(summarize(plan)) == 1
            assert len(list((Path(temp) / "results" / cfg.experiment).iterdir())) == 3


def check_saved_baseline():
    """有服务器快照时，验证原 complex 特征与模型计算逐值不变；从不写服务器结果。"""
    source = ROOT / "results_server/ssvepformer_loso_cuda/S01_seed2026/reproducibility/source"
    if not source.exists():
        return "SKIPPED: server source snapshot unavailable"

    def load(name):
        spec = importlib.util.spec_from_file_location(f"baseline_{name}", source / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    cfg = replace(CONFIG, device="cpu", cpu_threads=1, input_type="complex")
    raw = read_subject(2, cfg)[:2]
    old = load("data").preprocess(raw, cfg)
    np.testing.assert_array_equal(preprocess(raw, cfg), old)
    seed_everything(cfg)
    original = load("model").SSVEPformer(cfg).eval()
    current = SSVEPformer(cfg).eval()
    current.load_state_dict(original.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(current(torch.from_numpy(old)), original(torch.from_numpy(old)), rtol=0, atol=0)
    return "PASS: original complex preprocessing, state shapes and logits unchanged"


def main():
    torch.set_num_threads(1)
    output = ROOT / "smoke/inputs/results"
    output.mkdir(parents=True, exist_ok=True)
    counts = check_features()
    check_end_to_end(output)
    baseline = check_saved_baseline()
    write_json(output / "checks.json", {"checks": ["Both models: all three inputs, gradients, shapes and channel permutation",
               "Spectral cache reuse; time cache isolation; held-out data isolation",
               "Six synthetic train/evaluate/report workflows and checkpoint input types", baseline],
               "models": counts, "formal_training_started": False, "device": "cpu"})
    print("PASS: three input representations for both models;", baseline, flush=True)


if __name__ == "__main__":
    main()
