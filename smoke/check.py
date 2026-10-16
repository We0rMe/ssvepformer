"""独立的快速检查；所有输出位于 smoke/results，不启动正式训练。"""

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG, ROOT
from data import complex_spectrum, prepare_data, preprocess, read_subject, split_subjects
from evaluate import evaluate, evaluate_checkpoint, itr, summarize, write_csv
from model import SSVEPformer
from train import run, run_identity, run_single, save_checkpoint, seed_everything, train_epoch


def check_schedule(output):
    cfg = replace(CONFIG, N=3, M=2, device="cpu")
    scheduled = list(cfg.runs())
    assert [(c.test_subject, c.seed) for c in scheduled] == [
        (s, seed) for s in range(1, 4) for seed in (2026, 2027)]
    assert len({c.output_dir for c in scheduled}) == 6
    assert run_identity(scheduled[0]) == run_identity(next(replace(cfg, N=35, M=3).runs()))
    assert run_identity(scheduled[0]) != run_identity(scheduled[1])
    for values in ({"N": 0}, {"N": 36}, {"M": 0}):
        try:
            replace(cfg, **values).validate()
        except AssertionError:
            pass
        else:
            raise AssertionError(f"未拒绝无效参数 {values}")
    # 被试编号充当特征，逐折验证始终排除当前留出被试。
    subjects = np.broadcast_to(np.arange(1, 36)[:, None, None, None], (35, 240, 1, 1)).copy()
    for single in replace(cfg, N=35, M=1).runs():
        x, y, test_x, test_y = split_subjects(subjects, single)
        assert len(y) == 8160 and len(test_y) == 240
        assert single.test_subject not in torch.unique(x).tolist()
        assert torch.unique(test_x).tolist() == [single.test_subject]

    with tempfile.TemporaryDirectory(dir=output) as temp, patch("config.ROOT", Path(temp)):
        with patch("train.prepare_data", return_value=(None, {})) as prepare, \
                patch("train.run_single", return_value=True) as execute, patch("train.summarize"):
            run(cfg)
            assert prepare.call_count == 1 and execute.call_count == 6
            assert [(c.args[0].test_subject, c.args[0].seed) for c in execute.call_args_list] == [
                (c.test_subject, c.seed) for c in scheduled]
        with patch("train.prepare_data", return_value=(None, {})), \
                patch("train.run_single", side_effect=[True, False]) as execute, patch("train.summarize"):
            run(cfg)
            assert execute.call_count == 2, "停止后不得继续其他种子/折"
        stop = cfg.output_dir / "reproducibility" / "STOP"
        stop.touch()
        with patch("train.prepare_data") as prepare:
            run(cfg)
            prepare.assert_not_called()
        stop.unlink()

        single = next(cfg.runs())
        single.output_dir.mkdir(parents=True)
        (single.output_dir / "metrics.csv").touch()
        with patch("train.check_manifest", return_value="checked"), patch("train.seed_everything") as init:
            assert run_single(single, None, {}, stop)
            init.assert_not_called()

        summary_cfg = replace(cfg, N=2)
        for single, accuracy in zip(summary_cfg.runs(), (50, 60, 70, 80)):
            single.output_dir.mkdir(parents=True, exist_ok=True)
            write_csv(single.output_dir / "metrics.csv", [{
                "model": "SSVEPformer", "test_subject": f"S{single.test_subject}", "seed": single.seed,
                "duration_s": single.duration, "ACC_percent": accuracy,
                "ITR_bits_min": itr(accuracy, single), "correct": int(2.4 * accuracy), "trials": 240}])
        assert len(summarize(summary_cfg)) == 4
        report = (summary_cfg.output_dir / "report.md").read_text(encoding="utf-8")
        assert "60.00 ± 14.14" in report and "70.00 ± 14.14" in report and "65.00 ± 7.07" in report
        assert len(summarize(replace(summary_cfg, N=1, M=1))) == 1
        assert "单次" in (summary_cfg.output_dir / "report.md").read_text(encoding="utf-8")
        assert summarize(replace(summary_cfg, seed=2030)) == []
        assert "完成 0 / 4" in (summary_cfg.output_dir / "report.md").read_text(encoding="utf-8")
        for single in summary_cfg.runs():
            settings = single.output_dir / "reproducibility" / "settings.json"
            settings.parent.mkdir(exist_ok=True)
            settings.write_text(json.dumps({"config": single.to_dict()}), encoding="utf-8")
        switched = replace(summary_cfg, device="cuda:0")
        assert len(summarize(switched, evaluation=True)) == 4
        try:
            summarize(switched)
        except ValueError:
            pass
        else:
            raise AssertionError("训练汇总必须拒绝混合设备；只有独立复核允许切换推理设备")


def check_evaluation(output):
    with tempfile.TemporaryDirectory(dir=output) as temp:
        directory = Path(temp) / "reproducibility"
        directory.mkdir()
        checkpoint = directory / "checkpoint.pt"
        saved_cfg = replace(CONFIG, test_subject=7, seed=2027, device="cpu")
        saved = {"config": saved_cfg.to_dict(), "epoch": saved_cfg.epochs,
                 "model": {}, "history": []}
        torch.save(saved, checkpoint)
        with patch("data.prepare_data", return_value=(None, {"cache_id": "test", "target_frequencies": []})) as prepare, \
                patch("data.split_subjects", return_value=(None, None, None, torch.zeros(1))), \
                patch("model.SSVEPformer") as constructor, patch("evaluate.evaluate", return_value=({}, [])), \
                patch("evaluate.save_results", return_value={"ACC_percent": 50, "ITR_bits_min": 20}) as save, \
                patch("evaluate.write_json"):
            assert evaluate_checkpoint(checkpoint)
            actual_cfg = constructor.call_args.args[0]
            assert actual_cfg.test_subject == 7 and actual_cfg.seed == 2027
            assert prepare.call_args.args[0].data_dir == saved_cfg.data_dir
            assert save.call_args.kwargs["output"] == Path(temp)
            relocated = Path(temp) / "relocated_data"
            assert evaluate_checkpoint(checkpoint, relocated)
            assert prepare.call_args.args[0].data_dir == relocated
            assert save.call_args.kwargs["output"] == directory / "reevaluation"
            with patch("evaluate.configure_device", return_value=torch.device("cpu")), \
                    patch("evaluate.device_info", return_value={"device": "cuda:0"}):
                assert evaluate_checkpoint(checkpoint, device="cuda:0")
                assert constructor.call_args.args[0].device == "cuda:0"
                assert save.call_args.kwargs["output"] == directory / "reevaluation"
            saved["epoch"] = 1
            torch.save(saved, checkpoint)
            prepare.reset_mock()
            assert not evaluate_checkpoint(checkpoint)
            prepare.assert_not_called()


def main():
    output = ROOT / "smoke" / "results"
    output.mkdir(exist_ok=True)
    cfg = replace(CONFIG, device="cpu")
    seed_everything(cfg)
    passed = []
    check_schedule(output)
    passed.append("N/M scheduling, all 35 held-out splits, cache reuse, STOP propagation, completed skip, separate SD axes")
    check_evaluation(output)
    passed.append("Evaluation uses checkpoint subject/seed, routes relocated data separately, skips incomplete models")
    t = np.arange(cfg.samples) / cfg.sample_rate
    f = complex_spectrum(np.cos(2 * np.pi * 10 * t)[None, None], cfg)
    index = round((10 - cfg.frequency_range[0]) / cfg.frequency_resolution)
    assert f.shape[-1] == 562
    np.testing.assert_allclose(f[0, 0, index], .5, atol=1e-6)
    np.testing.assert_allclose(f[0, 0, index + cfg.features // 2], 0, atol=1e-6)
    passed.append("FFT bin and real/imaginary ordering")

    toy_cfg = replace(cfg, subjects=3, test_subject=1)
    toy = np.arange(3 * 240 * 9 * 562, dtype=np.float32).reshape(3, 240, 9, 562)
    before = split_subjects(toy, toy_cfg)
    toy[0] *= -1
    after = split_subjects(toy, toy_cfg)
    assert torch.equal(before[0], after[0]) and not torch.equal(before[2], after[2])
    np.testing.assert_array_equal(before[3], np.tile(np.arange(40), 6))
    passed.append("Held-out subject perturbation does not change training inputs; labels align")

    features, audit = prepare_data(cfg)
    x, y, test_x, test_y = split_subjects(features, cfg)
    assert len(y) == 8160 and len(test_y) == 240
    trials = read_subject(2, cfg)[:2]
    np.testing.assert_array_equal(preprocess(trials, cfg),
                                  np.concatenate([preprocess(trials[i:i + 1], cfg) for i in range(2)]))
    np.testing.assert_array_equal(features[1, :2], preprocess(trials, cfg))
    passed.append("35 subjects; 8160/240 split; independent filtering and cache match raw data")

    with tempfile.TemporaryDirectory(dir=output) as temp:
        seed_everything(cfg)
        model = SSVEPformer(cfg)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                                    momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        small_x, small_y = x[:80], y[:80]
        loss, _ = train_epoch(model, optimizer, small_x, small_y, cfg, 1)
        assert np.isfinite(loss)
        assert model.encoder[0].mlp[1].in_features == model.encoder[0].mlp[1].out_features == cfg.features
        checkpoint = Path(temp) / "checkpoint.pt"
        save_checkpoint(checkpoint, model, optimizer, cfg, "check", 1, [])
        expected_loss, _ = train_epoch(model, optimizer, small_x, small_y, cfg, 2)
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        resumed = SSVEPformer(cfg)
        resumed.load_state_dict(saved["model"])
        restored_optimizer = torch.optim.SGD(resumed.parameters(), lr=cfg.learning_rate,
                                             momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        restored_optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        actual_loss, _ = train_epoch(resumed, restored_optimizer, small_x, small_y, cfg, 2)
        assert expected_loss == actual_loss
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
        _, expected = evaluate(model, small_x, small_y, cfg)
        _, actual = evaluate(resumed, small_x, small_y, cfg)
        np.testing.assert_array_equal(expected, actual)
    passed.append("Full-size model: finite update, F-to-F MLP, bit-exact SGD resume, identical predictions")
    assert itr(0, cfg) == itr(100 / 40, cfg) == 0
    assert abs(itr(100, cfg) - 40 * np.log2(40)) < 1e-8
    passed.append("ITR chance and perfect-accuracy limits")
    (output / "checks.json").write_text(json.dumps({"passed": passed, "cache_id": audit["cache_id"]}, indent=2), encoding="utf-8")
    for item in passed:
        print("PASS:", item)


if __name__ == "__main__":
    main()
