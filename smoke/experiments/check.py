"""无原始数据依赖的实验回归检查；只生成临时合成结果，不启动正式实验。"""
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import savemat

from config import CONFIG, ROOT
from data import network_input, preprocess, split_subjects, subject_partition, validation_data, prepare_data
from evaluate import evaluate, evaluate_checkpoint, summarize
from fb.evaluate import summarize as fb_summarize
from fb.config import CONFIG as FB_CONFIG
from fb.data import preprocess as fb_preprocess, prepare_data as fb_prepare
from fb.evaluate import evaluate_checkpoint as fb_evaluate_checkpoint
from fb.model import FBSSVEPformer
from fb.train import training_plan, run_single as fb_run_single
from model import SSVEPformer
from train import seed_everything, train_epoch, run_single
from train import EarlyStopper, fit_stages

TYPES = ("complex", "amplitude", "time", "complex_channels")


def tiny(fb=False, **kwargs):
    cfg = replace(FB_CONFIG if fb else CONFIG, device="cpu", cpu_threads=1, N=1, M=1,
                  channels=("Oz",), frequency_range=(8., 8.4), duration=.024,
                  encoder_blocks=1, kernel_size=3, head_multiplier=1, spatial_multiplier=1,
                  epochs=6, batch_size=4, checkpoint_every=1, monitor_every=2,
                  early_min_epochs=2, early_patience=2)
    if fb:
        cfg = replace(cfg, joint_epochs=6)
    return replace(cfg, **kwargs)


def check_features():
    counts = []
    raw = np.random.default_rng(7).normal(size=(2, 9, 1500))
    for fb in (False, True):
        for duration in (.5, 1., 2.):
            for kind in TYPES:
                cfg = replace(FB_CONFIG if fb else CONFIG, device="cpu", cpu_threads=1,
                              duration=duration, input_type=kind)
                cfg.validate()
                processed = (fb_preprocess if fb else preprocess)(raw, cfg)
                x = network_input(processed, cfg)
                assert x.shape[-2:] == (cfg.input_channels, cfg.features)
                if kind == "amplitude":
                    # 独立 FFT 参考；直接验证无平方、无倍增、无额外归一化。
                    time_cfg = replace(cfg, input_type="time")
                    window = (fb_preprocess if fb else preprocess)(raw, time_cfg)
                    fft = np.fft.rfft(window, n=cfg.nfft, axis=-1) / cfg.samples
                    first, last = [round(f / cfg.frequency_resolution) for f in cfg.frequency_range]
                    np.testing.assert_allclose(x, np.abs(fft[..., first:last+1]), rtol=2e-4, atol=1e-7)
                seed_everything(cfg)
                model = (FBSSVEPformer if fb else SSVEPformer)(cfg)
                optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
                loss, _ = train_epoch(model, optimizer, torch.from_numpy(x), torch.tensor([0, 1]), cfg, 1)
                assert np.isfinite(loss)
                counts.append({"model": "FB" if fb else "base", "input": kind, "seconds": duration,
                               "shape": list(x.shape[1:]), "parameters": sum(p.numel() for p in model.parameters())})
    for duration in (0, .25, 6):
        try:
            replace(CONFIG, duration=duration).validate()
        except AssertionError:
            pass
        else:
            raise AssertionError("非法窗口没有拒绝")
    return counts


def check_splits():
    data = np.broadcast_to(np.arange(1, 36, dtype=np.float32)[:, None, None, None], (35, 240, 1, 6)).copy()
    cfg = tiny(input_type="time", early_stopping=True)
    for subject in range(1, 36):
        c = replace(cfg, test_subject=subject)
        split = subject_partition(c)
        assert split == subject_partition(replace(c, seed=999, duration=2., input_type="amplitude"))
        assert len(split['train']) == 30 and len(split['validation']) == 4
        assert set(split['train']).isdisjoint(split['validation'])
        assert subject not in split['train'] + split['validation']
        x, y, tx, ty = split_subjects(data, c)
        vx, vy = validation_data(data, c)
        assert (len(y), len(vy), len(ty)) == (7200, 960, 240)
        assert torch.unique(x).tolist() == split['train']
        assert torch.unique(vx).tolist() == split['validation']
        assert torch.unique(tx).tolist() == [subject]
        expected = torch.arange(40).repeat(24)
        torch.testing.assert_close(vy, expected)
        fixed = replace(c, early_stopping=False)
        assert len(split_subjects(data, fixed)[1]) == 8160 and validation_data(data, fixed) is None
    paths = {replace(cfg, input_type=k, duration=d, early_stopping=e).output_dir
             for k in TYPES for d in (.5, 1., 2.) for e in (False, True)}
    assert len(paths) == 24


def check_cache(folder):
    raw_dir = folder / 'fake_raw'
    raw_dir.mkdir()
    for subject in range(1, 36):
        (raw_dir / f'S{subject}.mat').write_bytes(b'synthetic metadata only')
    (raw_dir / '64-channels.loc').write_text('1 0 0 Oz\n')
    savemat(raw_dir / 'Freq_Phase.mat', {'freqs': np.arange(40), 'phases': np.zeros(40)})
    for fb in (False, True):
        sandbox = folder / ('fb_cache' if fb else 'base_cache')
        (sandbox / 'fb').mkdir(parents=True)
        for name in ('data.py', 'fb/data.py'):
            (sandbox / name).write_bytes((ROOT / name).read_bytes())
        module, prepare = ('fb.data', fb_prepare) if fb else ('data', prepare_data)
        cfg = tiny(fb, data_dir=raw_dir)

        def zeros(raw, c, *args):
            return np.zeros((240, 3, 1, c.cache_features) if fb else (240, 1, c.cache_features), np.float32)

        with patch(f'{module}.ROOT', sandbox), patch(f'{module}.read_subject', return_value=np.zeros((1, 1, 1500))) as read, \
                patch(f'{module}.preprocess', side_effect=zeros):
            features, audit = prepare(cfg)
            del features
            assert read.call_count == 35
            read.reset_mock()
            for kind in ('amplitude', 'complex_channels'):
                features, again = prepare(replace(cfg, input_type=kind, seed=2027, early_stopping=True))
                del features
                assert again['cache_id'] == audit['cache_id']
            read.assert_not_called()
            for new_cfg in (replace(cfg, duration=.028), replace(cfg, input_type='time')):
                features, changed = prepare(new_cfg)
                del features
                assert changed['cache_id'] != audit['cache_id']
            assert read.call_count == 70


def check_controller():
    controller = EarlyStopper(2, 4, .01)
    for epoch, loss in enumerate((1., .999, .998, .997), 1):
        best, stop = controller.update(loss, epoch)
        assert best and stop == (epoch == 4)
    assert controller.best_epoch == 4  # tiny improvements still save true minimum
    restored = EarlyStopper(2, 4, .01, controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
    assert restored.update(.9, 5) == (True, False)


def check_training(folder, fb, early):
    cfg = tiny(fb, early_stopping=early)
    factory = FBSSVEPformer if fb else SSVEPformer
    shape = (4, 3, 1, 6) if fb else (4, 1, 6)
    x = torch.tensor(np.random.default_rng(5).normal(size=shape), dtype=torch.float32)
    y = torch.tensor([0, 1, 2, 3])
    validation = (x[:3].clone(), y[:3].clone()) if early else None
    stages = training_plan(cfg) if fb else [('train', None, cfg.epochs)]
    name = 'FB-SSVEPformer' if fb else 'SSVEPformer'
    first_weights = {}

    def train_and_capture(network, opt, xx, yy, c, epoch):
        result = train_epoch(network, opt, xx, yy, c, epoch)
        if (epoch - 1) % c.epochs == 0:
            first_weights[(epoch - 1) // c.epochs] = {k: v.detach().clone() for k, v in network.state_dict().items()}
        return result

    def controlled_eval(model, xx, yy, c):
        result, pred = evaluate(model, xx, yy, c)
        if len(yy) == 3:
            result['loss'] = 1.0  # known best at epoch 1, patience termination at 3
        return result, pred

    def run_at(details, callback):
        details.mkdir(parents=True, exist_ok=True)
        seed_everything(cfg)
        model = factory(cfg)
        with patch('train.evaluate', side_effect=controlled_eval), patch('train.train_epoch', side_effect=train_and_capture):
            history, selections = fit_stages(model, x, y, cfg, details, 'test', 'synthetic', stages,
                                             callback, lambda _: None, lambda *args: None, validation, name)
        return model, history, selections

    base = folder / f'{name}_{early}'
    details = base / 'baseline' / 'reproducibility'
    expected, history, selections = run_at(details, lambda: False)
    saved = torch.load(details / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert saved['training_complete'] and saved['stage_complete']
    assert all(s['selected_epoch'] == (1 if early else cfg.epochs) for s in selections)
    assert all(s['epochs_ran'] == (3 if early else cfg.epochs) for s in selections)
    if early:
        for key, value in first_weights[len(stages)-1].items():
            torch.testing.assert_close(expected.state_dict()[key], value, rtol=0, atol=0)
    # Before first batch, mid-stage, stage boundary and (FB) joint stage.
    for threshold in ([1, 2, 6, 20] if fb and early else [1, 2, 6]):
        calls = 0

        def stop():
            nonlocal calls
            calls += 1
            return calls >= threshold

        resume_dir = base / f'resume_{threshold}' / 'reproducibility'
        try:
            run_at(resume_dir, stop)
        except InterruptedError:
            pass
        else:
            raise AssertionError('STOP did not interrupt')
        actual, actual_history, actual_selections = run_at(resume_dir, lambda: False)
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
        clean = lambda rows: [{k: v for k, v in row.items() if k != 'seconds'} for row in rows]
        assert clean(actual_history) == clean(history) and actual_selections == selections
    # Completed early-stopped checkpoint evaluates even though epoch < max_epochs.
    prepare_target = 'fb.evaluate.prepare_data' if fb else 'data.prepare_data'
    split_target = 'fb.evaluate.split_subjects' if fb else 'data.split_subjects'
    evaluator = fb_evaluate_checkpoint if fb else evaluate_checkpoint
    with patch(prepare_target, return_value=(None, {'cache_id': 'synthetic', 'target_frequencies': list(range(40))})), \
            patch(split_target, return_value=(x, y, x, y)):
        assert evaluator(details / 'checkpoint.pt')
    saved['training_complete'] = False
    torch.save(saved, base / 'incomplete.pt')
    with patch(prepare_target) as prepare:
        assert not evaluator(base / 'incomplete.pt')
        prepare.assert_not_called()


def check_end_to_end(folder):
    for fb in (False, True):
        for early in (False, True):
            cfg = tiny(fb, early_stopping=early, input_type='amplitude', epochs=1, batch_size=512)
            if fb:
                cfg = replace(cfg, joint_epochs=1)
            shape = (35, 240, 3, 1, 6) if fb else (35, 240, 1, 6)
            features = np.random.default_rng(2).normal(size=shape).astype(np.float32)
            audit = {'cache_id': 'synthetic', 'target_frequencies': list(range(40))}
            with patch('config.ROOT', folder / 'integration'):
                single = next(cfg.runs())
                runner = fb_run_single if fb else run_single
                assert runner(single, features, audit, single.output_dir / 'STOP')
                assert runner(single, features, audit, single.output_dir / 'STOP')  # completed-run skip
                settings = json.loads((single.output_dir / 'reproducibility/settings.json').read_text(encoding='utf-8'))
                assert settings['split'] == subject_partition(single)
                state = torch.load(single.output_dir / 'reproducibility/checkpoint.pt', map_location='cpu', weights_only=False)
                assert state['training_complete'] and state['config']['early_stopping'] == early
                assert len((fb_summarize if fb else summarize)(cfg)) == 1


def main():
    torch.set_num_threads(1)
    output = ROOT / 'smoke/experiments/results'
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    counts = check_features()
    checks.append('Both models: four representations, three windows, 24 forward/backward checks and independent FFT magnitude reference')
    check_splits()
    check_controller()
    checks.append('All 35 folds: train/validation/test subject isolation, seed-independent splits, disabled-mode 8160 trials, path isolation')
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        folder = Path(temporary)
        check_cache(folder)
        checks.append('Spectral cache sharing, time/window cache separation for both pipelines')
        for fb in (False, True):
            for early in (False, True):
                check_training(folder, fb, early)
        checks.append('Early and fixed training: best-weight restoration, bit-exact interrupted resume, FB stage transitions, final evaluation guards')
        check_end_to_end(folder)
        checks.append('Four amplitude train/test/report workflows, manifests, source snapshots and completed-run skip')
    (output / 'checks.json').write_text(json.dumps({'passed': checks, 'models': counts,
        'device': 'cpu', 'formal_training_started': False}, indent=2, ensure_ascii=False), encoding='utf-8')
    for item in checks:
        print('PASS:', item, flush=True)


if __name__ == '__main__':
    main()
