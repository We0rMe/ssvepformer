"""课程实验三、四：融合、测试扰动与预处理对照。入口 train.py。"""
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from filelock import FileLock

from config import CONFIG, ROOT, COURSE_EXPERIMENTS, COURSE_CHECKPOINTS, COURSE_NAME, COURSE_WINDOW_SECONDS, SAVE_FIGURES, FUSION, NOISE
from data import prepare_data, split_subjects, read_subject, preprocess, network_input
from model import SSVEPformer
from train import seed_everything, fit_stages
from evaluate import save_checkpoint_test, evaluate_test, TIMING_FIELDS, itr, write_json, write_csv, plot_training
from .figures import plot_signal, summarize
from runtime import configure_device, device_info


class FeatureFusion(nn.Module):
    """两个独立编码器 → 沿时间/频点平均池化 → 各投影 D 维 → 拼接分类。"""
    def __init__(self, cfg, kinds=("time", "complex"), projection_dim=64):
        super().__init__()
        self.lengths = [replace(cfg, input_type=k).features for k in kinds]
        self.encoders, self.projections = nn.ModuleList(), nn.ModuleList()
        for kind in kinds:
            c = replace(cfg, input_type=kind)
            net = SSVEPformer(c)
            self.encoders.append(nn.Sequential(net.channel_combination, net.encoder))
            self.projections.append(nn.Sequential(nn.Linear(c.input_channels * c.spatial_multiplier,
                                                             projection_dim), nn.GELU(), nn.Dropout(c.dropout)))
        self.classifier = nn.Linear(2 * projection_dim, cfg.classes)
        for layer in [*(p[0] for p in self.projections), self.classifier]:
            nn.init.normal_(layer.weight, std=cfg.init_std)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        parts = torch.split(x, self.lengths, dim=-1)
        pooled = [projection(encoder(part).mean(-1))
                  for part, encoder, projection in zip(parts, self.encoders, self.projections)]
        return self.classifier(torch.cat(pooled, dim=1))


class ScoreFusion(nn.Module):
    """固定 0.5/0.5 概率融合；返回 log probability，兼容共用评估函数。"""
    def __init__(self, first, second, first_length):
        super().__init__()
        self.first, self.second, self.first_length = first, second, first_length

    def forward(self, x):
        a = self.first(x[..., :self.first_length]).log_softmax(-1)
        b = self.second(x[..., self.first_length:]).log_softmax(-1)
        return torch.logaddexp(a, b) - np.log(2.)


def add_noise(raw, cfg, snr_db, subject, repeat, seed):
    """全试次加噪；以未污染分析窗内逐试次逐通道功率定义附加 SNR。"""
    if snr_db is None:
        return np.array(raw, copy=True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, subject, repeat]))
    noise = rng.standard_normal(raw.shape)
    window = slice(cfg.start, cfg.start + cfg.samples)
    reference = np.mean(np.square(raw[..., window], dtype=np.float64), axis=-1, keepdims=True)
    power = np.mean(noise[..., window] ** 2, axis=-1, keepdims=True)
    noise *= np.sqrt(reference / np.maximum(power, 1e-30) / (10. ** (snr_db / 10.)))
    return raw + noise


def combined(features, cfg, kinds, test_only=False):
    arrays = []
    for kind in kinds:
        c = replace(cfg, input_type=kind)
        if test_only:
            arrays.append(torch.from_numpy(network_input(features[kind], c)))
        else:
            arrays.append(split_subjects(features[kind], c))
    if test_only:
        return torch.cat(arrays, dim=-1)
    x = torch.cat([a[0] for a in arrays], dim=-1)
    tx = torch.cat([a[2] for a in arrays], dim=-1)
    return x, arrays[0][1], tx, arrays[0][3]


def protocol_id(cfg, options):
    ignored = {"N", "M", "experiment", "evaluation_checkpoint", "evaluation_data_dir"}
    value = {"config": {k: v for k, v in cfg.to_dict().items() if k not in ignored},
             "options": options, "runtime": device_info(cfg),
             "baseline_templates": {k: str(v) if v is not None else None for k, v in COURSE_CHECKPOINTS.items()},
             "data_files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                            for p in sorted(cfg.data_dir.glob("*")) if p.is_file()],
             "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in ("course/runner.py", "data.py", "model.py", "train.py", "evaluate.py", "runtime.py")}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_baseline(path, cfg):
    """复用前两项的模型，逐项核对训练、输入、裁窗与预处理设置。"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("model_name", "SSVEPformer") not in ("SSVEPformer", "Course-" + cfg.input_type):
        raise ValueError("这里只接受基础 SSVEPformer checkpoint")
    if not state.get("training_complete", state["epoch"] == cfg.epochs):
        raise ValueError("不能使用未完成的 baseline")
    ignored = {"N", "M", "experiment", "device", "data_dir", "evaluation_checkpoint", "evaluation_data_dir",
               "cpu_threads", "monitor_every", "checkpoint_every"}
    current = json.loads(json.dumps(cfg.to_dict()))
    saved = json.loads(json.dumps(state["config"]))
    saved.setdefault("input_type", "complex")  # 最早的版本只实现频率轴复频谱
    saved.setdefault("bandpass", True)
    for key in ("test_subject", "seed", "duration", "epochs"):
        if key not in saved:
            raise ValueError(f"{path} 缺少 {key}，无法核对复用条件")
    for key, value in current.items():
        if key not in ignored and saved.get(key, value) != value:
            raise ValueError(f"{path} 的 {key} 为 {saved[key]!r}，当前实验需要 {value!r}")
    model = SSVEPformer(cfg).to(cfg.device)
    model.load_state_dict(state["model"])
    return model, str(Path(path).resolve())


BRANCHES = {"time": ("time",), "complex": ("complex",), "score_tf": ("time", "complex"),
            "feature_tf": ("time", "complex"), "feature_tt": ("time", "time"),
            "feature_ff": ("complex", "complex")}
TRAINED_METHOD = {kinds: method for method, kinds in BRANCHES.items() if method != "score_tf"}


def train_model(cfg, kinds, folder, get_feature, projection_dim, should_stop):
    """只在未训练时训练；后续噪声测试不会改变权重。"""
    key = TRAINED_METHOD[kinds]
    external_key = key if cfg.bandpass else key + "_no_bandpass"
    template = COURSE_CHECKPOINTS.get(external_key) if len(kinds) == 1 else None
    if template:
        if should_stop():
            raise InterruptedError("STOP")
        path = Path(str(template).format(subject=cfg.test_subject, seed=cfg.seed, duration=cfg.duration)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"未找到 {external_key} 的权重 {path}。请在 config.py 核对 ARCHIVE 或 COURSE_CHECKPOINTS。")
        proof = {"path": str(path), "bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns}
        folder.mkdir(parents=True, exist_ok=True)
        reference = folder / "external_checkpoint.json"
        if reference.exists() and json.loads(reference.read_text(encoding="utf-8")) != proof:
            raise ValueError("复用的 checkpoint 已变化，请换新的课程实验 name")
        model, source = load_baseline(path, replace(cfg, input_type=kinds[0]))
        write_json(reference, proof)
        return model, source
    details = folder / "reproducibility"
    details.mkdir(parents=True, exist_ok=True)
    c = replace(cfg, input_type=kinds[0] if len(kinds) == 1 else "complex")
    signature = protocol_id(c, {"kinds": kinds, "projection_dim": projection_dim})

    seed_everything(c)
    model = (SSVEPformer(c) if len(kinds) == 1 else FeatureFusion(c, kinds, projection_dim)).to(c.device)
    checkpoint = details / "checkpoint.pt"
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError(f"训练设置已变化，请更换 COURSE_NAME: {checkpoint}")
        if state["training_complete"]:
            model.load_state_dict(state["model"])
            return model, str(checkpoint)
        del state
    features = {kind: get_feature(kind, cfg.bandpass) for kind in set(kinds)}
    x, y, _, _ = combined(features, c, kinds)
    write_json(details / "protocol.json", {"config": c.to_dict(), "kinds": kinds,
               "projection_dim": projection_dim,
               "signature": signature, "parameters": sum(p.numel() for p in model.parameters())})
    cache_id = hashlib.sha256(json.dumps({k: features[k].shape for k in kinds}, sort_keys=True).encode()).hexdigest()
    history, selections = fit_stages(model, x, y, c, details, signature, cache_id,
                  [("train", None, c.epochs)], should_stop, print, plot_training,
                  model_name="Course-" + key)
    plot_training(history, folder)
    write_json(details / "selection.json", selections)
    return model, str(details / "checkpoint.pt")


def save_prediction(path, y, pred):
    write_csv(path, [{"trial": i, "target": int(a), "prediction": int(b), "correct": int(a == b)}
                     for i, (a, b) in enumerate(zip(y.tolist(), pred))])


class FoldModels:
    """一折内按需取得模型；时域、频域基线可供融合和噪声实验复用。"""
    def __init__(self, cfg, model_dir, feature_cache, stop):
        self.cfg = cfg
        self.directory = model_dir
        self.features = feature_cache
        self.models = {}
        self.stop = stop
        self.labels = torch.arange(cfg.classes).repeat(cfg.blocks)

    def feature(self, kind, bp):
        key = (kind, bp)
        if key not in self.features:
            self.features[key], _ = prepare_data(replace(self.cfg, input_type=kind, bandpass=bp))
        return self.features[key]

    def model(self, method, bp, width=0):
        key = (method, bp, width)
        if key in self.models:
            return self.models[key]
        if self.stop():
            raise InterruptedError("STOP")
        kinds = BRANCHES[method]
        if method == "score_tf":
            first, time_source = self.model("time", bp)
            second, freq_source = self.model("complex", bp)
            result = ScoreFusion(first, second, self.cfg.samples), [time_source, freq_source]
        else:
            name = method if bp else method + "_no_bandpass"
            result = train_model(replace(self.cfg, bandpass=bp), kinds, self.directory / name,
                                 self.feature, width, self.stop)
        self.models[key] = result
        return result


def run_fusion(fold, study, record):
    """在同一留出被试的干净窗口上比较配置中列出的解码器。"""
    raw = read_subject(fold.cfg.test_subject, fold.cfg)
    arrays = {k: preprocess(raw, replace(fold.cfg, input_type=k, bandpass=True))
              for k in {kind for method in study.methods for kind in BRANCHES[method]}}
    for method in study.methods:
        kinds = BRANCHES[method]
        width = study.projection_dim if method.startswith("feature_") else 0
        model, source = fold.model(method, True, width)
        record(method, True, "clean", 0, model, combined(arrays, fold.cfg, kinds, True), source)


def run_noise(fold, study, record, figure_dir=None):
    """两个独立对照。加噪保留标准带通；带通消融只使用干净试次。"""
    raw = read_subject(fold.cfg.test_subject, fold.cfg)
    for condition in study.conditions:
        bp = condition == "added_noise"
        cfg = replace(fold.cfg, input_type="complex", bandpass=bp)
        model, source = fold.model("complex", bp)
        if condition == "no_bandpass":
            x = torch.from_numpy(preprocess(raw, cfg))
            record("complex", False, "clean", 0, model, x, source)
            continue
        for snr in study.snr_db:
            for repeat in range(study.repeats):
                if fold.stop():
                    raise InterruptedError("STOP")
                noisy = add_noise(raw, cfg, snr, cfg.test_subject, repeat, study.seed)
                if figure_dir is not None and repeat == 0:
                    plot_signal(raw, noisy, cfg, figure_dir, cfg.test_subject, snr)
                x = torch.from_numpy(preprocess(noisy, cfg))
                record("complex", True, snr, repeat, model, x, source)


def validate_study(task, study):
    if task == "fusion":
        if not study.methods or len(set(study.methods)) != len(study.methods) or set(study.methods) - BRANCHES.keys():
            raise ValueError("FusionStudy.methods 含重复或不支持的方法，或没有选择方法")
        if study.projection_dim < 1:
            raise ValueError("projection_dim 必须为正整数")
        return len(study.methods)
    conditions = set(study.conditions)
    if not conditions or len(conditions) != len(study.conditions) or conditions - {"added_noise", "no_bandpass"}:
        raise ValueError("NOISE.conditions 应从 added_noise 和 no_bandpass 中选择，每项最多一次")
    count = int("no_bandpass" in conditions)
    if "added_noise" in conditions:
        if not isinstance(study.repeats, int) or study.repeats < 1:
            raise ValueError("噪声重复次数须为正整数")
        if not study.snr_db or len(set(study.snr_db)) != len(study.snr_db) or any(
            v is None or not np.isfinite(v) for v in study.snr_db
        ):
            raise ValueError("snr_db 应填写互不重复的有限 dB 数值")
        count += len(study.snr_db) * study.repeats
    return count


def execute_study(task, study, cfg, root, cache, plots):
    per_fold = validate_study(task, study)
    folder = root / task
    folder.mkdir(exist_ok=True)
    stop = lambda: (root / "STOP").exists() or (folder / "STOP").exists()
    manifest = folder / "protocol.json"
    signature = protocol_id(cfg, asdict(study))
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["signature"] != signature:
        raise ValueError(f"{task} 已有不同的实验设置，请更换 COURSE_NAME")
    if not manifest.exists():
        source = folder / "source"
        source.mkdir(exist_ok=True)
        for filename in ("config.py", "course/runner.py", "course/figures.py", "course/__init__.py", "data.py", "model.py", "train.py", "evaluate.py", "runtime.py"):
            (source / filename).parent.mkdir(parents=True, exist_ok=True)
            (source / filename).write_bytes((ROOT / filename).read_bytes())
    write_json(manifest, {"signature": signature, "config": cfg.to_dict(), "study": asdict(study)})
    records = folder / "records.json"
    rows = json.loads(records.read_text(encoding="utf-8")) if records.exists() else []
    keys = {(r["subject"], r["seed"], r["method"], r["bandpass"], r["snr_db"], r["repeat"]) for r in rows}
    expected = cfg.N * cfg.M * per_fold

    def save_progress(state):
        count = sum(1 for subject, seed, *_ in keys if subject <= cfg.N and cfg.seed <= seed < cfg.seed + cfg.M)
        write_json(folder / "progress.json", {"state": state, "completed_conditions": count,
                   "expected_conditions": expected, "complete": count == expected})

    save_progress("running")
    try:
        for single in cfg.runs():
            if stop():
                raise InterruptedError("STOP")
            subject = single.test_subject
            run_name = f"S{subject:02d}_seed{single.seed}"
            if sum(k[:2] == (subject, single.seed) for k in keys) == per_fold:
                continue
            fold = FoldModels(single, root / "models" / run_name, cache, stop)

            def record(method, bp, snr, repeat, model, x, source):
                key = (subject, single.seed, method, bp, snr, repeat)
                if key in keys:
                    return
                if stop():
                    raise InterruptedError("STOP")
                test, pred = evaluate_test(model, x, fold.labels, single)
                out = folder / "predictions" / run_name
                out.mkdir(parents=True, exist_ok=True)
                save_prediction(out / f"{method}_bp{bp}_{snr}_r{repeat}.csv", fold.labels, pred)
                rows.append({"task": task, "subject": subject, "seed": single.seed, "method": method,
                             "condition": method if task == "fusion" else ("added_noise" if bp else "no_bandpass"),
                             "bandpass": bp, "snr_db": snr, "repeat": repeat, "acc": test["acc_percent"],
                             "itr": itr(test["acc_percent"], single), "parameters": sum(p.numel() for p in model.parameters()),
                             "checkpoints": source, **{k: test[k] for k in TIMING_FIELDS}})
                write_json(records, rows)
                keys.add(key)
                save_progress("running")

            if task == "fusion":
                run_fusion(fold, study, record)
            else:
                example = plots and subject == 1 and single.seed == cfg.seed
                run_noise(fold, study, record, folder if example else None)
        save_progress("complete")
    except (KeyboardInterrupt, InterruptedError):
        save_progress("stopped")
        raise
    finally:
        if rows:
            summarize(rows, folder, plots)


def evaluate_course_checkpoint(path, state, data_dir, device, collected):
    """按随模型保存的输入分支重建旧版和新版课程模型，在干净测试折上复测。"""
    from config import Config
    if not state.get("training_complete", False):
        print(f"跳过未完成的模型 {path}")
        return False
    values = dict(state["config"])
    values["data_dir"] = Path(data_dir or values["data_dir"])
    values["device"] = device or values["device"]
    cfg = Config(**values)
    cfg.validate()
    configure_device(cfg)
    info = json.loads((path.parent / "protocol.json").read_text(encoding="utf-8"))
    kinds = tuple(info["kinds"])
    if len(kinds) == 1:
        model = SSVEPformer(replace(cfg, input_type=kinds[0]))
    else:
        width = state["model"]["classifier.weight"].shape[1] // 2
        model = FeatureFusion(cfg, kinds, width)
    model.load_state_dict(state["model"])
    model.to(cfg.device)
    raw = read_subject(cfg.test_subject, cfg)
    arrays = {k: preprocess(raw, replace(cfg, input_type=k)) for k in set(kinds)}
    x = combined(arrays, cfg, kinds, test_only=True)
    y = torch.arange(cfg.classes).repeat(cfg.blocks)
    method = TRAINED_METHOD[kinds]
    return save_checkpoint_test(path, cfg, model, x, y, "Course-" + method, "+".join(kinds), collected)



def run_experiments(cfg=CONFIG, tasks=None, *, fusion=FUSION, noise=NOISE, name=COURSE_NAME,
                    duration=COURSE_WINDOW_SECONDS, plots=SAVE_FIGURES):
    tasks = COURSE_EXPERIMENTS if tasks is None else tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - {"fusion", "noise"}:
        raise ValueError("请在 COURSE_EXPERIMENTS 中选择 fusion、noise 或两者")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("COURSE_NAME 应为单个目录名")
    if cfg.early_stopping:
        raise ValueError("实验三、四使用固定训练预算，请设置 early_stopping=False")
    cfg = replace(cfg, duration=duration)
    cfg.validate()
    configure_device(cfg)
    root = ROOT / "results" / name / f"{duration:g}s"
    root.mkdir(parents=True, exist_ok=True)
    studies = {"fusion": fusion, "noise": noise}
    for task in tasks:
        validate_study(task, studies[task])
    with FileLock(str(root / "campaign.lock"), timeout=0):
        cache = {}
        try:
            for task in tasks:
                execute_study(task, studies[task], cfg, root, cache, plots)
        except (KeyboardInterrupt, InterruptedError):
            print(f"实验已停止，结果保存在 {root}", flush=True)


if __name__ == "__main__":
    run_experiments()
