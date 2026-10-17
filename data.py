"""Benchmark 读取、逐试次滤波/FFT、缓存及被试隔离。"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from scipy import signal
from scipy.io import loadmat

from config import ROOT


def channel_indices(cfg):
    lookup = {}
    for line in (cfg.data_dir / "64-channels.loc").read_text().splitlines():
        fields = line.split()
        if fields:
            lookup[fields[-1].upper()] = int(fields[0]) - 1
    return [lookup[name.upper()] for name in cfg.channels]


def read_subject(subject, cfg):
    raw = loadmat(cfg.data_dir / f"S{subject}.mat", variable_names=["data"])["data"]
    if raw.shape != (64, 1500, cfg.classes, cfg.blocks) or not np.isfinite(raw).all():
        raise ValueError(f"S{subject}: 数据形状或有限性检查失败")
    selected = raw[channel_indices(cfg)]
    # 第一个轴依次为 block 1 的 target 1..40、block 2 的 target 1..40，依此类推。
    trials = selected.transpose(3, 2, 0, 1).reshape(-1, len(cfg.channels), 1500)
    return trials


def complex_spectrum(windows, cfg):
    if windows.shape[-1] != cfg.samples:
        raise ValueError("时间窗长度不匹配")
    fft = np.fft.rfft(windows, n=cfg.nfft, axis=-1) / cfg.samples
    first, last = [round(f / cfg.frequency_resolution) for f in cfg.frequency_range]
    fft = fft[..., first:last + 1]
    return np.concatenate((fft.real, fft.imag), axis=-1).astype(np.float32)


def window_features(windows, cfg):
    """缓存内容：时域不补零、不 FFT；频域统一保留特征轴实虚拼接的基准格式。"""
    if windows.shape[-1] != cfg.samples:
        raise ValueError("时间窗长度不匹配")
    return windows.astype(np.float32) if cfg.input_type == "time" else complex_spectrum(windows, cfg)


def network_input(values, cfg):
    """每折训练/评估前转换一次；前导轴可包含 FB 子带，绝不跨试次或子带拼接。"""
    if cfg.input_type != "complex_channels":
        return values
    if values.shape[-2:] != (len(cfg.channels), 2 * cfg.spectral_bins):
        raise ValueError("通道拼接需要 [真实电极, 实部频点+虚部频点] 缓存")
    real, imaginary = np.split(values, 2, axis=-1)
    return np.concatenate((real, imaginary), axis=-2)  # Re(CH1..CHC), Im(CH1..CHC)


def preprocess(trials, cfg):
    if cfg.bandpass:
        sos = signal.butter(cfg.filter_order, cfg.frequency_range, btype="bandpass",
                            fs=cfg.sample_rate, output="sos")
        trials = signal.sosfiltfilt(sos, trials, axis=-1)
    return window_features(trials[..., cfg.start:cfg.start + cfg.samples], cfg)


def prepare_data(cfg):
    cfg.validate()
    keys = ("channels", "sample_rate", "duration", "start_seconds", "frequency_resolution",
            "frequency_range", "bandpass", "filter_order", "subjects", "classes", "blocks")
    files = [cfg.data_dir / f"S{s}.mat" for s in range(1, cfg.subjects + 1)]
    files += [cfg.data_dir / "64-channels.loc", cfg.data_dir / "Freq_Phase.mat"]
    spec = {"representation": "time" if cfg.input_type == "time" else "complex",
            "settings": {k: getattr(cfg, k) for k in keys},
            "sources": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
            "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cache_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    directory = ROOT / "cache" / cache_id
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "cache.lock"), timeout=600):
        return _prepare_cache(cfg, directory, cache_id, spec)


def _prepare_cache(cfg, directory, cache_id, spec):
    feature_file, audit_file = directory / "features.npy", directory / "audit.json"
    shape = (cfg.subjects, cfg.classes * cfg.blocks, len(cfg.channels), cfg.cache_features)
    if feature_file.exists() and audit_file.exists():
        features = np.load(feature_file, mmap_mode="r")
        if features.shape != shape or features.dtype != np.float32:
            raise ValueError("缓存形状不匹配")
        return features, json.loads(audit_file.read_text(encoding="utf-8"))
    temp = directory / "features.tmp.npy"
    features = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=shape)
    audit = {"cache_id": cache_id, "shape": shape, "zero_based_channels": channel_indices(cfg),
             "window": [cfg.start, cfg.start + cfg.samples], "nfft": None if cfg.input_type == "time" else cfg.nfft,
             "filtering": "Per-trial, whole-trial, zero-phase; offline temporal context",
             "normalization": "Raw microvolts; no fitted statistics" if cfg.input_type == "time"
                              else "FFT / window samples; no fitted dataset statistics",
             "subjects": [], "spec": spec}
    for subject in range(1, cfg.subjects + 1):
        trials = read_subject(subject, cfg)
        values = preprocess(trials, cfg)
        if not np.isfinite(values).all():
            raise ValueError(f"S{subject}: 预处理产生非有限值")
        features[subject - 1] = values
        audit["subjects"].append({"id": subject, "trials": len(trials),
                                   "raw_rms_uv": float(np.sqrt(np.mean(trials ** 2)))})
        if subject == 1 or subject % 5 == 0:
            print(f"数据审查/预处理：{subject}/{cfg.subjects}", flush=True)
    frequency = loadmat(cfg.data_dir / "Freq_Phase.mat")
    audit["target_frequencies"] = frequency["freqs"].ravel().tolist()
    audit["target_phases"] = frequency["phases"].ravel().tolist()
    features.flush()
    del features
    os.replace(temp, feature_file)
    audit_file.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return np.load(feature_file, mmap_mode="r"), audit


def split_subjects(features, cfg):
    test_index = cfg.test_subject - 1
    training_indices = [s for s in range(cfg.subjects) if s != test_index]
    assert test_index not in training_indices
    labels = np.tile(np.arange(cfg.classes, dtype=np.int64), cfg.blocks)
    train_x = network_input(np.concatenate([features[s] for s in training_indices]), cfg)
    test_x = network_input(np.array(features[test_index], copy=True), cfg)
    return (torch.from_numpy(train_x), torch.from_numpy(np.tile(labels, len(training_indices))),
            torch.from_numpy(test_x), torch.from_numpy(labels.copy()))
