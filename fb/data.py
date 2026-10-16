"""FB 的滤波与缓存；复用原始数据读取、裁窗参数和 FFT，不复用基础带通后的信号。"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from filelock import FileLock
from scipy import signal
from scipy.io import loadmat

from config import ROOT
from data import channel_indices, complex_spectrum, read_subject, split_subjects
from evaluate import write_json


def filter_bank(cfg):
    """返回每个子带的 SOS 系数及可审查的设计参数。论文未完全规定的选择见 README。"""
    designs = []
    for band in range(1, cfg.subbands + 1):
        low = band * cfg.band_step
        passband = [low, cfg.band_high]
        stopband = [low - cfg.lower_transition, cfg.upper_stop]
        order, cutoff = signal.cheb1ord(passband, stopband, cfg.passband_ripple,
                                        cfg.stopband_attenuation, fs=cfg.sample_rate)
        sos = signal.cheby1(order, cfg.passband_ripple, cutoff, btype="bandpass",
                           fs=cfg.sample_rate, output="sos")
        designs.append({"band": band, "passband_hz": passband, "stopband_hz": stopband,
                        "prototype_order": int(order), "sos": sos.tolist()})
    return designs


def preprocess(trials, cfg, designs=None):
    """[试次, 通道, 时间] → [试次, 子带, 通道, 复频谱]。每个试次独立处理。"""
    features = []
    for design in designs or filter_bank(cfg):
        # 1. 从原始完整试次分别滤出子带；不串联子带，也不先做 8–64 Hz 基础带通。
        filtered = signal.sosfiltfilt(np.asarray(design["sos"]), trials, axis=-1)
        # 2. 在滤波之后裁窗，默认 [0.64, 1.64) 秒；属于使用整试次上下文的离线处理。
        windows = filtered[..., cfg.start:cfg.start + cfg.samples]
        # 3. 共用 FFT：n=1250 自动补零；保留 8–64 Hz，再按 [实部, 虚部] 拼接为 562 维。
        features.append(complex_spectrum(windows, cfg))
    return np.stack(features, axis=1)


def prepare_data(cfg):
    cfg.validate()
    keys = ("channels", "subjects", "classes", "blocks", "sample_rate", "duration", "start_seconds",
            "frequency_resolution", "frequency_range", "subbands", "band_step", "band_high",
            "lower_transition", "upper_stop", "passband_ripple", "stopband_attenuation")
    files = [cfg.data_dir / f"S{s}.mat" for s in range(1, cfg.subjects + 1)]
    files += [cfg.data_dir / "64-channels.loc", cfg.data_dir / "Freq_Phase.mat"]
    spec = {"settings": {key: getattr(cfg, key) for key in keys},
            "sources": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
            "code": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ("data.py", "fb/data.py")}}
    cache_id = "fb_" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    directory = ROOT / "cache" / cache_id
    directory.mkdir(parents=True, exist_ok=True)
    feature_file, audit_file = directory / "features.npy", directory / "audit.json"
    shape = (cfg.subjects, cfg.classes * cfg.blocks, cfg.subbands, len(cfg.channels), cfg.features)
    with FileLock(str(directory / "cache.lock"), timeout=0):
        if feature_file.exists() and audit_file.exists():
            features = np.load(feature_file, mmap_mode="r")
            if features.shape != shape or features.dtype != np.float32:
                raise ValueError("FB 缓存形状或类型不匹配")
            return features, json.loads(audit_file.read_text(encoding="utf-8"))
        designs = filter_bank(cfg)
        temporary = directory / "features.tmp.npy"
        features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
        subjects = []
        for subject in range(1, cfg.subjects + 1):
            trials = read_subject(subject, cfg)
            values = preprocess(trials, cfg, designs)
            if not np.isfinite(values).all():
                raise ValueError(f"S{subject} 的 FB 特征含非有限数值")
            features[subject - 1] = values
            subjects.append({"id": subject, "trials": len(trials)})
            print(f"FB 数据审查/预处理：{subject}/{cfg.subjects}", flush=True)
        features.flush()
        del features
        os.replace(temporary, feature_file)
        frequency = loadmat(cfg.data_dir / "Freq_Phase.mat")
        audit = {"cache_id": cache_id, "shape": shape, "subjects": subjects,
                 "zero_based_channels": channel_indices(cfg), "window": [cfg.start, cfg.start + cfg.samples],
                 "nfft": cfg.nfft, "filter_bank": designs, "spec": spec,
                 "protocol": "Independent whole-trial zero-phase filters; no dataset-fitted normalization",
                 "target_frequencies": frequency["freqs"].ravel().tolist(),
                 "target_phases": frequency["phases"].ravel().tolist()}
        write_json(audit_file, audit)
    return np.load(feature_file, mmap_mode="r"), audit
