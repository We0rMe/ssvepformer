"""实验三、四的汇总表与论文插图。"""
import csv
from dataclasses import replace

import numpy as np

from data import preprocess
from evaluate import plt, write_csv


def plot_signal(raw, noisy, cfg, folder, subject, snr):
    """固定第一个目标、第一块、Oz 电极；不根据分类结果挑选样本。"""
    ch = cfg.channels.index("Oz") if "Oz" in cfg.channels else 0
    window = slice(cfg.start, cfg.start + cfg.samples)
    clean, changed = raw[0, ch, window], noisy[0, ch, window]
    filtered = []
    for value in (raw, noisy):
        result = preprocess(value[:1], replace(cfg, input_type="time", bandpass=True))
        filtered.append(result[0, ch])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    t = np.arange(cfg.samples) / cfg.sample_rate
    for a, b, row, label in ((clean, changed, axes[0], "Before bandpass"),
                              (*filtered, axes[1], "After bandpass")):
        row[0].plot(t, a, label="Recorded", alpha=.85)
        row[0].plot(t, b, label="Noisy recording", alpha=.65)
        freq = np.fft.rfftfreq(cfg.samples, 1 / cfg.sample_rate)
        for values, name in ((a, "Recorded"), (b, "Noisy recording")):
            row[1].plot(freq, np.abs(np.fft.rfft(values)) / cfg.samples, label=name)
        row[0].set(xlabel="Time (s)", ylabel="Amplitude (uV)", title=label)
        row[1].set(xlabel="Frequency (Hz)", ylabel="FFT magnitude (uV)", xlim=(0, 100))
        for ax in row:
            ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(f"S{subject:02d}, block 1, target 1, {cfg.channels[ch]}, added SNR={snr} dB")
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"noise_S{subject:02d}_{snr}dB.{suffix}", dpi=180)
    plt.close(fig)
    np.savez_compressed(folder / f"noise_S{subject:02d}_{snr}dB.npz", time=t, recorded=clean,
                        noisy=changed, filtered_recorded=filtered[0], filtered_noisy=filtered[1])


def summarize(rows, folder, plots=True):
    """先平均同被试同种子的噪声重复，再平均种子，最终按被试统计。"""
    write_csv(folder / "metrics.csv", rows)
    groups = {}
    for row in rows:
        key = (row["task"], row["method"], row["bandpass"], row["snr_db"])
        groups.setdefault(key, {}).setdefault(row["subject"], {}).setdefault(row["seed"], []).append(row)
    summaries = []
    for (task, method, bp, snr), subjects in groups.items():
        def subject_means(field):
            return [np.mean([np.mean([r[field] for r in repeat_rows]) for repeat_rows in seeds.values()])
                    for seeds in subjects.values()]
        values = subject_means("acc")
        summaries.append({"task": task, "method": method,
                          "condition": method if task == "fusion" else ("added_noise" if bp else "no_bandpass"), "bandpass": bp, "snr_db": snr,
                          "subjects": len(values), "acc_mean": float(np.mean(values)),
                          "itr_mean": float(np.mean(subject_means("itr"))),
                          "forward_ms_batch1": float(np.mean(subject_means("forward_ms_batch1"))),
                          "subject_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    write_csv(folder / "summary.csv", summaries)
    if not plots:
        return
    fusion = [r for r in rows if r["task"] == "fusion"]
    if fusion:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        methods = list(dict.fromkeys(r["method"] for r in fusion))
        subjects = sorted({r["subject"] for r in fusion})
        for subject in subjects:
            values = [np.mean([r["acc"] for r in fusion if r["subject"] == subject and r["method"] == m])
                      for m in methods]
            ax.plot(methods, values, ".-", alpha=.2, color="gray")
        means = [np.mean([r["acc"] for r in fusion if r["method"] == m]) for m in methods]
        ax.plot(methods, means, "o-", color="tab:blue", label="Mean")
        ax.set(ylabel="ACC (%)", title="Fusion: paired subjects", ylim=(0, 100)); ax.legend()
        for suffix in ("png", "svg"):
            fig.savefig(folder / f"fusion_paired.{suffix}", dpi=180)
        plt.close(fig)
        fusion_diagnostics(fusion, folder)
    noise = [s for s in summaries if s["condition"] == "added_noise"]
    if noise:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        levels = list(dict.fromkeys(r["snr_db"] for r in noise))
        for method, bp in dict.fromkeys((s["method"], s["bandpass"]) for s in noise):
            values = [next((s["acc_mean"] for s in noise if s["method"] == method and s["bandpass"] == bp
                            and s["snr_db"] == level), np.nan) for level in levels]
            ax.plot(range(len(levels)), values, "o-", label=f"{method}, BP={bp}")
        ax.set(xticks=range(len(levels)), xticklabels=[str(v) for v in levels], xlabel="Added SNR (dB)",
               ylabel="ACC (%)", ylim=(0, 100)); ax.legend(); ax.grid(alpha=.2)
        for suffix in ("png", "svg"):
            fig.savefig(folder / f"noise_accuracy.{suffix}", dpi=180)
        plt.close(fig)


def fusion_diagnostics(rows, folder):
    pairs = (("feature_tf", "score_tf"), ("score_tf", "complex"), ("feature_tf", "feature_tt"), ("feature_tf", "feature_ff"))
    methods = {r["method"] for r in rows}
    pairs = [(a, b) for a, b in pairs if {a, b} <= methods]
    show_overlap = {"time", "complex"} <= methods
    if not pairs and not show_overlap:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    records = []
    for index, (first, second) in enumerate(pairs):
        subject_values = {}
        for row in rows:
            if row["method"] != first:
                continue
            matches = [r for r in rows if r["method"] == second and r["subject"] == row["subject"] and r["seed"] == row["seed"]]
            if matches:
                subject_values.setdefault(row["subject"], []).append(row["acc"] - matches[0]["acc"])
        values = [float(np.mean(v)) for v in subject_values.values()]
        if values:
            axes[0].scatter(np.full(len(values), index), values, alpha=.5)
            axes[0].scatter(index, np.mean(values), marker="D", color="black")
            records += [{"comparison": first + "-" + second, "subject": s, "delta_acc_pp": float(np.mean(v))}
                        for s, v in subject_values.items()]
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set(xticks=range(len(pairs)), xticklabels=[f"{a} - {b}" for a, b in pairs],
                ylabel="Paired ACC difference (percentage points)")
    if not pairs:
        axes[0].set_visible(False)
    counts = np.zeros(4)
    for row in rows:
        if row["method"] != "time":
            continue
        base = folder / "predictions" / f"S{row['subject']:02d}_seed{row['seed']}"
        paths = [base / f"{kind}_bpTrue_clean_r0.csv" for kind in ("time", "complex")]
        if not all(p.exists() for p in paths):
            continue
        flags = []
        for path in paths:
            with path.open(encoding="utf-8-sig") as stream:
                flags.append(np.array([int(r["correct"]) for r in csv.DictReader(stream)]))
        a, b = flags
        counts += [np.sum((a == 1) & (b == 1)), np.sum((a == 1) & (b == 0)),
                   np.sum((a == 0) & (b == 1)), np.sum((a == 0) & (b == 0))]
    labels = ["Both correct", "Only time", "Only frequency", "Both wrong"]
    if counts.sum():
        axes[1].bar(labels, counts / counts.sum() * 100)
    else:
        axes[1].set_visible(False)
    axes[1].set(ylabel="Trial outcomes (%)", title="Repeated seeds describe the same held-out trials")
    axes[1].tick_params(axis="x", rotation=20)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"fusion_diagnostics.{suffix}", dpi=180)
    plt.close(fig)
    if records:
        write_csv(folder / "paired_differences.csv", records)
    if counts.sum():
        write_csv(folder / "error_overlap.csv", [{"outcome": label, "count_across_runs": int(n)} for label, n in zip(labels, counts)])
