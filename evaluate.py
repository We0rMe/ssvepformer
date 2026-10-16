"""评估、简洁报告与图像；也可直接运行以复核已保存模型。"""

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from config import CONFIG, ROOT, Config
from runtime import configure_device, device_info

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def itr(accuracy, cfg):
    p = accuracy / 100
    if not 0 <= p <= 1:
        raise ValueError("准确率必须在 0–100 之间")
    if p <= 1 / cfg.classes:
        return 0.0
    bits = math.log2(cfg.classes)
    if p < 1:
        bits += p * math.log2(p) + (1 - p) * math.log2((1 - p) / (cfg.classes - 1))
    return 60 * bits / (cfg.duration + cfg.gaze_seconds)


@torch.inference_mode()
def evaluate(model, x, y, cfg):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    loss = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, len(y), cfg.batch_size):
        batch_y = y[start:start + cfg.batch_size].to(device)
        logits = model(x[start:start + cfg.batch_size].to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("评估输出含非有限值")
        loss += torch.nn.functional.cross_entropy(logits, batch_y, reduction="sum").to(torch.float64)
        predictions.append(logits.argmax(1))
    predictions = torch.cat(predictions).cpu().numpy()
    correct = int((predictions == y.cpu().numpy()).sum())
    return {"loss": loss.item() / len(y), "acc_percent": 100 * correct / len(y),
            "correct": correct, "total": len(y)}, predictions


def plot_training(history, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epoch = [row["epoch"] for row in history]
    monitor = [row for row in history if row["monitor_loss"] is not None]
    for ax, online_key, monitor_key, label in zip(
            axes, ("train_loss", "train_acc"), ("monitor_loss", "monitor_acc"), ("Cross-entropy", "ACC (%)")):
        ax.plot(epoch, [r[online_key] for r in history], label="Training batches (dropout ON)", marker="." if len(epoch) == 1 else None)
        ax.plot([r["epoch"] for r in monitor], [r[monitor_key] for r in monitor], ".-",
                label="Fixed training subset (dropout OFF)")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[1].set_ylim(0, 100)
    fig.savefig(directory / "training.png", dpi=150)
    plt.close(fig)


def save_results(cfg, history, test, train, predictions, labels, frequencies, output=None):
    output = output or cfg.output_dir
    details = output / "reproducibility"
    details.mkdir(parents=True, exist_ok=True)
    metric = {"model": "SSVEPformer", "test_subject": f"S{cfg.test_subject}", "seed": cfg.seed,
              "duration_s": cfg.duration, "ACC_percent": round(test["acc_percent"], 4),
              "ITR_bits_min": round(itr(test["acc_percent"], cfg), 4),
              "correct": test["correct"], "trials": test["total"]}
    records = [{"block": i // cfg.classes + 1, "target": int(y) + 1,
                "frequency_hz": frequencies[int(y)], "prediction": int(p) + 1,
                "correct": int(y == p)} for i, (y, p) in enumerate(zip(labels, predictions))]
    write_csv(details / "predictions.csv", records)
    plot_training(history, output)
    confusion = np.zeros((cfg.classes, cfg.classes), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(confusion, vmin=0, vmax=cfg.blocks, cmap="Blues", origin="upper")
    ticks = np.arange(0, cfg.classes, 5)
    ax.set(xticks=ticks, xticklabels=ticks + 1, yticks=ticks, yticklabels=ticks + 1,
           xlabel="Predicted target (MAT order)", ylabel="True target (MAT order)",
           title=f"S{cfg.test_subject}: ACC {test['acc_percent']:.2f}%")
    fig.colorbar(im, ax=ax, label="Trials")
    fig.savefig(output / "confusion.png", dpi=150)
    plt.close(fig)
    minutes = sum(row["seconds"] for row in history) / 60
    text = f"""# SSVEPformer · S{cfg.test_subject} 测试结果

| 指标 | 结果 |
|---|---:|
| 测试 ACC | {test['acc_percent']:.2f}% |
| 测试 ITR | {itr(test['acc_percent'], cfg):.2f} bits/min |
| 正确 / 总试次 | {test['correct']} / {test['total']} |
| 最终训练集 ACC（关闭 Dropout） | {train['acc_percent']:.2f}% |
| 最后一轮训练 ACC（开启 Dropout） | {history[-1]['train_acc']:.2f}% |

设置：S{cfg.test_subject} 测试，其余 {cfg.subjects - 1} 名被试训练；种子 {cfg.seed}；{cfg.duration:g} 秒；{cfg.epochs} epoch；设备 {cfg.device}。
训练与训练集监测耗时约 {minutes:.1f} 分钟。固定最后一轮模型，测试集未参与早停或选模。

`training.png` 中蓝线是各训练批次开启 Dropout 的指标；橙线是固定训练子集关闭 Dropout 的监测指标，**不是验证集**。
测试集只在训练结束后评估。单次、单被试实验不报告“均值 ± 标准差”。

论文的 80.40% ACC、149.95 bits/min 是 **35 名被试平均结果**，与当前 S{cfg.test_subject} 单被试结果不构成同口径复现结论。
详细配置、训练逐轮 CSV、预测明细、模型和代码快照均在 `reproducibility/`。
"""
    
    (output / "report.md").write_text(text, encoding="utf-8")
    write_csv(output / "metrics.csv", [metric])
    return metric


def summarize(cfg, *, evaluation=False):
    """按被试与种子分别汇总；未完成的计划明确标为部分结果。"""
    rows = []
    for single in cfg.runs():
        path = single.output_dir / "metrics.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            row, = list(csv.DictReader(stream))
        assert row["test_subject"] == f"S{single.test_subject}" and int(row["seed"]) == single.seed
        assert int(row["trials"]) == cfg.classes * cfg.blocks
        assert abs(float(row["duration_s"]) - cfg.duration) < 1e-8
        settings = single.output_dir / "reproducibility" / "settings.json"
        if settings.exists():
            saved = json.loads(settings.read_text(encoding="utf-8"))["config"]
            current = json.loads(json.dumps(single.to_dict()))
            ignored = {"N", "M", "experiment", "evaluation_checkpoint", "evaluation_data_dir"}
            if evaluation:
                ignored.add("device")  # 更换推理设备时仍汇总原实验指标；训练入口继续严格检查设备。
            if any(saved[k] != current[k] for k in saved if k not in ignored):
                raise ValueError(f"{single.output_dir.name} 的设置不匹配，请使用新的 experiment")
        rows.append(row)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "test_subject", "seed", "duration_s", "ACC_percent", "ITR_bits_min", "correct", "trials"]
    write_csv(cfg.output_dir / "summary.csv", rows, fields=fields)
    complete = len(rows) == cfg.N * cfg.M
    lines = ["# SSVEPformer · LOSO 汇总", "",
             f"完成 {len(rows)} / {cfg.N * cfg.M} 次实验；计划 S1–S{cfg.N}，每折 {cfg.M} 个种子。",
             "完整计划已完成。" if complete else "**部分结果：尚不能作为完整计划的最终结果。**", "",
             "下表每行在同一种子下，对已完成被试分别计算 ACC、ITR 的均值 ± 样本标准差（ddof=1）。",
             "ITR 先逐被试计算再平均；仅有一个被试时不计算标准差。", "",
             "| 种子 | 已完成被试 / 计划 | ACC (%) | ITR (bits/min) |", "|---|---:|---:|---:|"]
    seed_means = []

    def mean_sd(values):
        return f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}" if len(values) > 1 else f"{values[0]:.2f}（单次）"

    for seed in range(cfg.seed, cfg.seed + cfg.M):
        selected = [r for r in rows if int(r["seed"]) == seed]
        if not selected:
            lines.append(f"| {seed} | 0 / {cfg.N} | — | — |")
            continue
        acc = [float(r["ACC_percent"]) for r in selected]
        speed = [float(r["ITR_bits_min"]) for r in selected]
        lines.append(f"| {seed} | {len(selected)} / {cfg.N} | {mean_sd(acc)} | {mean_sd(speed)} |")
        seed_means.append((np.mean(acc), np.mean(speed)))
    if complete and cfg.M > 1:
        means = np.array(seed_means)
        lines += ["", f"每个种子先对 {cfg.N} 名被试取平均，再在 {cfg.M} 个种子间统计：",
                  f"ACC：{mean_sd(means[:, 0])}%；ITR：{mean_sd(means[:, 1])} bits/min。",
                  "这里的标准差反映种子间变化，与上表的被试间标准差含义不同。"]
    lines += ["", "每折均使用固定最后一轮模型；留出被试不参与该折训练或选模。",
              "35 折全完成时才构成全被试 LOSO 评估。折间训练集重叠，不能把 N×M 个结果视作相互独立的样本。",
              "各折目录包含关键指标与图像；配置、checkpoint、明细和代码快照在各自 reproducibility/ 中。"]
    (cfg.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def evaluate_checkpoint(checkpoint, data_dir=None, device=None):
    from data import prepare_data, split_subjects
    from model import SSVEPformer
    checkpoint = Path(checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = state["config"].copy()
    trained_device = values.get("device", "cpu")
    values["device"] = device or trained_device
    for key in ("data_dir", "evaluation_checkpoint", "evaluation_data_dir"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    if data_dir is not None:
        values["data_dir"] = Path(data_dir)
    cfg = Config(**values)
    if state["epoch"] != cfg.epochs:
        print(f"跳过未完成模型：{checkpoint} ({state['epoch']}/{cfg.epochs})", flush=True)
        return False
    cfg.validate()
    selected_device = configure_device(cfg)
    features, audit = prepare_data(cfg)
    x, y, test_x, test_y = split_subjects(features, cfg)
    model = SSVEPformer(cfg).to(selected_device)
    model.load_state_dict(state["model"])
    train, _ = evaluate(model, x, y, cfg)
    test, predictions = evaluate(model, test_x, test_y, cfg)
    # 更换数据目录或推理设备的复核另存，避免替换原实验的正式指标。
    same_context = data_dir is None and cfg.device == trained_device
    output = checkpoint.parent.parent if same_context else checkpoint.parent / "reevaluation"
    metric = save_results(cfg, state["history"], test, train, predictions,
                          test_y.numpy(), audit["target_frequencies"], output=output)
    write_json(output / "reproducibility" / "evaluation.json",
               {"checkpoint": str(checkpoint), "data_dir": str(cfg.data_dir),
                "test_subject": cfg.test_subject, "epoch": state["epoch"], "cache_id": audit["cache_id"],
                "trained_device": trained_device, "evaluation_runtime": device_info(cfg)})
    print(f"S{cfg.test_subject:02d}, seed={cfg.seed}: ACC={metric['ACC_percent']:.2f}%, "
          f"ITR={metric['ITR_bits_min']:.2f} bits/min | {output / 'report.md'}", flush=True)
    return True


def main(cfg=CONFIG):
    if cfg.evaluation_checkpoint is not None:
        checkpoint = Path(cfg.evaluation_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        with FileLock(str(checkpoint.parent / "run.lock"), timeout=0):
            evaluate_checkpoint(checkpoint, cfg.evaluation_data_dir, cfg.device)
        return
    cfg.validate()
    details = cfg.output_dir / "reproducibility"
    details.mkdir(parents=True, exist_ok=True)
    completed, missing = 0, 0
    with FileLock(str(details / "campaign.lock"), timeout=0):
        for single in cfg.runs():
            checkpoint = single.output_dir / "reproducibility" / "checkpoint.pt"
            if not checkpoint.exists():
                missing += 1
                continue
            with FileLock(str(checkpoint.parent / "run.lock"), timeout=0):
                completed += evaluate_checkpoint(checkpoint, cfg.evaluation_data_dir, cfg.device)
        summarize(cfg, evaluation=True)
    print(f"复核完成 {completed} 个模型；{missing} 个尚无 checkpoint。", flush=True)


if __name__ == "__main__":
    main()
