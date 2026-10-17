"""FB 模型的最终测试、阶段曲线与汇总。运行方式：python -m fb.evaluate。"""

import json
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from evaluate import evaluate, itr, plt, summarize as summarize_base, write_csv, write_json
from fb.config import CONFIG, FBConfig
from fb.data import prepare_data, split_subjects
from fb.model import FBSSVEPformer
from runtime import configure_device, device_info


def plot_training(history, output):
    stages = list(dict.fromkeys(row["stage"] for row in history))
    fig, axes = plt.subplots(len(stages), 2, figsize=(11, 3 * len(stages)), squeeze=False,
                             constrained_layout=True)
    for stage, pair in zip(stages, axes):
        rows = [row for row in history if row["stage"] == stage]
        monitored = [row for row in rows if row["monitor_loss"] is not None]
        for ax, key, ylabel in zip(pair, ("loss", "acc"), ("Cross-entropy", "ACC (%)")):
            ax.plot([r["epoch"] for r in rows], [r[f"train_{key}"] for r in rows],
                    label="Training batches (dropout ON)")
            ax.plot([r["epoch"] for r in monitored], [r[f"monitor_{key}"] for r in monitored],
                    ".-", label="Fixed training subset (dropout OFF)")
            ax.set(title=stage, xlabel="Epoch within stage", ylabel=ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        pair[1].set_ylim(0, 100)
    fig.savefig(output / "training.png", dpi=130)
    plt.close(fig)


def save_results(cfg, model, history, test, train, predictions, labels, frequencies, output=None):
    output = output or cfg.output_dir
    details = output / "reproducibility"
    details.mkdir(parents=True, exist_ok=True)
    weights = model.fusion.weight.detach().flatten().tolist()
    metric = {"model": "FB-SSVEPformer", "input_type": cfg.input_type,
              "test_subject": f"S{cfg.test_subject}", "seed": cfg.seed,
              "duration_s": cfg.duration, "ACC_percent": round(test["acc_percent"], 4),
              "ITR_bits_min": round(itr(test["acc_percent"], cfg), 4),
              "correct": test["correct"], "trials": test["total"]}
    write_csv(details / "predictions.csv", [
        {"block": i // cfg.classes + 1, "target": int(label) + 1,
         "frequency_hz": frequencies[int(label)], "prediction": int(prediction) + 1,
         "correct": int(label == prediction)} for i, (label, prediction) in enumerate(zip(labels, predictions))])
    write_json(details / "fusion_weights.json", {"weights": weights, "note": "Shared across classes; unconstrained logits weights"})
    plot_training(history, output)
    confusion = np.zeros((cfg.classes, cfg.classes), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(confusion, vmin=0, vmax=cfg.blocks, cmap="Blues")
    ticks = np.arange(0, cfg.classes, 5)
    ax.set(xticks=ticks, xticklabels=ticks + 1, yticks=ticks, yticklabels=ticks + 1,
           xlabel="Predicted target (MAT order)", ylabel="True target (MAT order)",
           title=f"FB-SSVEPformer S{cfg.test_subject}: ACC {test['acc_percent']:.2f}%")
    fig.colorbar(im, ax=ax, label="Trials")
    fig.savefig(output / "confusion.png", dpi=150)
    plt.close(fig)
    report = f"""# FB-SSVEPformer · S{cfg.test_subject} 测试结果

| 指标 | 结果 |
|---|---:|
| 测试 ACC | {test['acc_percent']:.2f}% |
| 测试 ITR | {itr(test['acc_percent'], cfg):.2f} bits/min |
| 正确 / 总试次 | {test['correct']} / {test['total']} |
| 最终训练集 ACC（Dropout OFF） | {train['acc_percent']:.2f}% |

S{cfg.test_subject} 留出测试，其余 {cfg.subjects - 1} 名被试训练；seed={cfg.seed}，窗口 {cfg.duration:g} 秒，设备 {cfg.device}。
输入：{cfg.input_type}，每试次 {cfg.subbands}×{cfg.input_channels}×{cfg.features}；模型参数量：{sum(p.numel() for p in model.parameters())}。
{cfg.subbands} 个独立子网各预训练 {cfg.epochs} 轮，再联合训练 {cfg.joint_epochs} 轮。
训练及训练集监测共 {sum(r['seconds'] for r in history) / 60:.1f} 分钟；使用最后一轮联合模型。
预训练阶段不评估测试被试；测试集不参与梯度计算、早停或选模。

training.png 按子网预训练和联合训练分别绘图。橙线来自固定训练子集，不是验证集。
每次实验仅报告单次成绩，跨被试/种子的均值与标准差见上一级总报告。
滤波器系数、配置、各阶段逐轮记录、可学习融合权重、预测和 checkpoint 统一保存于 reproducibility/。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    # 最后写入，作为本次训练、测试与报告全部完成的标志。
    write_csv(output / "metrics.csv", [metric])
    return metric


def summarize(cfg, *, evaluation=False):
    rows = summarize_base(cfg, evaluation=evaluation)
    report = cfg.output_dir / "report.md"
    report.write_text(report.read_text(encoding="utf-8").replace(
        "# SSVEPformer · LOSO 汇总", "# FB-SSVEPformer · LOSO 汇总", 1), encoding="utf-8")
    return rows


def evaluate_checkpoint(checkpoint, data_dir=None, device=None):
    checkpoint = Path(checkpoint).resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("model_name") != "FB-SSVEPformer":
        raise ValueError("请提供 FB-SSVEPformer 的 checkpoint，不能使用基础模型 checkpoint")
    values = saved["config"].copy()
    values.setdefault("input_type", "complex")
    trained_device = values.get("device", "cpu")
    values["device"] = device or trained_device
    for key in ("data_dir", "evaluation_checkpoint", "evaluation_data_dir"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    if data_dir is not None:
        values["data_dir"] = Path(data_dir)
    cfg = FBConfig(**values)
    cfg.validate()
    if not saved["training_complete"] or saved["stage_index"] != cfg.subbands or saved["epoch"] != cfg.joint_epochs:
        print(f"跳过未完成联合训练的模型：{checkpoint}", flush=True)
        return False
    selected_device = configure_device(cfg)
    features, audit = prepare_data(cfg)
    if data_dir is None and audit["cache_id"] != saved["cache_id"]:
        raise ValueError("原始数据/处理代码发生变化；独立复核请显式设置 evaluation_data_dir")
    x, y, test_x, test_y = split_subjects(features, cfg)
    model = FBSSVEPformer(cfg).to(selected_device)
    model.load_state_dict(saved["model"])
    train, _ = evaluate(model, x, y, cfg)
    test, predictions = evaluate(model, test_x, test_y, cfg)
    same_context = data_dir is None and cfg.device == trained_device and "input_type" in saved["config"]
    output = checkpoint.parent.parent if same_context else checkpoint.parent / "reevaluation"
    metric = save_results(cfg, model, saved["history"], test, train, predictions,
                          test_y.numpy(), audit["target_frequencies"], output)
    write_json(output / "reproducibility" / "evaluation.json",
               {"checkpoint": str(checkpoint), "data_dir": str(cfg.data_dir), "cache_id": audit["cache_id"],
                "trained_device": trained_device, "evaluation_runtime": device_info(cfg)})
    print(f"FB S{cfg.test_subject}, seed={cfg.seed}: ACC={metric['ACC_percent']:.2f}%, "
          f"ITR={metric['ITR_bits_min']:.2f} bits/min", flush=True)
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
    with FileLock(str(details / "campaign.lock"), timeout=0):
        count = 0
        for single in cfg.runs():
            checkpoint = single.output_dir / "reproducibility" / "checkpoint.pt"
            if checkpoint.exists():
                with FileLock(str(checkpoint.parent / "run.lock"), timeout=0):
                    count += evaluate_checkpoint(checkpoint, cfg.evaluation_data_dir, cfg.device)
        summarize(cfg, evaluation=True)
    print(f"FB 已复核 {count} / {cfg.N * cfg.M} 个模型，其余尚未完成。", flush=True)


if __name__ == "__main__":
    main()
