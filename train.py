"""按被试、种子依次训练；每折独立初始化，日志显示在当前终端。"""

import ast
import hashlib
import json
import platform
import random
import shutil
import time
import traceback

import numpy as np
import scipy
import torch
from filelock import FileLock

from config import CONFIG, ROOT
from data import prepare_data, split_subjects
from evaluate import evaluate, plot_training, save_results, summarize, write_csv, write_json
from model import SSVEPformer
from runtime import capture_rng, configure_device, device_info, optimizer_to, restore_rng


def seed_everything(cfg):
    device = configure_device(cfg)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    return device


def train_epoch(model, optimizer, x, y, cfg, epoch):
    model.train()
    device = next(model.parameters()).device
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(cfg.seed + epoch))
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    for indices in order.split(cfg.batch_size):
        optimizer.zero_grad(set_to_none=True)
        # 特征留在 CPU；只将当前批次送到选定 GPU，避免 FB 全量特征占用显存。
        batch_x, batch_y = x[indices].to(device), y[indices].to(device)
        logits = model(batch_x)
        loss = torch.nn.functional.cross_entropy(logits, batch_y)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失非有限")
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().to(torch.float64) * len(indices)
        correct += (logits.detach().argmax(1) == batch_y).sum()
    return loss_sum.item() / len(y), 100 * correct.item() / len(y)


def save_checkpoint(path, model, optimizer, cfg, signature, epoch, history):
    temp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(), "signature": signature, "epoch": epoch,
                "history": history, **capture_rng(model)}, temp)
    temp.replace(path)


def run_identity(cfg, sources=None):
    """排除调度规模和报告代码，避免扩展 N/M 或修改注释使已有模型失效。"""
    ignored = {"N", "M", "experiment", "evaluation_checkpoint", "evaluation_data_dir"}
    settings = {k: v for k, v in cfg.to_dict().items() if k not in ignored}
    directory = sources or ROOT
    functions = ast.parse((directory / "train.py").read_text(encoding="utf-8-sig"))
    algorithm = [ast.dump(node, include_attributes=False) for node in functions.body
                 if isinstance(node, ast.FunctionDef) and node.name in {"seed_everything", "train_epoch"}]
    if len(algorithm) != 2:
        raise ValueError("无法核对原始训练函数")
    identity = {"config": settings, "algorithm": hashlib.sha256("\n".join(algorithm).encode()).hexdigest(),
                "sources": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                            for name in ("data.py", "model.py", "runtime.py")},
                "runtime": device_info(cfg),
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": np.__version__, "scipy": scipy.__version__}}
    return json.loads(json.dumps(identity))


def check_manifest(cfg, details):
    identity = run_identity(cfg)
    path = details / "settings.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if "identity" not in previous:
            raise ValueError("旧版实验不自动迁移到新 GPU 实现，请保留原结果并使用新的 experiment")
        if previous["identity"] != identity:
            raise ValueError("当前训练设置与已有实验不同，请修改 experiment 使用新的目录")
        return previous["signature"]
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {"config": cfg.to_dict(), "identity": identity, "signature": signature,
                "split": f"S{cfg.test_subject} test; all other subjects train",
                "metrics": "ACC; ITR with T=duration+gaze_seconds; no SD for one run",
                "monitor": "First block of each training subject; eval mode; not validation data"}
    write_json(path, manifest)
    source_dir = details / "source"
    source_dir.mkdir(exist_ok=True)
    for name in ("config.py", "data.py", "model.py", "train.py", "evaluate.py", "runtime.py", "README.md", "requirements.txt"):
        shutil.copy2(ROOT / name, source_dir / name)
    return signature


def run_single(cfg, features, audit, stop_path):
    cfg.validate()
    output, details = cfg.output_dir, cfg.output_dir / "reproducibility"
    details.mkdir(parents=True, exist_ok=True)
    with FileLock(str(details / "run.lock"), timeout=0), (details / "training.log").open("a", encoding="utf-8") as log:
        def message(text):
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()

        signature = check_manifest(cfg, details)
        audit_path = details / "data_audit.json"
        if audit_path.exists() and json.loads(audit_path.read_text(encoding="utf-8"))["cache_id"] != audit["cache_id"]:
            raise ValueError("数据源或预处理发生变化，请使用新的 experiment")
        if (output / "metrics.csv").exists():
            message(f"实验已完成：{output / 'report.md'}；需要复核时运行 evaluate.py。")
            return True
        status_path = details / "status.json"
        write_json(status_path, {"state": "preparing", "epoch": 0, "epochs": cfg.epochs})
        try:
            write_json(details / "data_audit.json", audit)
            x, y, test_x, test_y = split_subjects(features, cfg)
            device = seed_everything(cfg)
            model = SSVEPformer(cfg).to(device)
            parameters = sum(p.numel() for p in model.parameters())
            settings = json.loads((details / "settings.json").read_text(encoding="utf-8"))
            settings["model"] = {"input_shape": [cfg.input_channels, cfg.features], "parameters": parameters}
            write_json(details / "settings.json", settings)
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                                        momentum=cfg.momentum, weight_decay=cfg.weight_decay)
            # 所有训练被试各取第一个 block，仅观察训练拟合，绝不包含测试被试。
            monitor_indices = torch.cat([torch.arange(i * cfg.classes * cfg.blocks,
                                                      i * cfg.classes * cfg.blocks + cfg.classes)
                                         for i in range(cfg.subjects - 1)])
            monitor_x, monitor_y = x[monitor_indices], y[monitor_indices]
            checkpoint, history, start = details / "checkpoint.pt", [], 0
            if checkpoint.exists():
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if state["signature"] != signature:
                    raise ValueError("checkpoint 与实验配置不匹配")
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                optimizer_to(optimizer, device)
                restore_rng(state, model)
                history, start = state["history"], state["epoch"]
            message(f"SSVEPformer | {device} | seed={cfg.seed} | test=S{cfg.test_subject} | "
                    f"train={len(y)}, test={len(test_y)} | init_std={cfg.init_std} | epoch {start}/{cfg.epochs}")
            message(f"输入 {cfg.input_type}: {cfg.input_channels}×{cfg.features} | 参数量 {parameters:,}")
            for epoch in range(start + 1, cfg.epochs + 1):
                if stop_path.exists() or (details / "STOP").exists():
                    if history:
                        save_checkpoint(checkpoint, model, optimizer, cfg, signature, history[-1]["epoch"], history)
                    raise InterruptedError("STOP 文件请求停止")
                tick = time.perf_counter()
                loss, accuracy = train_epoch(model, optimizer, x, y, cfg, epoch)
                monitor = None
                if epoch == 1 or epoch % cfg.monitor_every == 0 or epoch == cfg.epochs:
                    monitor, _ = evaluate(model, monitor_x, monitor_y, cfg)
                row = {"epoch": epoch, "train_loss": loss, "train_acc": accuracy,
                       "monitor_loss": monitor["loss"] if monitor else None,
                       "monitor_acc": monitor["acc_percent"] if monitor else None,
                       "seconds": time.perf_counter() - tick}
                history.append(row)
                write_csv(details / "history.csv", history)
                stop = stop_path.exists() or (details / "STOP").exists()
                if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs or stop:
                    save_checkpoint(checkpoint, model, optimizer, cfg, signature, epoch, history)
                suffix = f" | train monitor (dropout OFF)={monitor['acc_percent']:.2f}%" if monitor else ""
                message(f"Epoch {epoch:03d}/{cfg.epochs} | loss={loss:.4f} | "
                        f"train ACC (dropout ON)={accuracy:.2f}%{suffix} | {row['seconds']:.1f}s")
                write_json(status_path, {"state": "training", "epoch": epoch, "epochs": cfg.epochs,
                                         "train_loss": loss, "train_acc_dropout_on": accuracy,
                                         "monitor_acc_dropout_off": row["monitor_acc"]})
                if monitor:
                    plot_training(history, output)

                if stop:
                    raise InterruptedError("STOP 文件请求停止")
            # 当前留出被试首次进入模型发生在固定训练轮数结束后。
            train_metrics, _ = evaluate(model, x, y, cfg)
            test_metrics, predictions = evaluate(model, test_x, test_y, cfg)
            metric = save_results(cfg, history, test_metrics, train_metrics, predictions,
                                  test_y.numpy(), audit["target_frequencies"], parameters=parameters)
            write_json(status_path, {"state": "complete", "epoch": cfg.epochs, "epochs": cfg.epochs})
            message(f"TEST S{cfg.test_subject}: ACC={metric['ACC_percent']:.2f}% | "
                    f"ITR={metric['ITR_bits_min']:.2f} bits/min | {metric['correct']}/{metric['trials']}")
            message(f"报告：{output / 'report.md'}")
            return True
        except (KeyboardInterrupt, InterruptedError):
            write_json(status_path, {"state": "stopped", "resume": "last saved checkpoint"})
            message("训练已停止。删除 STOP 后重新运行 train.py 可恢复最近保存的 checkpoint。")
            return False
        except Exception:
            write_json(status_path, {"state": "failed", "error": traceback.format_exc()})
            raise


def run(cfg=CONFIG):
    cfg.validate()
    configure_device(cfg)
    print(f"计算设备：{device_info(cfg)}", flush=True)
    details = cfg.output_dir / "reproducibility"
    details.mkdir(parents=True, exist_ok=True)
    stop_path = details / "STOP"
    with FileLock(str(details / "campaign.lock"), timeout=0):
        write_json(details / "plan.json", cfg.to_dict())
        if stop_path.exists():
            print(f"已停止：恢复前请删除 {stop_path}", flush=True)
            return
        print(f"计划：S1–S{cfg.N}，每折 {cfg.M} 个种子（{cfg.seed}–{cfg.seed + cfg.M - 1}），"
              f"共 {cfg.N * cfg.M} 次实验。", flush=True)
        try:
            features, audit = prepare_data(cfg)
            for index, single in enumerate(cfg.runs(), 1):
                if stop_path.exists():
                    print("STOP：整个实验计划已停止。", flush=True)
                    break
                print(f"\n[{index}/{cfg.N * cfg.M}] S{single.test_subject:02d}, seed={single.seed}", flush=True)
                if not run_single(single, features, audit, stop_path):
                    break
                summarize(cfg)
        except KeyboardInterrupt:
            print("实验计划已停止；重新运行可从最近 checkpoint 恢复。", flush=True)
        finally:
            summarize(cfg)


if __name__ == "__main__":
    run()
