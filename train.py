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


def seed_everything(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(True)


def train_epoch(model, optimizer, x, y, cfg, epoch):
    model.train()
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(cfg.seed + epoch))
    loss_sum, correct = 0.0, 0
    for indices in order.split(cfg.batch_size):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[indices])
        loss = torch.nn.functional.cross_entropy(logits, y[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失非有限")
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * len(indices)
        correct += int((logits.detach().argmax(1) == y[indices]).sum())
    return loss_sum / len(y), 100 * correct / len(y)


def save_checkpoint(path, model, optimizer, cfg, signature, epoch, history):
    temp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(), "signature": signature, "epoch": epoch,
                "history": history, "torch_rng": torch.get_rng_state()}, temp)
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
                            for name in ("data.py", "model.py")},
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": np.__version__, "scipy": scipy.__version__}}
    return json.loads(json.dumps(identity))


def check_manifest(cfg, details):
    identity = run_identity(cfg)
    path = details / "settings.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if "identity" not in previous:
            # 仅兼容已完成的旧单折：核对原始配置、数据/模型代码及数值训练函数。
            old = run_identity(cfg, details / "source")
            old["config"] = {k: previous["config"][k] for k in identity["config"]}
            old["versions"] = previous["versions"]
            if old != identity or not (details.parent / "metrics.csv").exists():
                raise ValueError("旧实验与当前训练设置不兼容，请使用新的 experiment")
            previous["identity"] = identity
            previous["compatibility_note"] = "已核对原始训练设置；保留原始 signature、checkpoint 与代码快照。"
            write_json(path, previous)
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
    for name in ("config.py", "data.py", "model.py", "train.py", "evaluate.py", "README.md", "requirements.txt"):
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
            seed_everything(cfg)
            model = SSVEPformer(cfg)
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
                torch.set_rng_state(state["torch_rng"])
                history, start = state["history"], state["epoch"]
            message(f"SSVEPformer | CPU | seed={cfg.seed} | test=S{cfg.test_subject} | "
                    f"train={len(y)}, test={len(test_y)} | init_std={cfg.init_std} | epoch {start}/{cfg.epochs}")
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
                                  test_y.numpy(), audit["target_frequencies"])
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
