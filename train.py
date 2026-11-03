"""按被试、种子依次训练；每折独立初始化，日志显示在当前终端。"""

import ast
import hashlib
import json
import math
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
from data import prepare_data, split_subjects, subject_partition, validation_data
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


# 共用训练流程：基础模型和 FB 都调用此处，避免重复实现早停与恢复。
class EarlyStopper:
    def __init__(self, patience, min_epochs, min_delta, state=None):
        self.patience, self.min_epochs, self.min_delta = patience, min_epochs, min_delta
        self.best_loss, self.anchor_loss, self.best_epoch, self.bad_epochs = None, None, 0, 0
        if state is not None:
            for key in ("best_loss", "anchor_loss", "best_epoch", "bad_epochs"):
                setattr(self, key, state[key])

    def update(self, loss, epoch):
        if not math.isfinite(loss):
            raise FloatingPointError("验证损失非有限")
        best = self.best_loss is None or loss < self.best_loss
        if best:
            self.best_loss, self.best_epoch = float(loss), epoch
        if self.anchor_loss is None or loss < self.anchor_loss - self.min_delta:
            self.anchor_loss, self.bad_epochs = float(loss), 0
        else:
            self.bad_epochs += 1
        return best, self.should_stop(epoch)

    def should_stop(self, epoch):
        return epoch >= self.min_epochs and self.bad_epochs >= self.patience

    def state_dict(self):
        return {k: getattr(self, k) for k in ("best_loss", "anchor_loss", "best_epoch", "bad_epochs")}


def fit_stages(model, x, y, cfg, details, signature, cache_id, stages,
               should_stop, message, plot, validation=None, model_name="SSVEPformer"):
    """接口不接收测试数据。checkpoint.model 保存最新状态，阶段完成后改为选中状态。"""
    if cfg.early_stopping != (validation is not None):
        raise ValueError("早停必须提供独立验证集；关闭时不使用验证数据")
    checkpoint = details / "checkpoint.pt"
    saved, history, selections = None, [], []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("format_version") != 2:
            raise ValueError("旧 checkpoint 仅供评估；新功能请使用新 experiment")
        if saved["signature"] != signature or saved["cache_id"] != cache_id:
            raise ValueError("checkpoint 与设置或数据不匹配")
        model.load_state_dict(saved["model"])
        restore_rng(saved, model)
        history, selections = saved["history"], saved["selections"]
        if saved["training_complete"]:
            return history, selections

    for stage_index, (stage, band, limit) in enumerate(stages):
        if saved is not None and (stage_index < saved["stage_index"] or
                                 (stage_index == saved["stage_index"] and saved["stage_complete"])):
            continue
        network = model if band is None else model.subnetworks[band]
        stage_x = x if band is None else x[:, band]
        stage_validation = None if validation is None else (
            validation[0] if band is None else validation[0][:, band], validation[1])
        monitor_indices = torch.cat([
            torch.arange(i, min(i + cfg.classes, len(y)))
            for i in range(0, len(y), cfg.classes * cfg.blocks)])
        optimizer = torch.optim.SGD(network.parameters(), lr=cfg.learning_rate,
                                    momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        patience = min(cfg.early_patience, 5) if stage == "joint" else cfg.early_patience
        minimum = min(cfg.early_min_epochs, 5 if stage == "joint" else limit, limit)
        controller = EarlyStopper(patience, minimum, cfg.early_min_delta)
        best_weights, epoch = None, 0
        if saved is not None and stage_index == saved["stage_index"]:
            optimizer.load_state_dict(saved["optimizer"])
            optimizer_to(optimizer, next(model.parameters()).device)
            epoch, best_weights = saved["epoch"], saved["best_model"]
            controller = EarlyStopper(patience, minimum, cfg.early_min_delta, saved["early_state"])

        def save(stage_complete=False, training_complete=False):
            state = {"format_version": 2, "model_name": model_name,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": cfg.to_dict(), "signature": signature, "cache_id": cache_id,
                     "stage_index": stage_index, "stage": stage, "epoch": epoch,
                     "stage_complete": stage_complete, "training_complete": training_complete,
                     "history": history, "selections": selections,
                     "early_state": controller.state_dict(), "best_model": best_weights,
                     **capture_rng(model)}
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoint)

        message(f"{stage}: {epoch}/{limit} epoch | train={len(y)}, "
                f"validation={0 if validation is None else len(validation[1])} | early={cfg.early_stopping}")
        stopped_early = cfg.early_stopping and controller.should_stop(epoch)
        while epoch < limit and not stopped_early:
            if should_stop():
                save()
                raise InterruptedError("STOP：已保存最新训练状态，未使用测试数据")
            epoch += 1
            tick = time.perf_counter()
            loss, accuracy = train_epoch(network, optimizer, stage_x, y, cfg, stage_index * cfg.epochs + epoch)
            monitor = None
            if epoch == 1 or epoch % cfg.monitor_every == 0 or epoch == limit:
                monitor, _ = evaluate(network, stage_x[monitor_indices], y[monitor_indices], cfg)
            val = None
            if cfg.early_stopping:
                val, _ = evaluate(network, *stage_validation, cfg)
                improved, stopped_early = controller.update(val["loss"], epoch)
                if improved:
                    best_weights = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
            row = {"stage": stage, "epoch": epoch, "train_loss": loss, "train_acc": accuracy,
                   "monitor_loss": None if monitor is None else monitor["loss"],
                   "monitor_acc": None if monitor is None else monitor["acc_percent"],
                   "val_loss": None if val is None else val["loss"],
                   "val_acc": None if val is None else val["acc_percent"],
                   "seconds": time.perf_counter() - tick}
            history.append(row)
            write_csv(details / "history.csv", history)
            stop = should_stop()
            if epoch % cfg.checkpoint_every == 0 or epoch == limit or stop or stopped_early:
                save()
            write_json(details / "status.json", {"state": "training", **row,
                                                  "best_epoch": controller.best_epoch})
            suffix = "" if val is None else f" | val loss={val['loss']:.5f}, ACC={val['acc_percent']:.2f}%"
            message(f"{stage} {epoch:03d}/{limit} | loss={loss:.4f} | train ACC={accuracy:.2f}%"
                    f"{suffix} | {row['seconds']:.1f}s")
            if monitor is not None or stopped_early:
                plot(history, details.parent)
            if stop:
                raise InterruptedError("STOP：已保存当前阶段，停止整个实验计划")

        if cfg.early_stopping:
            if best_weights is None:
                raise RuntimeError("缺少最佳验证权重，拒绝测试")
            network.load_state_dict(best_weights)
        selection = {"stage": stage, "epochs_ran": epoch,
                     "selected_epoch": controller.best_epoch if cfg.early_stopping else epoch,
                     "best_validation_loss": controller.best_loss if cfg.early_stopping else None,
                     "reason": "patience" if stopped_early else "max_epochs",
                     "selection": "validation_min_loss" if cfg.early_stopping else "last_epoch"}
        selections.append(selection)
        save(stage_complete=True, training_complete=stage_index == len(stages) - 1)
        write_json(details / "selection.json", {"early_stopping": cfg.early_stopping, "stages": selections})
        message(f"{stage} 完成：训练 {epoch} 轮，采用第 {selection['selected_epoch']} 轮")
        saved = None
    return history, selections


def save_checkpoint(path, model, optimizer, cfg, signature, epoch, history):
    temp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(), "signature": signature, "epoch": epoch,
                "history": history, **capture_rng(model)}, temp)
    temp.replace(path)


def _source_identity(directory, fb=False):
    """按功能核对实现，兼容本次纯文件合并；不因模块搬迁改变训练身份。"""
    def nodes(name):
        return ast.parse((directory / name).read_text(encoding="utf-8-sig")).body

    def digest(node):
        # 旧 fit_stages 的局部导入在合并后省去，调用的仍是同一个 train_epoch。
        if isinstance(node, ast.FunctionDef) and node.name == "fit_stages":
            node.body = [n for n in node.body if not (
                isinstance(n, ast.ImportFrom) and n.module == "train" and
                [(a.name, a.asname) for a in n.names] == [("train_epoch", None)])]
        return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()

    code = nodes("train.py")
    if (directory / "training.py").exists():  # 只读取已有结果里的旧源码快照。
        code += nodes("training.py")
    required = {"seed_everything", "train_epoch", "EarlyStopper", "fit_stages", "run_single"}
    result = {n.name: digest(n) for n in code
              if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in required}
    if set(result) != required:
        raise ValueError("源码快照不包含完整的早停实现，不能接续此断点")
    for name in ("data.py", "model.py", "runtime.py"):
        result[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    result["evaluate"] = digest(next(n for n in nodes("evaluate.py")
                                      if isinstance(n, ast.FunctionDef) and n.name == "evaluate"))
    config_files = [("Config", "settings.py" if (directory / "settings.py").exists() else "config.py")]
    if fb:
        config_files += [("FBConfig", "fb/settings.py" if (directory / "fb/settings.py").exists() else "fb/config.py")]
        for name in ("fb/data.py", "fb/model.py"):
            result[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for node in nodes("fb/train.py"):
            if isinstance(node, ast.FunctionDef) and node.name in {"training_plan", "train_stages", "run_single"}:
                result["fb." + node.name] = digest(node)
    for class_name, filename in config_files:
        cls = next(n for n in nodes(filename) if isinstance(n, ast.ClassDef) and n.name == class_name)
        for node in cls.body:
            if isinstance(node, ast.FunctionDef):
                result[class_name + "." + node.name] = digest(node)
    return result


def run_identity(cfg, sources=None, *, fb=False):
    ignored = {"N", "M", "experiment", "evaluation_checkpoint", "evaluation_data_dir"}
    identity = {"schema": 2,
                "config": {k: v for k, v in cfg.to_dict().items() if k not in ignored},
                "sources": _source_identity(sources or ROOT, fb),
                "runtime": device_info(cfg),
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": np.__version__, "scipy": scipy.__version__}}
    return json.loads(json.dumps(identity))


def manifest_matches(previous, identity, details, *, fb=False):
    old = previous.get("identity", {})
    if old == identity:
        return True
    if old.get("schema") == 2:
        return False
    # 仅允许参数、环境、数据和实现均一致的旧文件布局，不放宽训练设置检查。
    if any(old.get(k) != identity.get(k) for k in ("config", "runtime", "versions", "cache_id")):
        return False
    source = details / "source"
    try:
        if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != value
               for name, value in old.get("sources", {}).items()):
            return False
        return _source_identity(source, fb) == identity["sources"]
    except (OSError, ValueError, StopIteration, SyntaxError):
        return False


def check_manifest(cfg, details):
    identity = run_identity(cfg)
    path = details / "settings.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if "identity" not in previous:
            raise ValueError("旧版实验不自动迁移到新 GPU 实现，请保留原结果并使用新的 experiment")
        if not manifest_matches(previous, identity, details):
            raise ValueError("当前训练设置与已有实验不同，请修改 experiment 使用新的目录")
        return previous["signature"]
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {"config": cfg.to_dict(), "identity": identity, "signature": signature,
                "split": subject_partition(cfg),
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
            message(f"输入 {cfg.input_type}: {cfg.input_channels}×{cfg.features} | 参数量 {parameters:,}")
            should_stop = lambda: stop_path.exists() or (details / "STOP").exists()
            history, selections = fit_stages(
                model, x, y, cfg, details, signature, audit["cache_id"],
                [("train", None, cfg.epochs)], should_stop, message, plot_training,
                validation=validation_data(features, cfg))
            if should_stop():
                raise InterruptedError("STOP：训练已保存，最终测试尚未执行")
            # 完成选模后，测试被试才首次进入模型。
            train_metrics, _ = evaluate(model, x, y, cfg)
            test_metrics, predictions = evaluate(model, test_x, test_y, cfg)
            metric = save_results(cfg, history, test_metrics, train_metrics, predictions,
                                  test_y.numpy(), audit["target_frequencies"], parameters=parameters, selections=selections)
            write_json(status_path, {"state": "complete", "epoch": history[-1]["epoch"], "epochs": cfg.epochs, "selections": selections})
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
