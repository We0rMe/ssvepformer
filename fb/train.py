"""FB 入口：子网逐个预训练 → 全网联合训练 → 留出被试测试。python -m fb.train。"""

import hashlib
import json
import platform
import shutil
import time
import traceback

import numpy as np
import scipy
import torch
from filelock import FileLock

from config import ROOT
from evaluate import evaluate, write_csv, write_json
from train import seed_everything, train_epoch
from fb.config import CONFIG
from fb.data import prepare_data, split_subjects
from fb.evaluate import plot_training, save_results, summarize
from fb.model import FBSSVEPformer


def training_plan(cfg):
    """(阶段名称, 子网索引或 None, 轮数)；None 表示训练整个模型。"""
    return [(f"subband_{band + 1}", band, cfg.epochs) for band in range(cfg.subbands)] + [
        ("joint", None, cfg.joint_epochs)]


def check_manifest(cfg, audit, details):
    ignored = {"N", "M", "experiment", "evaluation_checkpoint", "evaluation_data_dir"}
    sources = ("data.py", "model.py", "train.py", "evaluate.py",
               "fb/data.py", "fb/model.py", "fb/train.py", "fb/evaluate.py")
    identity = {"config": {k: v for k, v in cfg.to_dict().items() if k not in ignored},
                "cache_id": audit["cache_id"],
                "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": np.__version__, "scipy": scipy.__version__}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    manifest = details / "settings.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("FB 训练设置、数据或实现已变化，请使用新的 experiment")
        return signature
    write_json(manifest, {"config": cfg.to_dict(), "signature": signature, "identity": identity,
                          "plan": training_plan(cfg), "split": f"S{cfg.test_subject} test; all others train",
                          "monitor": "First block of each training subject; not an independent validation set"})
    write_json(details / "data_audit.json", audit)
    for name in (*sources, "config.py", "fb/config.py", "fb/README.md", "requirements.txt"):
        target = details / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return signature


def save_checkpoint(path, model, optimizer, cfg, signature, cache_id, stage_index, epoch, history):
    saved = {"model_name": "FB-SSVEPformer", "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "config": cfg.to_dict(), "signature": signature,
             "cache_id": cache_id, "stage_index": stage_index, "epoch": epoch, "history": history,
             "training_complete": stage_index == cfg.subbands and epoch == cfg.joint_epochs,
             "torch_rng": torch.get_rng_state()}
    temporary = path.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)


def train_stages(model, x, y, monitor_indices, cfg, details, signature, cache_id, should_stop, message):
    """只接收训练数据；测试数据没有进入这段两阶段训练代码的接口。"""
    checkpoint = details / "checkpoint.pt"
    history, saved = [], None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["signature"] != signature or saved["cache_id"] != cache_id:
            raise ValueError("FB checkpoint 与当前设置或数据不匹配")
        model.load_state_dict(saved["model"])
        torch.set_rng_state(saved["torch_rng"])
        history = saved["history"]
    for stage_index, (stage, band, epochs) in enumerate(training_plan(cfg)):
        if saved is not None and stage_index < saved["stage_index"]:
            continue
        network = model if band is None else model.subnetworks[band]
        stage_x = x if band is None else x[:, band]
        monitor_x, monitor_y = stage_x[monitor_indices], y[monitor_indices]
        # 每个子网独立优化；联合阶段新建 SGD，同时更新所有子网与融合层。
        optimizer = torch.optim.SGD(network.parameters(), lr=cfg.learning_rate,
                                    momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        start = 0
        if saved is not None and stage_index == saved["stage_index"]:
            optimizer.load_state_dict(saved["optimizer"])
            start = saved["epoch"]
        message(f"{stage}: {start}/{epochs} epoch | {'全网联合训练' if band is None else '独立子网预训练'}")
        for epoch in range(start + 1, epochs + 1):
            if should_stop():
                save_checkpoint(checkpoint, model, optimizer, cfg, signature, cache_id,
                                stage_index, epoch - 1, history)
                raise InterruptedError("STOP：已保存当前阶段，停止整个 FB 计划")
            tick = time.perf_counter()
            loss, accuracy = train_epoch(network, optimizer, stage_x, y, cfg, stage_index * cfg.epochs + epoch)
            monitor = None
            if epoch == 1 or epoch % cfg.monitor_every == 0 or epoch == epochs:
                monitor, _ = evaluate(network, monitor_x, monitor_y, cfg)
            history.append({"stage": stage, "epoch": epoch, "train_loss": loss, "train_acc": accuracy,
                            "monitor_loss": monitor["loss"] if monitor else None,
                            "monitor_acc": monitor["acc_percent"] if monitor else None,
                            "seconds": time.perf_counter() - tick})
            write_csv(details / "history.csv", history)
            stop = should_stop()
            if epoch % cfg.checkpoint_every == 0 or epoch == epochs or stop:
                save_checkpoint(checkpoint, model, optimizer, cfg, signature, cache_id,
                                stage_index, epoch, history)
            write_json(details / "status.json", {"state": "training", "stage": stage, "epoch": epoch,
                                                  "epochs": epochs, "train_loss": loss,
                                                  "train_acc_dropout_on": accuracy,
                                                  "monitor_acc_dropout_off": history[-1]["monitor_acc"]})
            suffix = f" | train monitor OFF={monitor['acc_percent']:.2f}%" if monitor else ""
            message(f"{stage} {epoch:03d}/{epochs} | loss={loss:.4f} | train ACC ON={accuracy:.2f}%"
                    f"{suffix} | {history[-1]['seconds']:.1f}s")
            if monitor:
                plot_training(history, details.parent)
            if stop:
                raise InterruptedError("STOP：已保存当前阶段，停止整个 FB 计划")
    return history


def run_single(cfg, features, audit, stop_path):
    cfg.validate()
    details = cfg.output_dir / "reproducibility"
    details.mkdir(parents=True, exist_ok=True)
    with FileLock(str(details / "run.lock"), timeout=0), (details / "training.log").open("a", encoding="utf-8") as log:
        def message(text):
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()

        signature = check_manifest(cfg, audit, details)
        if (cfg.output_dir / "metrics.csv").exists():
            message(f"FB 已完成，跳过：{cfg.output_dir.name}")
            return True
        try:
            x, y, test_x, test_y = split_subjects(features, cfg)
            seed_everything(cfg)
            model = FBSSVEPformer(cfg)
            monitor_indices = torch.cat([torch.arange(i * cfg.classes * cfg.blocks,
                                                      i * cfg.classes * cfg.blocks + cfg.classes)
                                         for i in range(cfg.subjects - 1)])
            should_stop = lambda: stop_path.exists() or (details / "STOP").exists()
            history = train_stages(model, x, y, monitor_indices, cfg, details, signature,
                                   audit["cache_id"], should_stop, message)
            if should_stop():
                raise InterruptedError("STOP：联合训练已保存，最终测试尚未执行")
            # 所有子网预训练和联合训练均结束后，才使用留出被试进行最终测试。
            train_metrics, _ = evaluate(model, x, y, cfg)
            test_metrics, predictions = evaluate(model, test_x, test_y, cfg)
            metric = save_results(cfg, model, history, test_metrics, train_metrics, predictions,
                                  test_y.numpy(), audit["target_frequencies"])
            write_json(details / "status.json", {"state": "complete", "stage": "joint", "epoch": cfg.joint_epochs})
            message(f"FB TEST S{cfg.test_subject}: ACC={metric['ACC_percent']:.2f}% | "
                    f"ITR={metric['ITR_bits_min']:.2f} bits/min | {metric['correct']}/{metric['trials']}")
            return True
        except (KeyboardInterrupt, InterruptedError):
            write_json(details / "status.json", {"state": "stopped", "resume": "last saved stage and epoch"})
            message("FB 已停止，不继续其他折/种子；删除对应 STOP 后重新运行可恢复最近 checkpoint。")
            return False
        except Exception:
            write_json(details / "status.json", {"state": "failed", "error": traceback.format_exc()})
            raise


def run(cfg=CONFIG):
    cfg.validate()
    details = cfg.output_dir / "reproducibility"
    details.mkdir(parents=True, exist_ok=True)
    stop_path = details / "STOP"
    with FileLock(str(details / "campaign.lock"), timeout=0):
        write_json(details / "plan.json", cfg.to_dict())
        if stop_path.exists():
            print(f"FB 已停止；恢复前删除 {stop_path}", flush=True)
            return
        print(f"FB：{cfg.N} 折 × {cfg.M} 种子；每次 {cfg.subbands}×{cfg.epochs} 轮预训练 + "
              f"{cfg.joint_epochs} 轮联合训练。", flush=True)
        try:
            features, audit = prepare_data(cfg)
            for index, single in enumerate(cfg.runs(), 1):
                if stop_path.exists():
                    break
                print(f"\n[{index}/{cfg.N * cfg.M}] FB S{single.test_subject:02d}, seed={single.seed}", flush=True)
                if not run_single(single, features, audit, stop_path):
                    break
                summarize(cfg)
        except KeyboardInterrupt:
            print("FB 计划已停止，重新运行可恢复最近 checkpoint。", flush=True)
        finally:
            summarize(cfg)


if __name__ == "__main__":
    run()
