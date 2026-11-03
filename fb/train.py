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
from data import subject_partition, validation_data
from evaluate import evaluate, write_csv, write_json
from train import seed_everything, train_epoch, fit_stages, run_identity, manifest_matches
from fb.config import CONFIG
from fb.data import prepare_data, split_subjects
from fb.evaluate import plot_training, save_results, summarize
from fb.model import FBSSVEPformer
from runtime import capture_rng, configure_device, device_info, optimizer_to, restore_rng


def training_plan(cfg):
    """(阶段名称, 子网索引或 None, 轮数)；None 表示训练整个模型。"""
    return [(f"subband_{band + 1}", band, cfg.epochs) for band in range(cfg.subbands)] + [
        ("joint", None, cfg.joint_epochs)]


def check_manifest(cfg, audit, details):
    sources = ("config.py", "data.py", "model.py", "train.py", "evaluate.py", "runtime.py",
               "fb/config.py", "fb/data.py", "fb/model.py", "fb/train.py", "fb/evaluate.py")
    identity = run_identity(cfg, fb=True)
    identity["cache_id"] = audit["cache_id"]
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    manifest = details / "settings.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if not manifest_matches(previous, identity, details, fb=True):
            raise ValueError("FB 训练设置、数据或实现已变化，请使用新的 experiment")
        return previous["signature"]
    write_json(manifest, {"config": cfg.to_dict(), "signature": signature, "identity": identity,
                          "plan": training_plan(cfg), "split": subject_partition(cfg),
                          "monitor": "First block of each training subject; not an independent validation set"})
    write_json(details / "data_audit.json", audit)
    for name in (*sources, "fb/README.md", "requirements.txt"):
        target = details / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return signature


def save_checkpoint(path, model, optimizer, cfg, signature, cache_id, stage_index, epoch, history):
    saved = {"model_name": "FB-SSVEPformer", "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "config": cfg.to_dict(), "signature": signature,
             "cache_id": cache_id, "stage_index": stage_index, "epoch": epoch, "history": history,
             "training_complete": stage_index == cfg.subbands and epoch == cfg.joint_epochs,
             **capture_rng(model)}
    temporary = path.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)


def train_stages(model, x, y, monitor_indices, cfg, details, signature, cache_id, should_stop, message,
                 validation=None):
    """兼容原调用接口；训练监测按实际训练被试重建，验证集单独传入。"""
    history, _ = fit_stages(model, x, y, cfg, details, signature, cache_id, training_plan(cfg),
                            should_stop, message, plot_training, validation, "FB-SSVEPformer")
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
            device = seed_everything(cfg)
            model = FBSSVEPformer(cfg).to(device)
            parameters = sum(p.numel() for p in model.parameters())
            settings = json.loads((details / "settings.json").read_text(encoding="utf-8"))
            settings["model"] = {"input_shape": [cfg.subbands, cfg.input_channels, cfg.features],
                                 "parameters": parameters}
            write_json(details / "settings.json", settings)
            message(f"FB 输入 {cfg.input_type}: {cfg.subbands}×{cfg.input_channels}×{cfg.features} | 参数量 {parameters:,}")
            should_stop = lambda: stop_path.exists() or (details / "STOP").exists()
            history, selections = fit_stages(
                model, x, y, cfg, details, signature, audit["cache_id"], training_plan(cfg),
                should_stop, message, plot_training, validation_data(features, cfg), "FB-SSVEPformer")
            if should_stop():
                raise InterruptedError("STOP：联合训练已保存，最终测试尚未执行")
            # 所有子网预训练和联合训练均结束后，才使用留出被试进行最终测试。
            train_metrics, _ = evaluate(model, x, y, cfg)
            test_metrics, predictions = evaluate(model, test_x, test_y, cfg)
            metric = save_results(cfg, model, history, test_metrics, train_metrics, predictions,
                                  test_y.numpy(), audit["target_frequencies"], selections=selections)
            write_json(details / "status.json", {"state": "complete", "stage": "joint", "epoch": history[-1]["epoch"], "selections": selections})
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
    configure_device(cfg)
    print(f"FB 计算设备：{device_info(cfg)}", flush=True)
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
