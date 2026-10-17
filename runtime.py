"""两个模型共用的 CPU/CUDA 设置与断点状态恢复。"""

import os

import torch


def configure_device(cfg):
    """显式选择设备，不在 GPU 配置错误时悄悄退回 CPU。"""
    device = torch.device(cfg.device)
    if device.type == "cuda":
        # 必须在首次 CUDA 矩阵计算前设置，供确定性 cuBLAS 运算使用。
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError("请将 CUBLAS_WORKSPACE_CONFIG 设置为 :4096:8 或 :16:8")
        if not torch.cuda.is_available():
            raise RuntimeError("已选择 CUDA，但 PyTorch 未检测到可用 GPU。请检查 GPU 版 torch/驱动，或设置 device='cpu'。")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"不存在 cuda:{index}，当前可见 GPU 数量为 {torch.cuda.device_count()}")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 使用 FP32，不默认启用 AMP 或 TF32，保持实验数值口径清晰。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def device_info(cfg):
    device = torch.device(cfg.device)
    return {"device": str(device), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
            "precision": "FP32; TF32/AMP disabled", "deterministic": True}


def capture_rng(model):
    device = next(model.parameters()).device
    return {"torch_rng": torch.get_rng_state(), "execution_device": device.type,
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(saved, model):
    device = next(model.parameters()).device
    if saved.get("execution_device", "cpu") != device.type:
        raise ValueError("不自动将 CPU/GPU 混合续训；请用新 experiment 重训，并保留原实验。")
    if device.type == "cuda" and saved.get("cuda_rng") is None:
        raise ValueError("CUDA checkpoint 缺少随机状态，无法按原轨迹恢复")
    torch.set_rng_state(saved["torch_rng"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)


def optimizer_to(optimizer, device):
    """SGD 断点通过 CPU 读取后，将动量缓冲区放回模型设备。"""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
