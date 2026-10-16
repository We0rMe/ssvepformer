"""只修改本文件即可设置 FB 实验；共用参数继承根目录 Config 的默认值。"""

from dataclasses import dataclass
from pathlib import Path

from config import Config


@dataclass(frozen=True)
class FBConfig(Config):
    experiment: str = "fb_ssvepformer_loso_cuda"  # 新 GPU 实验目录，名称须以 fb_ 开头。
    device: str = "cuda:0"  # 可改为 cuda:1 使用第二张卡；本机无 GPU 检查时使用 cpu。
    N: int = 1  # 默认先评估 S1；改为 35 即完整 LOSO，其余被试始终全部参与训练。
    M: int = 1  # 每折种子数，依次使用 seed…seed+M-1。
    seed: int = 2026  # 起始随机种子；每折从头初始化全部子网与融合层。
    duration: float = 1.0  # 分类窗口长度（秒）；其余裁窗/FFT 设置沿用根配置。
    cpu_threads: int = 8  # 正式 FB 训练的 CPU 线程数，不修改其他进程的线程设置。
    evaluation_checkpoint: Path | None = None  # None：批量复核；也可填写单个 FB checkpoint.pt 的完整路径。
    evaluation_data_dir: Path | None = None  # None：读取 checkpoint 的数据路径；指定迁移目录时另存复核结果。
    epochs: int = 100  # 每个子网分别预训练的轮数；3 子网共执行 3×100 轮。
    joint_epochs: int = 20  # 预训练结束后，全部子网与融合层一起训练的轮数。
    subbands: int = 3  # 子带数；每个子带对应一个不共享权重的 SSVEPformer。
    band_step: float = 9.0  # Dataset 2 第 m 个子带的通带下限 = m×9 Hz，m 从 1 开始。
    band_high: float = 80.0  # 所有子带共同的通带上限（Hz），不同于 FFT 特征上限 64 Hz。
    lower_transition: float = 2.0  # 下侧过渡带宽度；阻带下边界 = 通带下限减去此值。
    upper_stop: float = 90.0  # 上侧阻带边界（Hz）；80–90 Hz 为上侧过渡带。
    passband_ripple: float = 0.5  # 单向 Chebyshev-I 通带最大波纹（dB），定阶与设计使用同一值。
    stopband_attenuation: float = 40.0  # 单向滤波器要求的阻带最小衰减（dB），据此自动定阶。
    bandpass: bool = False  # FB 不先执行基础 8–64 Hz Butterworth；继承的 filter_order 不参与 FB。

    def validate(self):
        super().validate()
        assert self.experiment.startswith("fb_"), "FB 实验目录须以 fb_ 开头，避免混入基础模型结果"
        assert isinstance(self.subbands, int) and self.subbands >= 1
        assert isinstance(self.joint_epochs, int) and self.joint_epochs >= 1
        assert not self.bandpass, "FB 直接对原始试次做子带滤波，无需基础带通"
        assert 0 < self.lower_transition < self.band_step
        assert self.subbands * self.band_step < self.band_high < self.upper_stop < self.sample_rate / 2
        assert 0 < self.passband_ripple < self.stopband_attenuation


CONFIG = FBConfig()  # 其他参数如学习率、Dropout、N/M 规则、评估路径均继承；可在上面同名覆盖。
