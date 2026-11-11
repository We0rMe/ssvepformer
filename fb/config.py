"""FB 实验配置：顶部修改常用选项，下方固定滤波器组参数。"""
from dataclasses import dataclass
from pathlib import Path
from config import Config, CONFIG as BASE

@dataclass(frozen=True)
class FBConfig(Config):
    data_dir: Path = BASE.data_dir
    experiment: str = "fb_ssvepformer_complex_channels"
    input_type: str = "complex_channels"  # complex / amplitude / time / complex_channels
    duration: float = 0.8  # 秒，例如 0.5、1.0、2.0。
    device: str = "cuda:1"
    N: int = 35
    M: int = 3
    seed: int = 2026
    early_stopping: bool = False  # 所有子网和联合阶段共用独立验证被试；各阶段独立早停。
    epochs: int = 100  # 每个子网的轮数上限。
    joint_epochs: int = 20  # 联合阶段上限。
    early_patience: int = 15  # 联合阶段内固定采用 min(此值, 5)。
    batch_size: int = 128
    learning_rate: float = 0.001
    evaluation_checkpoint: Path | None = None
    evaluation_data_dir: Path | None = None

    # 固定实现参数
    cpu_threads: int = 8  # 正式 FB 训练的 CPU 线程数，不修改其他进程的线程设置。
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


CONFIG = FBConfig()
