"""修改此处配置，然后直接运行 train.py；不使用命令行参数。"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # 项目根目录，自动定位，不需修改。


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(r"G:\SSVEP\data\benchmark")  # 含 S1.mat…S35.mat 和通道/频率文件的数据目录。
    experiment: str = "ssvepformer_loso_cuda"  # 新 GPU 实验目录；旧 CPU 结果保留在 ssvepformer_loso。
    device: str = "cuda:0"  # cpu：CPU；cuda/cuda:0：第一张卡；cuda:1：第二张卡。无可用 GPU 时明确报错。
    N: int = 35  # 留出折数：依次测试 S1…SN；其余所有被试训练，不是只使用前 N 名被试。
    M: int = 1  # 每折独立训练的种子数，使用 seed…seed+M-1，不从其他种子模型继续训练。
    seed: int = 2026  # 起始随机种子；M=1 时所有折均使用此种子。
    evaluation_checkpoint: Path | None = None  # None：复核当前 N×M 计划；或填写某个 checkpoint.pt 的完整路径。
    evaluation_data_dir: Path | None = None  # None：使用模型保存的数据路径；可指定同一 Benchmark 数据的迁移目录。

    subjects: int = 35  # 数据集固定属性：被试总数；控制折数请改 N，不要改此项。
    classes: int = 40  # 数据集固定属性：目标类别数，与 MAT 类别顺序一致。
    blocks: int = 6  # 数据集固定属性：每被试的 block 数，每 block 各类别一个试次。
    channels: tuple = ("O1", "Oz", "O2", "PO3", "POz", "PO4", "Pz", "PO5", "PO6")  # 按名称及此顺序取通道。
    sample_rate: int = 250  # 已降采样的数据采样率（Hz）；修改数值不会执行重采样。
    duration: float = 1.0  # 每试次分类窗口长度（秒），须对应整数采样点且不超过 FFT 长度。
    start_seconds: float = 0.64  # 从试次起点裁窗：0.5 秒提示 + 0.14 秒视觉延迟。
    frequency_resolution: float = 0.2  # FFT 频率分辨率（Hz），由采样率/此值计算补零长度。
    frequency_range: tuple = (8.0, 64.0)  # 复频谱保留范围（Hz，含端点），也是带通滤波的通带。
    bandpass: bool = True  # 是否在整段试次上独立执行零相位带通，再裁窗。
    filter_order: int = 4  # Butterworth 设计阶数，供 scipy.signal.butter 使用。

    init_std: float = 0.01  # 卷积/线性权重初始化的标准差；论文 variance 的歧义见 README。
    spatial_multiplier: int = 2  # 通道组合输出倍数 = 输入通道数 × 此倍数。
    encoder_blocks: int = 2  # 串联的残差子编码器数量。
    kernel_size: int = 31  # 编码器 Conv1d 沿频谱轴的卷积核长度，须为正奇数。
    head_multiplier: int = 6  # 分类头隐藏层宽度 = 类别数 × 此倍数。
    dropout: float = 0.5  # 训练时随机丢弃特征的概率；监测和测试时自动关闭。
    epochs: int = 100  # 每折每种子的固定训练轮数；使用最后一轮，不按测试成绩选模型。
    batch_size: int = 128  # 训练及评估批大小，末批不足此数时保留。
    learning_rate: float = 0.001  # SGD 学习率，当前不使用学习率调度器。
    momentum: float = 0.9  # SGD 动量系数。
    weight_decay: float = 0.001  # SGD 的 L2 权重衰减系数。

    cpu_threads: int = 8  # PyTorch CPU 计算线程数；各折/种子按顺序运行。
    monitor_every: int = 5  # 固定训练子集监测间隔（epoch）；第 1 轮和最后一轮也监测。
    checkpoint_every: int = 5  # checkpoint 覆盖保存间隔（epoch）；最后一轮和 STOP 停止前也保存。
    gaze_seconds: float = 0.5  # ITR 的额外注视转移时间（秒），总决策时间 = duration + 此值。
    test_subject: int = 1  # 内部单折上下文；由调度器按 1…N 自动赋值，不用于选择起始折。

    @property
    def output_dir(self):
        """实验总目录；单折配置中 experiment 自动附加 Sxx_seedxxxx。"""
        return ROOT / "results" / self.experiment

    @property
    def samples(self):
        """分类窗口采样点数，自动推导。"""
        return round(self.duration * self.sample_rate)

    @property
    def start(self):
        """分类窗口起点的零基采样索引，自动推导。"""
        return round(self.start_seconds * self.sample_rate)

    @property
    def nfft(self):
        """FFT 总点数（含补零），自动推导。"""
        return round(self.sample_rate / self.frequency_resolution)

    @property
    def features(self):
        """单通道实部和虚部拼接后的特征长度，自动推导。"""
        return 2 * (round((self.frequency_range[1] - self.frequency_range[0]) / self.frequency_resolution) + 1)

    def validate(self):
        assert self.device in ("cpu", "cuda") or (
            self.device.startswith("cuda:") and self.device[5:].isdigit()), "device 应为 cpu、cuda 或 cuda:编号"
        assert isinstance(self.N, int) and 1 <= self.N <= self.subjects, "N 必须在 1…subjects 之间"
        assert isinstance(self.M, int) and self.M >= 1, "M 必须为正整数"
        assert isinstance(self.seed, int) and 0 <= self.seed < 2**32
        assert self.subjects == 35, "当前加载器针对完整的 35 被试 Benchmark；折数请改 N"
        assert 1 <= self.test_subject <= self.subjects
        assert self.classes == 40 and self.blocks == 6 and self.sample_rate == 250
        assert 0 < self.samples <= self.nfft and self.start + self.samples <= 1500
        assert abs(self.samples - self.duration * self.sample_rate) < 1e-8
        assert abs(self.sample_rate / self.nfft - self.frequency_resolution) < 1e-8
        assert len(set(c.upper() for c in self.channels)) == len(self.channels)
        assert 0 < self.frequency_range[0] < self.frequency_range[1] < self.sample_rate / 2
        assert self.start >= 0 and abs(self.start - self.start_seconds * self.sample_rate) < 1e-8
        assert all(abs(f / self.frequency_resolution - round(f / self.frequency_resolution)) < 1e-8
                   for f in self.frequency_range), "频谱端点须落在 FFT 频点上"
        assert self.kernel_size > 0 and self.kernel_size % 2 and 0 <= self.dropout < 1
        assert min(self.epochs, self.batch_size, self.monitor_every, self.checkpoint_every) > 0
        assert min(self.cpu_threads, self.filter_order, self.encoder_blocks,
                   self.spatial_multiplier, self.head_multiplier) > 0
        assert self.init_std > 0 and self.learning_rate > 0
        assert self.momentum >= 0 and self.weight_decay >= 0 and self.gaze_seconds >= 0

    def runs(self):
        """每折独立配置；修改 N/M 不改变已有单折的路径与训练参数。"""
        assert self.seed + self.M <= 2**32, "连续种子不能超出 NumPy 支持范围"
        for subject in range(1, self.N + 1):
            for seed in range(self.seed, self.seed + self.M):
                yield replace(self, test_subject=subject, seed=seed,
                              experiment=f"{self.experiment}/S{subject:02d}_seed{seed}")

    def to_dict(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


CONFIG = Config()
