"""实验配置：顶部修改常用选项，下方保存固定参数和自动推导规则。"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    # 常用实验选项：日常只需修改这一段。
    data_dir: Path = Path(r"I:\jjh\SSVEP\data\benchmark")
    experiment: str = "ssvepformer_complex"  # 修改其他训练设置时换实验名。
    input_type: str = "complex"  # complex / amplitude / time / complex_channels
    duration: float = 1.0  # 窗口秒数，例如 0.5、1.0、2.0；必须对应整数采样点。
    device: str = "cuda:1"
    N: int = 35  # 测试 S1…SN；每折训练候选始终来自其余 34 人。
    M: int = 3  # 每折独立种子数。
    seed: int = 2026
    early_stopping: bool = False  # True：独立 4 被试验证、恢复最佳模型；False：全 34 人、最后一轮。
    epochs: int = 100  # 固定轮数 / 早停的最大轮数。新早停实验可预先设为 200。
    early_patience: int = 15
    batch_size: int = 128  # 7200/8160 个训练试次，约 57/64 个 batch/epoch。
    learning_rate: float = 0.001
    evaluation_checkpoint: Path | None = None  # 指定旧/新 checkpoint 路径可单独复核。
    evaluation_data_dir: Path | None = None  # 服务器数据迁移后指定本地路径。

    # 固定实现参数
    subjects: int = 35  # 数据集固定属性：被试总数；控制折数请改 N，不要改此项。
    classes: int = 40  # 数据集固定属性：目标类别数，与 MAT 类别顺序一致。
    blocks: int = 6  # 数据集固定属性：每被试的 block 数，每 block 各类别一个试次。
    channels: tuple = ("O1", "Oz", "O2", "PO3", "POz", "PO4", "Pz", "PO5", "PO6")  # 按名称及此顺序取通道。
    sample_rate: int = 250  # 已降采样的数据采样率（Hz）；修改数值不会执行重采样。
    start_seconds: float = 0.64  # 从试次起点裁窗：0.5 秒提示 + 0.14 秒视觉延迟。
    frequency_resolution: float = 0.2  # FFT 频率分辨率（Hz），由采样率/此值计算补零长度。
    frequency_range: tuple = (8.0, 64.0)  # 复频谱保留范围（Hz，含端点），也是带通滤波的通带。
    bandpass: bool = True  # 是否在整段试次上独立执行零相位带通，再裁窗。
    filter_order: int = 4  # Butterworth 设计阶数，供 scipy.signal.butter 使用。
    init_std: float = 0.01  # 卷积/线性权重初始化的标准差；论文 variance 的歧义见 README。
    spatial_multiplier: int = 2  # 通道组合输出 = 网络输入通道数 × 此值；通道拼接时默认输出 36 通道。
    encoder_blocks: int = 2  # 串联的残差子编码器数量。
    kernel_size: int = 31  # 编码器 Conv1d 沿频谱轴的卷积核长度，须为正奇数。
    head_multiplier: int = 6  # 分类头隐藏层宽度 = 类别数 × 此倍数。
    dropout: float = 0.5  # 训练时随机丢弃特征的概率；监测和测试时自动关闭。
    momentum: float = 0.9  # SGD 动量系数。
    weight_decay: float = 0.001  # SGD 的 L2 权重衰减系数。
    cpu_threads: int = 8  # PyTorch CPU 计算线程数；各折/种子按顺序运行。
    monitor_every: int = 5  # 固定训练子集监测间隔（epoch）；第 1 轮和最后一轮也监测。
    checkpoint_every: int = 5  # checkpoint 覆盖保存间隔（epoch）；最后一轮和 STOP 停止前也保存。
    gaze_seconds: float = 0.5  # ITR 的额外注视转移时间（秒），总决策时间 = duration + 此值。
    test_subject: int = 1  # 内部单折上下文；由调度器按 1…N 自动赋值，不用于选择起始折。
    validation_subjects: int = 4  # 34 个开发被试中留 4 人，30 人训练（7200 trials）。
    validation_seed: int = 2026  # 固定划分，与训练随机种子独立。
    early_min_epochs: int = 30  # FB 联合阶段为 5；各阶段均不超过其轮数上限。
    early_min_delta: float = 1e-4  # 验证交叉熵下降阈值，不是准确率百分点。

    # 自动推导与合法性检查。
    @property
    def output_dir(self):
        """自动隔离实验名、输入、窗口、早停模式；单折目录位于其下。"""
        parts = Path(self.experiment).parts
        import config
        mode = "early" if self.early_stopping else "fixed"
        return config.ROOT / "results" / parts[0] / self.input_type / f"{self.duration:g}s" / mode / Path(*parts[1:])

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
    def spectral_bins(self):
        """保留的频点数，与实虚部如何拼接无关。"""
        return round((self.frequency_range[1] - self.frequency_range[0]) / self.frequency_resolution) + 1

    @property
    def input_channels(self):
        """网络输入通道数；channels 始终只表示真实电极，不把实虚部当成新电极。"""
        return len(self.channels) * (2 if self.input_type == "complex_channels" else 1)

    @property
    def features(self):
        """网络每通道的长度：时间采样点、双倍频点或单倍频点。"""
        if self.input_type == "time":
            return self.samples
        return self.spectral_bins * (2 if self.input_type == "complex" else 1)

    @property
    def cache_features(self):
        """缓存按真实电极存放；幅度谱和两种复频谱布局共用 [实部, 虚部]。"""
        return self.samples if self.input_type == "time" else 2 * self.spectral_bins

    def validate(self):
        assert self.input_type in ("complex", "time", "complex_channels", "amplitude"), "不支持的 input_type"
        assert self.device in ("cpu", "cuda") or (
            self.device.startswith("cuda:") and self.device[5:].isdigit()), "device 应为 cpu、cuda 或 cuda:编号"
        assert isinstance(self.early_stopping, bool)
        assert 1 <= self.validation_subjects < self.subjects - 1
        assert isinstance(self.validation_seed, int) and 0 <= self.validation_seed < 2**32
        assert self.early_patience > 0 and self.early_min_epochs > 0 and self.early_min_delta >= 0
        assert isinstance(self.N, int) and 1 <= self.N <= self.subjects, "N 必须在 1…subjects 之间"
        assert isinstance(self.M, int) and self.M >= 1, "M 必须为正整数"
        assert isinstance(self.seed, int) and 0 <= self.seed < 2**32
        assert self.subjects == 35, "当前加载器针对完整的 35 被试 Benchmark；折数请改 N"
        assert 1 <= self.test_subject <= self.subjects
        assert self.classes == 40 and self.blocks == 6 and self.sample_rate == 250
        assert 0 < self.samples and self.start + self.samples <= 1500
        if self.input_type != "time":
            assert self.samples <= self.nfft, "频域窗口不能超过 FFT 长度"
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
