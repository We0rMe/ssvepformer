# FB-SSVEPformer

FB = Filter Bank，即滤波器组。它让多个独立的 SSVEPformer 分别观察不同的谐波范围，最后学习如何合并各自的分类分数。

## 与当前 SSVEPformer 的关系

本实现单独放在 `fb/` 下。根目录的五个核心脚本保持不变，正在执行的基础模型训练不需要重启。
共用原始 MAT 读取、通道顺序、FFT、被试隔离、SSVEPformer 子网、单轮 SGD 和 ACC/ITR 计算。
FB 的缓存以 `fb_` 开头，实验目录也以 `fb_` 开头，不使用基础模型的 checkpoint。
现有环境已具备全部依赖，无需安装或更新软件包。

| 文件 | 用途 |
|---|---|
| `config.py` | FB 参数和共享参数的覆盖处 |
| `data.py` | 时域滤波器组、特征缓存 |
| `model.py` | 独立子网及可学习融合层 |
| `train.py` | 两阶段训练、恢复、LOSO 调度 |
| `evaluate.py` | 最终测试、阶段曲线、结果汇总 |

## 先读懂数据流程

基础 SSVEPformer 当前的处理顺序是：

```text
原始完整试次（6 秒）
→ 8–64 Hz 四阶 Butterworth 零相位滤波
→ 裁取 [0.64, 0.64+d) 秒
→ FFT 的 n 参数自动补零（默认补至 1250 点，0.2 Hz 分辨率）
→ 选取 8–64 Hz 频点，含两端
→ 实部在前、虚部在后拼接（每通道 562 维）
```

FB 的处理顺序是：

```text
                         → 子带 1： 9–80 Hz → 裁窗 → FFT → 8–64 Hz 复频谱 → 子网 1 ┐
同一个原始完整试次 ───────→ 子带 2：18–80 Hz → 裁窗 → FFT → 8–64 Hz 复频谱 → 子网 2 ├→ 学习加权融合
                         → 子带 3：27–80 Hz → 裁窗 → FFT → 8–64 Hz 复频谱 → 子网 3 ┘
```

各子带从原始试次分别滤出，不先做基础模型的 8–64 Hz 滤波，也不在 FFT 后再分带。
子带在频率上相互重叠；80 Hz 是时域滤波通带上界，64 Hz 是最终特征上界，二者不是同一个设置。
FFT 用实际窗口采样点数归一化；补零加密频率采样网格，并不会增加原始信号所含的信息。
默认每个样本形状是 `[3 子带, 9 通道, 562 特征]`。

每个试次单独进行整试次零相位滤波，然后裁窗；使用窗口外的同试次上下文，属于离线评估。
没有跨试次拼接滤波，也没有使用全体被试估计均值/方差等统计量。

## 论文已规定和实现自行补充的部分

正式论文 2.2、2.5、2.6 节及图 3 规定了 Dataset 2 的 `[9m,80]` Hz 子带、默认 3 子带、8–64 Hz 复频谱、各子网预训练 100 轮及全网联合训练 20 轮。
论文没有完整给出滤波器类型、阶数、波纹、上侧阻带位置；“下界额外 2 Hz 带宽”也未给出明确的滤波器设计调用。

本实现明确采用以下可调整选择，不能声称是作者源代码的原样实现：

- Chebyshev-I，通带 `[9m,80]` Hz，阻带边界 `[9m-2,90]` Hz；2 Hz 解释为下侧过渡带。
- 单向滤波器通带最大波纹 0.5 dB、阻带最小衰减 40 dB，`cheb1ord` 自动定阶，`cheby1` 使用相同的 0.5 dB。随后用 `sosfiltfilt` 双向零相位滤波，理想幅频响应变为单向响应的平方。
- 每个子网复用当前项目已采用的 `init_std=0.01`；融合卷积也如此初始化。正式论文的 variance=0.01 与此不同，根 README 已记录该歧义。
- 使用 `Conv1d(S,1,1,bias=False)` 融合子网 logits，各类别共享子带权重；权重可正可负，不提前 softmax，也不使用 FBCCA 的固定权重公式。
- 每个预训练子网使用独立 SGD；联合阶段新建 SGD，更新全部子网和融合层，不冻结子网。论文没有说明跨阶段是否保留动量，本实现不保留。

参考：[正式论文](https://doi.org/10.1016/j.neunet.2023.04.045)、[SciPy 定阶](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb1ord.html)、[SciPy 滤波器设计](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html)。
用户提供的[第三方项目](https://github.com/okbalefthanded/ssvepformer)介绍的是 Dataset 1 的基础模型复现，不能作为本 FB 实现在 Benchmark 上达到论文精度的证据。

## 如何修改参数、启动与复核

修改 `fb/config.py`，不必改动根配置。`FBConfig` 继承根 `Config` 默认值，因此通用参数的含义一致。
需要调整学习率、线程数或评估路径时，可直接在 `FBConfig` 中同名覆盖，例如 `cpu_threads: int = 4`。
继承的 `filter_order` 不用于 FB；`bandpass` 固定为 False，表示不额外串联基础 Butterworth。
`frequency_range` 在 FB 中仅控制 FFT 特征范围；`band_step/band_high` 控制子带通带。

默认 N=1、M=1、seed=2026，先评估一个留出被试。N=35 才运行完整 LOSO。
每折仍然使用其余全部 34 名被试训练；同一折的不同种子从头初始化，不共享训练所得权重。
每次默认实际执行 3×100+20=320 个阶段内 epoch，其中联合 epoch 同时训练三个子网，耗时高于基础模型。

当前基础模型仍在运行，FB 代码准备完成后由你决定启动时间；本次实现不自动启动 FB 正式任务。
准备运行时，在项目根目录 PowerShell 中执行：

```powershell
Set-Location G:\SSVEP\SSVEPformer
.\.venv\python.exe -u -m fb.train
```

日志直接显示在当前终端。不能用 `python fb/train.py` 代替 `-m fb.train`，否则同名模块的导入路径可能混淆。
训练完成自动测试，无需再运行评估才能得到结果。另行复核保存模型：

```powershell
.\.venv\python.exe -u -m fb.evaluate
```

默认复核 FB 当前 N/M 计划中已完成联合训练的模型，预训练 checkpoint 不用于测试。
单模型复核可将 `evaluation_checkpoint` 改为 `Path(r"完整的 FB checkpoint.pt 路径")`。
数据路径默认从 checkpoint 读取；同一份 Benchmark 迁移后可同名覆盖 `evaluation_data_dir`。指定新数据路径的复核结果另存至 checkpoint 旁的 `reevaluation/`，不会改写原成绩。

## 缓存、恢复和输出

首次运行生成独立 FB 缓存：一个 `features.npy`（默认约 486 MiB）和 `audit.json`。
以后各折/种子共用该缓存。N、M、seed 不进入缓存标识；数据文件路径/大小/修改时间、预处理设置和相关处理代码变化才会产生新缓存。
“每折独立配置”是为每次实验填写测试被试、种子、结果路径，并非复制或重新处理整套原始数据。

```text
results/fb_ssvepformer_loso/
  report.md, summary.csv                 # 总进度和均值 ± 标准差
  reproducibility/plan.json
  S01_seed2026/
    report.md, metrics.csv               # 关键结果
    training.png, confusion.png          # 按阶段分行绘制的训练曲线；最终混淆矩阵
    reproducibility/                    # 配置、数据审查、训练记录、预测、融合权重、模型、代码快照
```

每个实验只保留一个最新 checkpoint，包含所有子网、融合层、当前阶段的优化器、阶段编号、epoch 和随机数状态。
中断后重新运行原命令会恢复当前阶段，不重复已完成的子网；完成项会跳过。
只增减 N/M 可补跑/缩小计划；更换子带数、滤波或训练设置时请换新的 `experiment`。

创建 `results/fb_ssvepformer_loso/reproducibility/STOP` 可在 epoch 边界保存并停止整个 FB 计划。
也可在当前单折的 reproducibility 中创建 STOP；恢复前删除它。Ctrl+C 恢复至最近一次定期保存的 checkpoint。
FB 的 STOP、锁、日志与基础模型完全分开。

训练监测始终只取本折训练被试的第一 block，关闭 Dropout；不按监测值早停。
测试被试只在联合训练结束后使用。各折独立 LOSO；此前开发诊断使用过部分被试的数据，这并非未经开发观察的外部验证。
ITR 的决策时间是窗口长度加 0.5 秒。总报告区分被试间与种子间标准差，只有一个结果时不报告标准差。

轻量检查脚本：`python -m smoke.fb.check`。只使用少量合成/训练侧数据，单线程，不生成正式实验或全量 FB 缓存；结果均在 `smoke/fb/results/`。
