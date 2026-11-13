# SSVEPformer / FB-SSVEPformer

课程实验三（时频融合）、四（噪声与预处理）的方案、配置和运行说明见 [实验三四设计与运行](实验三四设计与运行.md)。在 config.py 设置 COURSE_EXPERIMENTS 后用 python train.py 顺序运行；为空时保持原训练行为。新实验采用固定训练预算，不使用早停，结果写入独立目录。

## 输入和窗口

| input_type | 含义 | 基础模型 / 每个 FB 子网的输入 |
|---|---|---|
| `complex` | 沿特征轴拼接 `[Re, Im]` | 9×562 |
| `amplitude` | `sqrt(Re² + Im²)`，幅度谱 | 9×281 |
| `time` | 滤波后裁窗的时域信号 | 9×round(250×duration) |
| `complex_channels` | 沿通道轴拼接实部和虚部 | 18×281 |

注：

幅度谱是归一化 FFT 的模，不是功率谱，没有额外平方、对数或单边倍增。与复频谱使用同一个滤波、裁窗和 FFT 结果，只去掉相位；FB 对每个子带分别应用同样转换。三种频域表示共享 `[Re, Im]` 缓存，time 使用独立缓存；不同窗口独立缓存。

根据原始论文中的说明，窗口从试次起点后 0.64 秒开始，采样率 250 Hz，必须精确对应整数采样点。频域窗口最长 5 秒，FFT 固定补零到 1250 点，保留 8–64 Hz、含端点共 281 个频点；0.2 Hz 是频率采样网格，补零不会提高短窗本身的分辨能力。

网络尺寸随输入自动适配，并在报告记录参数量。时域窗口变短会减少网络参数；幅度谱和复频谱的参数量也不同。

## 运行、恢复与复核

在现有 PyTorch 环境中，从项目根目录执行。先填写真实 `data_dir`；单卡服务器用 `device="cuda:0"`，本地无 GPU 检查使用 cpu。GPU 不可用会明确报错。

```powershell
python -u train.py                   # 默认执行课程实验三、四；COURSE_EXPERIMENTS = () 时训练基础模型
python -u -m fb.train                # FB 模型
```

可用现有 requirements.txt 准备环境；运行前检查 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`。

输出路径自动隔离为：

```text
results/<experiment>/<input_type>/<duration>s/<fixed或early>/
  report.md, summary.csv              # 总进度与统计
  reproducibility/plan.json
  S01_seed2026/
    report.md, metrics.csv, training.png, confusion.png
    reproducibility/
      settings.json                  # 全量配置、实际被试名单、形状/参数量
      selection.json                 # 各阶段实际轮数、采用轮次、停止原因
      history.csv                    # 训练/训练监测/独立验证损失与准确率、耗时
      checkpoint.pt                  # 模型、优化器、阶段、早停状态、随机状态
      data_audit.json, predictions.csv, status.json, training.log
      source/                        # 源码快照
```

例如 `results/ssvepformer_experiments/amplitude/0.5s/early/`。更换输入、窗口、早停开关自动分目录；改变学习率、轮数上限等其他训练设置要更换 experiment，以免与旧断点混用。N/M 可以扩展，已完成项跳过。

训练曲线蓝线是训练批次（Dropout ON），橙线是固定训练子集监测（Dropout OFF），开启早停才有独立验证集绿线。报告区分最后执行轮次和最终采用轮次；FB 按阶段绘图。测试集不绘制逐轮曲线。

每 5 轮、STOP 请求、早停触发及阶段完成都会保存 checkpoint。完成的 checkpoint.model 是选中的最终权重；未完成时同时保存最新训练状态和当前最佳权重，恢复耐心计数与随机状态。重新执行相同训练命令即可恢复。Ctrl+C 使用最近一次已保存状态，可能重算最多约 5 轮。

在计划或单折的 reproducibility 目录创建 `STOP`，会在 epoch 边界保存并停止整个计划；恢复前删除 STOP。例如：

```powershell
New-Item -Path .\results\ssvepformer_experiments\amplitude\0.5s\early\reproducibility\STOP -ItemType File
```

重新复核用 `python -u evaluate.py` 或 `python -u -m fb.evaluate`。可在对应配置填写 evaluation_checkpoint 指定单模型；服务器数据迁移后填写 evaluation_data_dir。模型输入、窗口、早停状态从 checkpoint 读取。新旧数据路径或推理设备不同、以及旧格式 checkpoint 的复核，写入 checkpoint 旁的 reevaluation 目录。

旧结果、缓存和 checkpoint 保留；本次代码变更后不接续旧格式训练断点，新实验从头训练。旧 checkpoint 可显式指定路径复核；FB 旧数据摘要与当前实现不一致时，需要显式填写 evaluation_data_dir。已完成旧实验若要完全按当时程序复核，可使用其 source 快照。

单次报告 ACC、ITR；总报告先逐种子计算被试间均值与样本标准差，全部完成后再统计种子间变化。N×M 个结果不视为相互独立的样本。

## 课程实验三、四和历史模型复测

在根目录 config.py 的 COURSE_EXPERIMENTS 选择任务。FUSION.methods 列出实验三要评估的模型，NOISE.conditions 选择实验四的“附加噪声”与“干净数据关闭带通”两项对照。归档的时域和频率轴复频谱模型按被试和种子加载。运行方式和输出字段见 [实验三四设计与运行](./实验三四设计与运行.md)。

从项目根目录复测归档目录中的全部 checkpoint 可运行以下命令，设备编号和数据路径按机器调整。

```powershell
python -u evaluate.py --checkpoint results_server/1000ms --data-dir "G:/SSVEP/data/benchmark" --device cuda:0
```

目录内生成 latency_summary.csv。forward_ms_batch1 是设备就绪的单个试次通过完整模型的平均毫秒数。每个 checkpoint 的独立结果在其旁边的 reevaluation/metrics.csv。

## 文件职责

| 文件 | 职责 |
|---|---|
| config.py / fb/config.py | 配置：顶部常用选项，下方固定参数、推导尺寸和检查 |
| data.py / fb/data.py | 预处理、缓存；基础模块共用表示转换与被试划分 |
| model.py / fb/model.py | 原模型结构，按配置自动确定输入尺寸 |
| train.py | 基础模型入口、共用训练流程、早停、最佳权重恢复、断点与 LOSO 调度 |
| fb/train.py | FB 入口、子网预训练与联合训练调度；复用根 train.py 的训练流程 |
| evaluate.py / fb/evaluate.py | 评估、报告、曲线和汇总 |
| course/runner.py | 课程实验三、四的模型、对照条件与运行调度 |
| course/figures.py | 课程实验按被试汇总、绘图与配对差值 |
| runtime.py | 设备、随机状态及优化器迁移 |

## 原 complex 输入的数据与模型口径

- 通过通道名定位九个通道，零基索引为 `[60,61,62,54,55,56,47,53,57]`。
- 每名被试 240 个试次；保留 MAT 的类别顺序，不按频率排序标签，不重叠切窗。
- 1 秒窗口对应 `160:410`，250 点补零到 1250；频谱取含端点的 8–64 Hz。
- FFT 除以实际窗口点数；实部后接虚部，输入为 `9×562`，无跨被试拟合归一化。
- `Conv1d(9,18,1)` 通道组合；两个独立残差编码器；普通 `Conv1d(18,18,31)`；每通道共享 `Linear(562,562)`，保留 GELU；MLP 头为 `10116→240→40`。
- LayerNorm 在每个空间通道的频谱维 F 上计算，沿用参考实现；论文正文的“channel dimension”未给出无歧义的 normalized_shape。
- SGD：学习率 0.001、momentum 0.9、weight decay 0.001；batch 128、dropout 0.5，默认固定 100 epoch；开启早停时按验证集选模。

关闭早停时，每折留出被试的 240 个试次全部用于最终测试，其余 34 名被试的 8160 个试次全部用于训练。开启早停时，改为 30 人训练、4 人独立验证、1 人测试。
训练集监测使用每个训练被试的第一个 block，关闭早停时共 1360 个训练试次，开启时共 1200 个，关闭 Dropout；它仅用于观察拟合，**不是独立验证集**。
在同一折中，每次监测固定使用相同的第一 block；第 1、5、10、…、100 轮监测。橙线只是连接这些实际测量点，没有另行拟合。
关闭 Dropout 可消除随机丢弃特征带来的波动，观察完整网络的拟合能力；蓝线则边更新参数边累计所有训练批次，Dropout 开启，二者并非同口径。
橙线也仅来自训练集的一个子集，往往高于蓝线，不能据此判断新被试上的泛化表现。
不按训练子集监测值选模型。关闭早停时采用最后一轮；开启时仅按独立验证集交叉熵选模，测试被试不参与该折优化和选模。
此前初始化诊断使用过 S2–S17 的部分数据。因此本次全被试 LOSO 属于已固定设置后的复现评估，不应表述为完全未经任何开发观察的独立外部验证。

# 实验三、四补充实验

在项目根目录的 config.py 中确认 Config.data_dir 和 Config.device，运行 `python -u train.py`。默认队列 `("fusion", "noise")` 依次执行以下补充实验。只做实验三时填 `("fusion",)`，只做实验四时填 `("noise",)`，要运行基础模型原有训练流程时填 `()`。

实验三、四共用 1 秒分析窗、9 个电极和留一被试划分。频域输入固定为 `complex`，每个电极的 281 个实部值后接 281 个虚部值，输入尺寸为 9 × 562。原实验的带通时域单支、带通复频谱单支准确率直接取已有结果，按同一被试、训练种子和窗口长度配对。

## 实验三　两种时频融合

| FUSION.methods | 同一试次的两支输出如何融合 | 权重来源 |
|---|---|---|
| score_tf | 对时域和复频谱单支分别做 softmax，取两组 40 类概率的算术平均，再选最大概率类别 | 加载同折、同种子的两个单支 checkpoint |
| feature_tf | 两个编码器各自池化并投影至 64 维，拼接成 128 维后输入共同分类头 | 随机初始化，联合训练双支和分类头，默认固定 100 轮 |

两种融合都与前两项已有单支结果比较。两种融合之间的差值也按同一被试和种子计算。特征融合改变了分类头和训练方式，这些差异应计入结果解释。

## 实验四　测试加噪与带通消融

共同参照来自已有的干净复频谱实验，即整试次执行 8–64 Hz 带通、裁窗、提取复频谱。NOISE.conditions 可以选择下列任一项或两项。

| 条件 | 训练 | 测试 | 每折、每种子的新增记录 |
|---|---|---|---|
| added_noise | 加载已完成的带通复频谱模型，固定权重 | 在留出被试的完整时域试次加高斯噪声，再执行原带通、裁窗和复频谱提取 | 20、10、0 dB，每档 3 次，共 9 条 |
| no_bandpass | 用干净数据重新训练复频谱单支，关闭额外的时域带通 | 干净数据，使用相同的无带通流程 | 1 条 |

关闭带通只改变额外的时域滤波。两组均保留 0.64 秒的裁窗起点、1 秒的窗长、8–64 Hz 的 FFT 频点和实虚拼接顺序。加噪后按分析窗内每个试次、每个电极的功率比 `10 log10(P_recorded/P_added)` 设置附加 SNR。相同被试、相同重复编号在不同 SNR 使用同一份随机噪声并改变幅度。三个训练种子共用同一组噪声，便于配对。

## 如何复用权重

config.py 中的 ARCHIVE 默认指向项目内的 results_server/1000ms。每次运行按测试被试和训练种子展开 FOLD_CHECKPOINT。以 S01、种子 2026 为例，程序使用以下文件。

```text
results_server/1000ms/ssvepformer_time/time/S01_seed2026/reproducibility/checkpoint.pt
results_server/1000ms/ssvepformer_loso_cuda/S01_seed2026/reproducibility/checkpoint.pt
```

1. 若服务器保留相同目录结构，把归档放在项目的 results_server 下即可。归档在别处时，把 ARCHIVE 改成相应窗口的目录，例如 `Path("/data/archive/1000ms")`。
2. 若子目录名不同，在 COURSE_CHECKPOINTS 中修改对应路径。模板保留 `{subject:02d}` 和 `{seed}`。复频谱应选择上面的 ssvepformer_loso_cuda，频率轴实虚拼接的权重与 complex_channels 的权重对应不同输入。
3. 程序检查模型类型、训练完成状态、被试、种子、带通、窗长及训练配置，然后加载权重。指定的文件缺失或不匹配时会报告该问题。只有把相应项明确设为 None，程序才会按需训练该单支；已经在本补充实验目录训练完成的对应模型会自动复用。
4. complex_no_bandpass 默认是 None。若已有同配置的无带通模型，可在此项填写它的模板。feature_tf 在本补充实验目录按被试和种子保存，使用同一设置再次运行时可恢复断点或复用已完成的模型。

## 结果与图像

默认结果目录为 `results/course_exp34_supplement/1s/`。fusion/ 和 noise/ 各自包含 metrics.csv、summary.csv、逐试次预测及 progress.json。metrics.csv 的 condition 标识比较条件，checkpoints 记录实际权重路径。summary.csv 先对同一折、同一个训练种子的噪声重复取平均，再对训练种子取平均，最后对被试求均值。干净带通的基准取自上面复频谱归档各折的 metrics.csv。

SAVE_FIGURES=True 时保存融合配对图、加噪准确率曲线，以及 S01 第一块、第一目标、Oz 电极在加噪前后的波形和幅度谱。图像保存为 PNG 和 SVG，信号示例另附 NPZ。训练期间可在上述窗口目录或任务目录创建 STOP；删除标记并用同一设置重启即可继续。

## 用旧 checkpoint 补测单试次推理耗时

从项目根目录运行。下例复测 1 秒目录，设备编号和数据路径按实际机器填写。把目录换成某个 `checkpoint.pt` 即可只测一个模型。

```powershell
python -u evaluate.py --checkpoint results_server/1000ms --data-dir "G:/SSVEP/data/benchmark" --device cuda:0
```

命令按保存的模型配置读取该折留出的测试被试。每份 checkpoint 的结果写入其旁边的 `reevaluation/metrics.csv`，目录模式还在指定目录生成 `latency_summary.csv`，每个已完成模型占一行。已有单支 SSVEPformer、FB-SSVEPformer 和课程特征融合 checkpoint 都可使用。课程特征融合需要 checkpoint 旁的 `protocol.json`。得分融合由两份单支 checkpoint 组成，运行 `fusion` 中的 `score_tf` 即可测量完整双分支调用。

`forward_ms_batch1` 为设备上单个测试试次的平均前向耗时，单位为毫秒。先切换评估模式、关闭梯度，并用单试次预热 10 次。随后每次送入一个已在设备上的试次，并在 CUDA 上同步等待该次推理完成。每折默认对 240 个真实测试试次取算术平均。计时范围包括模型调用、分支和模型内部融合，排除文件读取、滤波、FFT、数据搬运及指标计算。准确率和 ITR 仍使用原批量评估流程。对照模型的耗时应使用同一台机器、同一设备和相同运行环境。

代码检查可运行 `python -m smoke.experiments.check_course`（两个课程实验及恢复流程），以及 `python -m smoke.experiments.check`（原基础模型与 FB 模型的回归检查）。


参考：[正式论文](https://doi.org/10.1016/j.neunet.2023.04.045)
