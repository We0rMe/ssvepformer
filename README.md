# SSVEPformer / FB-SSVEPformer

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
python -u train.py                   # 基础模型
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

## 文件职责

| 文件 | 职责 |
|---|---|
| config.py / fb/config.py | 配置：顶部常用选项，下方固定参数、推导尺寸和检查 |
| data.py / fb/data.py | 预处理、缓存；基础模块共用表示转换与被试划分 |
| model.py / fb/model.py | 原模型结构，按配置自动确定输入尺寸 |
| train.py | 基础模型入口、共用训练流程、早停、最佳权重恢复、断点与 LOSO 调度 |
| fb/train.py | FB 入口、子网预训练与联合训练调度；复用根 train.py 的训练流程 |
| evaluate.py / fb/evaluate.py | 评估、报告、曲线和汇总 |
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

参考：[正式论文](https://doi.org/10.1016/j.neunet.2023.04.045)