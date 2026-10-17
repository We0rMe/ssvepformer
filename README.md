# SSVEPformer / FB_SSVEPformer

## 运行

项目与环境准备：

```powershell
git clone https://github.com/We0rMe/ssvepformer
cd SSVEPformer
conda create -n ssvepformer-gpu python=3.11 pip -y
conda activate ssvepformer-gpu
python -m pip install -r requirements.txt
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

实验环境使用 64 位 Python 3.11、`torch==2.14.1+cu126`。
把 Benchmark 数据路径填写到 `config.py` 的 `data_dir` 处，并检查 `device="cuda:0"`、`experiment="ssvepformer_loso_cuda"`、`N=35`、`M=3`、`seed=2026`。
注意：
- GPU 不可用或编号错误会直接报错，不会悄悄退回 CPU。
- 切换 input_type 会自动使用独立目录；改变其他训练设置或重复整组实验时再更换 experiment，已有不匹配配置会报错。


设备检查，而后启动训练SSVEPformer：

```powershell
python -u -m smoke.gpu.check
python -u train.py
# 训练结束会自动测试；需要重新复核模型时可执行：
python -u evaluate.py
```

若训练中断则可重新运行，会自动断点继训。

设备检查使用合成数据，对 CPU 及所有可见 GPU 分别测试两个模型的前向/反向、评估和断点恢复；无正式训练。
结果位于 `smoke/gpu/results/checks.json`。若环境无 GPU 时明确记录 CUDA 检查为 SKIPPED。

`device="cuda:1"` 使用第二张显卡；`device="cpu"` 使用 CPU。单个进程仅使用所选的一张卡，不会自动双卡分摊 35 折。
如需同时训练两个模型，可在启动前把基础模型设置为 cuda:0，把 `fb/config.py` 中的 FB 设置为 cuda:1，再分别运行两个入口。

设置fb/config.py内的额外参数后（剩余参与继承于SSVEPformer.config），启动训练FB-SSVEPformer：

```powershell
python -u -m fb.train
python -u -m fb.evaluate
```

两模型使用独立结果目录；不要同时对同一个实验目录训练和评估。

## 输入特征对比

SSVEPformer 修改 `config.py`，FB 修改 `fb/config.py` 中的 `input_type`。两者各自选择，互不跟随。
保持 N=35、M=3、seed=2026、duration=1.0、batch_size=128 及其他训练设置一致。输入变化时重新初始化训练，不载入另一种输入的权重。

| input_type | 含义 | 单个子网输入 | SSVEPformer 参数量 | 三子带 FB 参数量 |
|---|---|---|---:|---:|
| `complex` | 每个电极内按 `[Re, Im]` 沿特征轴拼接，原实验 | 9×562 | 3,099,184 | 9,297,555 |
| `time` | 滤波后裁取时域，不补零、不 FFT | 9×250 | 1,239,664 | 3,718,995 |
| `complex_channels` | `[Re(CH1..CH9), Im(CH1..CH9)]` 沿通道轴拼接 | 18×281 | 2,681,726 | 8,045,181 |

例如先设 `input_type: str = "time"`，运行 `python -u train.py` 或 `python -u -m fb.train`；完成后改为 `"complex_channels"` 再运行相同命令即可。
两种新增实验都用原来的 35 折、2026–2028 三个种子。现有 `results_server/` 中已完成的 complex 结果直接作基线，本次不会重新训练或移动它们。
当前配置保留 complex 默认值；启动新增实验前请显式选择 time 或 complex_channels。

原实现是在 `data.py` 的预处理阶段完成实虚拼接并缓存。现在仍把频域的基准缓存存成每电极 `[Re, Im]`，每折划分后由 `network_input` 一次性转换成网络所需布局；训练与评估共用转换，不在每个 batch 重复 FFT。
complex 和 complex_channels 共用同一份频域缓存。time 使用独立的滤波后时域缓存，保持原微伏数值，不新增全局归一化或拟合统计量。
FB 使用自己的子带滤波缓存；每个子带内采用相同转换，不混合子带轴。第一次运行新版会因处理代码摘要变化建立新缓存，以后各折/种子复用。

采用先全部实部、后全部虚部的通道顺序，便于检查与逆变换。交错顺序只是固定通道置换：首层为覆盖全部输入通道的 1×1 卷积，适当置换权重后两种顺序的表达能力相同，但相同种子下不保证数值训练轨迹一致。
沿通道拼接完整保留原来的复频谱数值，没有新增/丢弃相位信息；改变的是网络怎样混合实虚部。
`channels` 始终是 9 个实际电极，`input_channels` 和 `features` 自动推导。通道组合保持 2×输入通道数规则，因此三种输入的输出通道为 18、18、36；LayerNorm、MLP 与分类头尺寸随之适配。
这不是参数量严格匹配的对照：表示方式改变也改变了模型参数量和卷积轴的含义，应结合上述形状与参数量解读，不能只把差异归因为信息量。暂不实现幅度谱或 b/a。

正式论文表 7（Dataset 2，1 秒）的时域 ACC 为 71.21 ± 23.38% / 74.55 ± 23.52%，复频谱为 80.40 ± 20.45% / 83.19 ± 18.87%（依次为 SSVEPformer / FB）。表 7 没有给出 ITR。
论文的 ± 表示被试间差异，应与本项目逐种子的 35 被试统计对照；三种子均值的 ± 是种子间变化，两者不可混用。

新版输出路径为 `results/<experiment>/<input_type>/Sxx_seedxxxx/`，总报告在 `<input_type>/report.md`。模型配置、参数量和输入形状记录在各折 `reproducibility/settings.json`；训练日志及单折报告也显示形状和参数量。
评估输入类型从 checkpoint 读取，不会随当前 input_type 覆盖保存模型的结构。旧 checkpoint 缺少该字段时按 complex 解读，复核结果另存 reevaluation；迁移后的服务器模型请显式配置 evaluation_checkpoint 与本地 evaluation_data_dir。
本次实现变更后不接续旧版训练 checkpoint；旧结果与代码快照保留用于复现，新的对比从头训练。

输入专项检查：`python -u -m smoke.inputs.check`。CPU 上检查三种输入、缓存复用、被试隔离、模型梯度及小型完整训练/评估；输出统一在 `smoke/inputs/results/`。

## 项目结构

主目录五个功能入口/模块，一个共用的设备辅助模块(runtime.py)：

| 文件 | 职责 |
|---|---|
| `config.py` | 数据路径、N/M 调度、评估入口、窗口、模型和优化器参数 |
| `data.py` | 数据审查、滤波、FFT、缓存、被试划分 |
| `model.py` | SSVEPformer |
| `train.py` | 训练、训练集监测、断点与日志 |
| `evaluate.py` | 模型评估、ACC/ITR、CSV、报告和图片 |
| `runtime.py` | 两模型及评估共用的 CPU/CUDA 设置、随机状态恢复、优化器动量迁移与设备记录 |

runtime 保留为一个共用辅助模块，避免训练与评估的设备逻辑重复或相互导入。已删除耗时外推函数及两处调用，仅保留每 epoch 实测耗时。

## 输出结构

```text
results/
  ssvepformer_loso_cuda/<input_type>/
    report.md                  # 完成进度、各个种子下的被试均值 ± 标准差
    summary.csv                # 每折每种子一行关键测试指标
    reproducibility/plan.json  # 当前调度计划
    S01_seed2026/               # 每折/种子一个目录；后续为 S02_seed2026 等
      report.md
      metrics.csv
      training.png
      confusion.png
      reproducibility/         # 配置、审查、模型与详细记录
        settings.json
        data_audit.json
        history.csv
        predictions.csv
        checkpoint.pt
        status.json
        training.log
        source/
  previous_s01/                # 旧实现的结果，单独归档
  ssvepformer_loso/            # 已停止的 CPU 实验，保留结果和 S26 断点
smoke/
  check.py
  results/                     # 新的检查与初始化诊断
  previous/                    # 旧冒烟脚本及结果
```

训练期间先生成 `training.png`，最终评估结束才生成测试指标、报告和混淆矩阵。
已有匹配的预处理缓存会复用；仅数据源/预处理设置变化时重新预处理。不同折、种子共用逐试次独立计算的特征缓存，没有跨被试拟合统计量。
每 5 epoch 覆盖保存一次 checkpoint；重新运行会跳过完成项、恢复中断项、从头训练未开始项。
若要同配置完整重跑，请使用新的 experiment。
如需在 epoch 结束处保存并停止整个计划，可在另一个 PowerShell 终端执行：

```powershell
New-Item G:\SSVEP\SSVEPformer\results\ssvepformer_loso_cuda\time\reproducibility\STOP -ItemType File
```

也支持在当前单折的 reproducibility 内创建 STOP；停止后不会继续下一折或下一个种子。
恢复前删除对应 STOP，再运行 train.py。Ctrl+C 则恢复至最近定期保存的 checkpoint，可能重算最多约 5 个 epoch。

总报告在每次完成后更新，未全部完成时标为“部分结果”。每个种子单独计算被试间均值 ± 样本标准差（ddof=1）。
若 M>1，全部完成后还计算“每种子的被试均值”在种子间的均值 ± 标准差，不混淆两种变异来源。只有一个样本时不制造标准差。

STOP 示例针对 time；其他输入换成对应的输入目录名。

## 原 complex 输入的数据与模型口径

- 通过通道名定位九个通道，零基索引为 `[60,61,62,54,55,56,47,53,57]`。
- 每名被试 240 个试次；保留 MAT 的类别顺序，不按频率排序标签，不重叠切窗。
- 1 秒窗口对应 `160:410`，250 点补零到 1250；频谱取含端点的 8–64 Hz。
- FFT 除以实际窗口点数；实部后接虚部，输入为 `9×562`，无跨被试拟合归一化。
- `Conv1d(9,18,1)` 通道组合；两个独立残差编码器；普通 `Conv1d(18,18,31)`；每通道共享 `Linear(562,562)`，保留 GELU；MLP 头为 `10116→240→40`。
- LayerNorm 在每个空间通道的频谱维 F 上计算，沿用参考实现；论文正文的“channel dimension”未给出无歧义的 normalized_shape。
- SGD：学习率 0.001、momentum 0.9、weight decay 0.001；batch 128、dropout 0.5，固定 100 epoch。

每折留出被试的 240 个试次全部用于最终测试，其余 34 名被试的 8160 个试次全部用于训练。
训练集监测使用每个训练被试的第一个 block，共 1360 个训练试次，关闭 Dropout；它仅用于观察拟合，**不是独立验证集**。
在同一折中，每次监测固定使用相同的第一 block；第 1、5、10、…、100 轮监测。橙线只是连接这些实际测量点，没有另行拟合。
关闭 Dropout 可消除随机丢弃特征带来的波动，观察完整网络的拟合能力；蓝线则边更新参数边累计所有训练批次，Dropout 开启，二者并非同口径。
橙线也仅来自训练集的一个子集，往往高于蓝线，不能据此判断新被试上的泛化表现。
不按监测值早停/选 checkpoint，每折的留出被试不参与该折优化和选模；固定最后一轮模型。
此前初始化诊断使用过 S2–S17 的部分数据。因此本次全被试 LOSO 属于已固定设置后的复现评估，不应表述为完全未经任何开发观察的独立外部验证。

ITR 采用 40 类、`T=数据长度+0.5秒`，低于随机水平时记 0。
各单次实验报告 ACC 与 ITR；均值与标准差在总报告中计算。
已有 S1/2026：223/240，ACC 92.92%，ITR 183.14 bits/min。论文的 80.40% ACC 和 149.95 bits/min 是 35 名被试的平均结果；单折不能等同于完整复现。

参考：[正式论文](https://doi.org/10.1016/j.neunet.2023.04.045)
