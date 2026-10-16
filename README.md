# SSVEPformer：Benchmark LOSO 评估

默认 **SSVEPformer、cuda:0、N=35 折、M=1 个种子、seed=2026、1 秒、100 epoch**。
第 i 折仅用 Si 测试，其他全部 34 名被试训练；依次运行 S1…SN，每折依次使用 seed…seed+M-1。
各折/种子均独立初始化模型和优化器，不运行 FB 模型。
新 GPU 实验使用 `ssvepformer_loso_cuda`，从 S1 重新训练；原 CPU 实验 `ssvepformer_loso` 独立保留，不混合统计。
2026-10-16 原 CPU 任务已完成 S1–S25，并在 S26 第 24 轮保存后停止；STOP 文件仍保留在原实验中。

## Windows GPU 服务器

建议在服务器上重新训练完整 35 折，保证同一实验的设备与数值设置一致。旧 CPU 成绩与 checkpoint 保留作对照。
CPU→GPU 续练在技术上可迁移参数，但不同设备不保证相同随机轨迹；本实现不自动接受旧版/跨设备训练断点。
同一设备、实现、配置及环境下的新版实验支持正常断点恢复。单独评估保存的模型可以选择不同设备。

先在本机提交并推送代码，然后在服务器 Conda 终端中克隆项目、新建环境，无需复制 `.venv`：

```powershell
git clone "你的私人仓库URL" SSVEPformer
cd SSVEPformer
conda create -n ssvepformer-gpu python=3.11 pip -y
conda activate ssvepformer-gpu
python -m pip install -r requirements.txt
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

环境使用 64 位 Python 3.11、`torch==2.14.1+cu126`；本机 CPU 环境没有被重新安装。
把 Benchmark 数据单独复制到服务器，再修改 `config.py` 的 `data_dir` 为实际位置。
检查 `device="cuda:0"`、`experiment="ssvepformer_loso_cuda"`、`N=35`、`M=1`、`seed=2026`。
GPU 不可用或编号错误会直接报错，不会悄悄退回 CPU。

在服务器先运行设备检查，然后启动训练：

```powershell
python -u -m smoke.gpu.check
python -u train.py
# 训练结束已自动测试；需要重新复核模型时：
python -u evaluate.py
```

设备检查使用合成数据，对 CPU 及所有可见 GPU 分别测试两个模型的前向/反向、评估和断点恢复；无正式训练。
结果位于 `smoke/gpu/results/checks.json`。本机没有 GPU 时明确记录 CUDA 检查为 SKIPPED，不能把它当成服务器实测通过。

`device="cuda:1"` 使用第二张显卡；`device="cpu"` 使用 CPU。单个进程仅使用所选的一张卡，不会自动双卡分摊 35 折。
如需同时训练两个模型，可在启动前把基础模型设置为 cuda:0，把 `fb/config.py` 中的 FB 设置为 cuda:1，再分别运行两个入口。
两模型使用独立结果目录；不要同时对同一个实验目录训练和评估。

保持 FP32、原 batch=128 和优化器参数，启用确定性算法，关闭 AMP/TF32 和 cuDNN 自动测速。
CPU 完成滤波/FFT，GPU 完成模型计算。模型、当前批次及优化器动量在所选设备上；完整特征缓存留在 CPU。
checkpoint 保存 CPU 随机状态及当前 CUDA 设备的随机状态；配置、GPU 名称、CUDA/cuDNN 和精度设置记入复现资料。
即便同一种子，也不承诺 CPU 与 GPU、不同硬件/软件环境间逐位一致，见 [PyTorch 可复现性说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。

## 训练需要多久

服务器尚未实测，不能只凭 RTX 4090D 型号确定耗时。SSVEPformer 的 35 折单种子共 3500 个 epoch，每 epoch 为 64 个训练 batch。
日志每 5 轮用近期实际耗时外推训练总时长（含训练监测，排除首轮预热），不会读取测试准确率来选择配置。

| 服务器实测平均每轮耗时 | 3500 轮训练部分 |
|---|---:|
| 2 秒 | 约 1.94 小时 |
| 5 秒 | 约 4.86 小时 |
| 10 秒 | 约 9.72 小时 |

这些是条件计算，不是显卡速度预测；另加预处理、保存/绘图、最终评估开销。
直接开始正式训练即可测得速度，前几轮会继续用于这次实验，不需要为测速重新开始。
FB 的各子网预训练和联合训练分别估时，不能把某一阶段的速度直接当成 FB 全部阶段速度。

## 使用

服务器激活上述环境后直接使用 `python`；本地现有 CPU 环境则先设 `device="cpu"` 和一个新的实验名，再使用 `.venv\python.exe`。日志显示在当前终端：

```powershell
Set-Location G:\SSVEP\SSVEPformer
.\.venv\python.exe -u train.py
```

修改参数只需编辑 `config.py`，每个字段都有用途注释。`N` 范围为 1…35；`M` 为正整数。
`N=5` 表示只做 S1…S5 的五折，每折依然用其他 34 名被试训练。
`M=3, seed=2026` 表示每折分别使用 2026、2027、2028 三个种子。
`test_subject` 是调度器自动填写的单折上下文，不用手动修改；`subjects/classes/blocks/sample_rate` 是数据属性，不能当作任意超参数修改。

实验名 `experiment` 决定结果总目录；改变数据处理、模型或训练设置时使用新实验名，避免混合结果。
只增减 N/M 不影响已有单折，增加后补跑缺少的实验；减少后汇总只包括当前计划。
日志就在运行命令的当前终端中输出，不启动其他窗口或后台程序。不要同时启动训练与批量评估。
本机 CPU 环境已建立在 `.venv/`，无需激活即可使用，不要用当前面向 GPU 的 requirements 覆盖它。

`train.py` 在每次训练结束后自动测试并更新汇总，因此不必额外运行评估才能得到结果。
完成后如需重新从保存的模型复核，可运行：

```powershell
.\.venv\python.exe -u evaluate.py
```

评估不训练模型，模型/预处理参数读取 checkpoint，执行设备使用当前配置的 device：

- `evaluation_checkpoint = None`：按当前 experiment、N、M、seed 定位所有计划中的模型；跳过不存在或未训练完的模型。
- 只评估某个模型：例如 `evaluation_checkpoint = ROOT / "results/ssvepformer_loso_cuda/S01_seed2026/reproducibility/checkpoint.pt"`。
- `evaluation_data_dir = None`：使用该模型保存的 `data_dir`；如果同一份 Benchmark 数据迁移，可填写 `Path(r"D:\data\benchmark")`。目录需包含全部 35 个 MAT 文件和辅助文件。它不支持任意格式的新数据集。
- 留出被试编号来自 checkpoint，不由当前 `test_subject` 指定。更换数据目录或评估设备时，复核报告另存到 checkpoint 所在目录的 `reevaluation/`，不覆盖原实验成绩，也不纳入原实验汇总。

默认复核更新该折的报告与图像；批量模式同时更新总汇总，指定单模型时不更新总汇总。
快速检查：`.\.venv\python.exe smoke\check.py`，输出只放在 `smoke/results/`。

主目录保留五个功能入口/模块，增加一个两个模型共用的设备辅助模块：

| 文件 | 职责 |
|---|---|
| `config.py` | 数据路径、N/M 调度、评估入口、窗口、模型和优化器参数 |
| `data.py` | 数据审查、滤波、FFT、缓存、被试划分 |
| `model.py` | SSVEPformer |
| `train.py` | 训练、训练集监测、断点与日志 |
| `evaluate.py` | 模型评估、ACC/ITR、CSV、报告和图片 |
| `runtime.py` | CPU/CUDA 设置、随机状态恢复、运行设备记录、耗时提示 |

## 输出

```text
results/
  ssvepformer_loso_cuda/
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
New-Item G:\SSVEP\SSVEPformer\results\ssvepformer_loso_cuda\reproducibility\STOP -ItemType File
```

也支持在当前单折的 reproducibility 内创建 STOP；停止后不会继续下一折或下一个种子。
恢复前删除对应 STOP，再运行 train.py。Ctrl+C 则恢复至最近定期保存的 checkpoint，可能重算最多约 5 个 epoch。

总报告在每次完成后更新，未全部完成时标为“部分结果”。每个种子单独计算被试间均值 ± 样本标准差（ddof=1）。
若 M>1，全部完成后还计算“每种子的被试均值”在种子间的均值 ± 标准差，不混淆两种变异来源。只有一个样本时不制造标准差。

## 这次审查与修正

旧结果已从 checkpoint 重算：S1 正确 135/240，ACC 56.25%；关闭 Dropout 的完整训练集 ACC 为 51.94%，最后一轮开启 Dropout 的训练 ACC 为 19.14%。
因此旧测试值计算正确，但旧日志没有清楚区分模型模式，且训练收敛不足。

新版采用 `init_std=0.01`（卷积/线性层 bias=0），与用户提供的第三方实现一致。
正式论文写的是“variance 0.01”，字面对应 `std=0.1`，这是原文与参考实现的差异，不能声称两者相同。
仅用 S2–S17 的 640 个训练试次、相同种子和顺序诊断 15 epoch，`std=0.1` / `0.01` 的训练集评估 ACC 分别为 3.44% / 10.31%；该小规模诊断支持使用较小初始化继续复现，但不等于证明最终精度。

预处理统一采用 8–64 Hz 四阶 Butterworth 零相位带通，与参考项目的数据流程一致，再裁窗、FFT。
论文明确为时域基线规定此滤波，对频域输入的措辞存在歧义；旧实现未额外带通，新实现将此选择明确写入配置。
滤波逐试次独立执行于完整 6 秒试次，然后提取 `[0.64,1.64)` 秒。它使用窗口外的同试次时间上下文，属于离线协议，不能宣称在线因果性能。
采集时已经做过 50 Hz 陷波，不重复施加。

## 数据与模型口径

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

参考：[正式论文](https://doi.org/10.1016/j.neunet.2023.04.045)、[第三方实现](https://github.com/okbalefthanded/ssvepformer)。
