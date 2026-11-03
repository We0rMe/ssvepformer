# FB-SSVEPformer

FB = Filter Bank，即滤波器组。它让多个独立的 SSVEPformer 分别观察不同的谐波范围，最后学习如何合并各自的分类分数。

## 与当前 SSVEPformer 的关系

本实现单独放在 `fb/` 下。原 CPU 任务已保存并停止；两个模型均已适配 CPU/CUDA。
共用原始 MAT 读取、通道顺序、FFT、被试隔离、SSVEPformer 子网、单轮 SGD 和 ACC/ITR 计算。
FB 的缓存以 `fb_` 开头，实验目录也以 `fb_` 开头，不使用基础模型的 checkpoint。
设备、CUDA 随机状态和优化器缓冲区由根目录 `runtime.py` 共用。服务器环境准备见根 README。

| 文件 | 用途 |
|---|---|
| `config.py` | 顶部常用实验选项，下方固定设置与检查 |
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

## 如何运行

修改 `fb/config.py`，数据路径默认沿用根配置；输入、窗口、训练参数独立。固定滤波与结构参数放在同一文件的常用选项下方。默认 cuda:0、N=35、M=3、complex、1 秒、关闭早停；原 FB 网络与滤波器组保持不变。

`input_type` 支持 complex、amplitude、time、complex_channels；输入形状分别为每子带 9×562、9×281、9×round(250×duration)、18×281。幅度谱是复频谱的模，不是功率谱。

```powershell
python -u -m fb.train
python -u -m fb.evaluate
```

必须从项目根目录以模块方式启动。单独复核 checkpoint 时，在 fb/config.py 中设置 evaluation_checkpoint；字符串路径也可直接使用。数据迁移后设置 evaluation_data_dir。

早停关闭时为 34 人训练、1 人测试，三个子网各 100 轮后联合 20 轮。开启时为 30 人训练、4 人验证、1 人测试，各子网独立按验证交叉熵早停并恢复最佳权重；之后联合阶段再独立早停。子网默认至少 30 轮、patience=15；联合至少 5 轮、patience=min(early_patience,5)，均不超过配置的阶段上限。验证名单跨表示、窗口、种子固定；不使用测试被试选模。

输出自动进入 `results/<fb_experiment>/<input_type>/<duration>s/<fixed或early>/`。每个阶段的实际执行轮数、采用轮次和停止原因保存在单折 reproducibility/selection.json。新 checkpoint 支持恢复当前阶段、优化器、早停计数和随机状态；已完成子网不重复训练。

在计划或单折 reproducibility 目录创建 STOP 可停止整个计划，删除后用相同命令恢复。原有结果保留；旧格式断点仅作显式评估，不接续新版训练。缓存、配置公平性、报告和完整操作说明见[根 README](../README.md)。

新功能检查：`python -u -m smoke.experiments.check`，无需原始数据，覆盖两个模型、四种输入、三种窗口、早停和断点恢复。设备检查：`python -u -m smoke.gpu.check`。原 `smoke.fb.check` 仍可在具备 Benchmark 数据的环境运行滤波与真实试次检查。
