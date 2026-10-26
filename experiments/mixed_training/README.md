# Real/synthetic mixing with small forecasting models

本目录实现每个真实数据集独立训练的混合实验，不是跨数据集预训练。五模型、三 seed、九比例的完整网格为 **7 × 5 × 3 × 9 = 945 runs**。

本地 Exchange Rate 单数据集 135 次正式初筛已完成；指标、基线与成本结论见 [SCREENING_RESULTS.md](SCREENING_RESULTS.md)。

## 环境与运行位置

所有命令在 ForecastPFN 仓库根目录执行，推荐 Python 3.11。当前本地使用 conda base 的 `D:\software\Anacoda\python.exe`，无需激活环境；GPU 可用时自动使用 CUDA。

此套实验只需要 PyTorch、NumPy、pandas、SciPy、Matplotlib，**不需要安装根目录的旧版全量 requirements 或 TensorFlow**。为原生成器拆出了轻量 `prior_constants.py`，仅分离 TensorFlow 序列化依赖，没有重写趋势、季节性、噪声或随机参数采样公式。

```bash
python -m pip install -r experiments/mixed_training/requirements.txt
python -m experiments.mixed_training.tests
```

远程 GPU 主机应预先安装适配其驱动的 PyTorch 版本。`--device cpu` 强制 CPU，`--device cuda:0` 显式选 GPU；默认 `auto`。软件版本及设备型号会写入实验元数据。

## 直接执行完整七数据集扩展（本地未启动该训练网格）

将 `/path/to/long_term_forecast` 替换为远程真实数据根目录。不要使用 `--stage screen` 代替 `full`；后者默认选择全部七个数据集。

```bash
# 可选：仅检查真实数据、时间边界并缓存原生成器合成数据，不训练。
python -m experiments.mixed_training.run --stage prepare --data-root /path/to/long_term_forecast --output experiments/results/mixed_full

# 完整 945-run 网格；默认 2,000 updates/run，batch=64。
python -m experiments.mixed_training.run --stage full --data-root /path/to/long_term_forecast --output experiments/results/mixed_full

# 独立核验全部预测指标、来源曝光和 checkpoint 选择，再生成报告。
python -m experiments.mixed_training.verify --output experiments/results/mixed_full --expected-runs 945
python -m experiments.mixed_training.summarize --output experiments/results/mixed_full
```

正式运行之前可在**远程硬件上**先测冒烟成本，再估计完整网格。冒烟计时不能准确迁移到另一种 CPU/GPU。

```bash
python -m experiments.mixed_training.run --stage smoke --data-root /path/to/long_term_forecast --output experiments/results/remote_smoke
python -m experiments.mixed_training.run --stage estimate --data-root /path/to/long_term_forecast --profile-output experiments/results/remote_smoke --output experiments/results/remote_cost
```

`cost_estimate.json` 将训练、周期验证和最终评估分开外推，并按上下文长度/验证窗口数做粗略校正。规划区间不是统计置信区间；不含首次合成数据生成、绘图和排队时间。

## 本地冒烟与单数据集正式初筛

PowerShell：

```powershell
$python = 'D:\software\Anacoda\python.exe'
$dataRoot = 'D:\Code\TimeSeries\datasets\tslib-data\long_term_forecast'

# 5 模型 × 1 seed × 3 比例(0%、50%、100%)，各 50 steps，共 15 runs。
& $python -m experiments.mixed_training.run --stage smoke --data-root $dataRoot --output experiments/results/mixed_smoke_checked
& $python -m experiments.mixed_training.verify --output experiments/results/mixed_smoke_checked --expected-runs 15

# Exchange Rate：5 模型 × 3 seeds × 9 比例，共 135 runs。
& $python -m experiments.mixed_training.run --stage screen --data-root $dataRoot --output experiments/results/mixed_exchange
& $python -m experiments.mixed_training.verify --output experiments/results/mixed_exchange --expected-runs 135
& $python -m experiments.mixed_training.summarize --output experiments/results/mixed_exchange
```

覆盖参数示例：`--datasets etth1`、`--models mlp gru`、`--seeds 42`、`--fractions 0 0.125 0.5 1`、`--steps 500`。正式默认值集中在 `configs/default.json`。改变步数、数据生成设置或训练代码时使用新输出目录，防止混入不同协议的结果。

## 划分、采样与训练协议

- 真实数据按时间 70%/10%/20% 固定划分，再构造窗口。窗口全部目标属于相应区间；验证和测试允许读取起点前已经观测到的历史。测试为 frozen-weight rolling origin，不在测试段更新参数。
- 每个任务采用一个明确的目标列，所有模型与比例共享同一完整真实训练窗口池。
- 合成占比 `p=0,1/8,...,1`。batch 中有 `64*(1-p)` 个真实完整窗口、`64*p` 个合成完整窗口，仅打乱 batch 维度，不混合或打乱窗口内部的时间点。
- **本实现固定合成池**：默认原生成器生成 1,024 条独立序列，每条提供 4 个目标不重叠的窗口，共 4,096 个候选窗口；所有 p/模型/seeds 共享该池。比例控制实际曝光，不控制候选池大小。这样 p=1 也有有限、明确定义的样本池；不采用 `p/(1-p)` 扩池规则。
- 各来源按随机排列循环抽样，真实抽样流在不同 p 下共享前缀。保存每个窗口曝光次数。同一模型/seed 在所有 p 下初始权重一致。
- 合成目标与历史都使用观测值 `series_values * noise`，直接调用 `src/synthetic_generation/generate_series.py::generate`。日频、小时、周频分别使用原对应生成分支；无 transition/random walk。非有限合成序列最多重试 100 次并记录拒绝次数，不裁剪来伪装有效样本。
- 每个窗口只用历史计算均值和标准差，标准差下限 `1e-4`；未来目标使用同一变换。所有模型输出均为线性头，允许负值。
- 五模型：Linear；DLinear（均值分解，两条线性头）；两隐藏层宽 64 的 MLP；单层 hidden=32 GRU；d_model=32、4 heads、2 层、FFN=64 的 Transformer。模型均直接输出 H 步，无未来输入。
- Adam 1e-3、梯度范数裁剪 1.0、batch=64、2,000 updates。每 100 steps 遍历完整真实验证集；**按原尺度 validation RMSE 最小值选 best**，同时保留第 2,000 步 final；不提前停止。p=0/1 不对空来源求均值。
- 总 batch MSE 自然等价于 `(1-p)*mean_real_loss + p*mean_synthetic_loss`，不再额外重复加权。
- 每轮训练完成后才评估真实测试集，训练过程中不读取测试指标。p=1 仍用目标真实验证集选择 checkpoint，不能称为完全未接触目标域的零样本实验。

## 七个单变量任务

| 名称 | 文件相对路径 | 目标列 | 频率 / L / H | SeasonalNaive 周期 |
|---|---|---|---|---:|
| exchange | exchange_rate/exchange_rate.csv | 0 | 日 / 96 / 14 | 7 |
| etth1 | ETT-small/ETTh1.csv | OT | 小时 / 96 / 24 | 24 |
| etth2 | ETT-small/ETTh2.csv | OT | 小时 / 96 / 24 | 24 |
| electricity | electricity/electricity.csv | OT | 小时 / 96 / 24 | 24 |
| traffic | traffic/traffic.csv | OT | 小时 / 96 / 24 | 24 |
| weather | weather/weather.csv | T (degC) | 完整小时均值 / 96 / 24 | 24 |
| illness | illness/national_illness.csv | OT | 周 / 104 / 14 | 52 |

这是指定单变量任务，不等同于原论文的聚合数据或常见多变量 leaderboard。周期预先指定，不按测试结果选择。

Weather 的本地文件有相同时间/目标值的重复记录与内部缺测：配置显式允许删除完全相同的时间–目标记录，对 10 分钟网格的缺测仅前向填充以提供因果历史。然后计算完整小时均值，时间标签是该小时结束时刻；边缘不足 6 条的小时丢弃。**任何目标窗口只要包含填充影响的小时就排除**，MASE 分母也排除对应缺测配对；时间轴不因内部缺测而压缩。清洗计数写入 manifest。其他数据集不默认启用这种规则，有重复/缺测/频率不匹配就明确报错。

## 输出、图表与复核

每个 `<dataset>/<model>/p_<fraction>/seed_<seed>/` 保存：

- `config.json`、`split_manifest.json`：运行设置、源码摘要、数据校验值、时间边界、目标列和窗口数量。
- `train_log.csv`：mixed/real/synthetic 训练 MSE、真实验证 MAE/RMSE、更新步数和两来源累计曝光。
- `best.pt`、`final.pt`：验证最优与最终权重。
- `metrics.json`：best/final 的验证与测试 MAE、RMSE、MSE、MASE、各 horizon 指标，以及参数量、时间、最优步数、实际曝光量。
- `predictions.npz`：全部真实测试预测起点、目标、best/final 预测；`exposures.npz` 保存逐样本曝光计数。

`summarize` 输出 `report/`：

- `runs.csv`：所有模型、比例、seed、checkpoint 的逐次结果。
- `aggregate.csv`：各比例跨 seed 的均值、标准差及实际完成数。
- `validation_selected.csv`：每模型/每checkpoint **按跨 seed 验证 RMSE** 选择一个比例后，报告其测试误差及相对 real-only 的改善。
- 每个数据集的比例曲线（MAE/RMSE × best/final）、模型对比图、训练曲线、固定首个测试起点的预测示例，以及原生成器采样图。
- `report.md`：便于阅读的图文报告。标准差不是置信区间；不直接平均不同量纲数据集的原始 RMSE。

`verify` 独立从保存的预测重算误差、检查相同比例组的初始化配对、曝光总量、完整测试窗口和验证选择规则。`tests` 检查时间边界、只依赖历史的缩放、缺测目标排除、配额、五模型梯度和 baseline。

## 中断与重跑

`metrics.json` 是最后写入的完成标记。重复运行相同命令会跳过已完成且签名一致的任务；中途失败的任务从同一 seed 重新训练，不声称支持 optimizer 中途续训。源码、设备或配置不一致时拒绝复用已有完成结果。

可按数据集/模型/seed 拆分命令顺序执行；多设备并行时建议每个进程使用独立输出目录，避免共享 cache 或元数据竞争。每份报告会注明实际完成数，不能把部分网格误认为完整实验。

原始数据不复制到仓库；合成缓存、预测数组、权重被 Git 忽略。本地 results 的轻量指标和图表可以保留，但远程运行会重新生成缓存；不依赖本机 `.experiment_deps` 或绝对路径。
