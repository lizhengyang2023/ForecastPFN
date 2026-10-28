# 真实—模拟 Mixup 实验

本目录实现 [`../real_synthetic_mixup_plan.md`](../real_synthetic_mixup_plan.md) 的第一版：单变量真实窗口分别与同任务真实窗口或 ForecastPFN 模拟窗口逐点线性组合。数据切分、模拟池、五个小模型和原尺度评估复用 `experiments/utils/`；本目录保留 Mixup 专用采样、训练、核验与配置。运行命令均在 **ForecastPFN 仓库根目录**执行。

## 环境与数据

推荐 Python 3.11，安装 `experiments/sample_mixed/requirements.txt` 中的依赖；本地示例使用已有 Conda Python。`--data-root` 指向包含 `exchange_rate/`、`ETT-small/` 等目录的 `long_term_forecast`。默认自动选择 CUDA，亦可用 `--device cpu`。

```powershell
$python = 'D:\software\Anacoda\python.exe'
$dataRoot = 'D:\Code\TimeSeries\datasets\tslib-data\long_term_forecast'
& $python -m pip install -r experiments/sample_mixed/requirements.txt
```

## 1. 确定性测试

检查真实锚点在三组中共享同一抽样流、两种 Mixup 共享增强位置、真实伙伴不与自身相同、历史与目标使用相同权重，以及同一配置和 seed 的完整训练重放得到相同预测。

```powershell
& $python -m experiments.mixup.tests
```

## 2. 可选：只准备数据

检查真实时间切分并缓存原 ForecastPFN 模拟序列，不训练模型。默认是 Exchange Rate；`--stage full` 默认七个既定单变量任务。

```powershell
& $python -m experiments.mixup.run --stage prepare --data-root $dataRoot --output experiments/results/mixup_prepare
```

## 3. 冒烟训练及核验

默认 Exchange Rate，Linear/GRU、seed 42；每模型训练 Real-only、Real–Real (`λ=0.75`)、Real–Synthetic (`λ=0.75`) 三组，每组 50 updates，共 **6 runs**。`--synthetic-series 32` 仅供快速冒烟，正式实验仍使用默认 1,024 条合成序列。运行结束后从保存的测试预测重算指标，并检查曝光数、时间边界与同模型同 seed 的初始权重。

```powershell
& $python -m experiments.mixup.run --stage smoke --data-root $dataRoot --output experiments/results/mixup_smoke --synthetic-series 32
& $python -m experiments.mixup.verify --output experiments/results/mixup_smoke --expected-runs 6
```

## 汇总实验结果

训练完成并运行 `verify` 后，使用独立的 Mixup 汇总入口；它读取 `<dataset>/<model>/<mode>/lambda_<weight>/seed_<seed>/metrics.json`，不适用 `sample_mixed.summarize` 的合成占比目录结构。冒烟和正式实验都可以使用同一接口：

```powershell
& $python -m experiments.mixup.summarize --output experiments/results/mixup_smoke
& $python -m experiments.mixup.summarize --output experiments/results/mixup_exchange
```

输出写入对应目录的 `report/`：`report.md` 为图文总结，`runs.csv` 为逐 seed 的 best/final 指标，`aggregate.csv` 为跨 seed 均值、标准差及数量，`paired.csv` 给出同模型、同 seed 相对 Real-only 和同权重 Real–Real 的差值，`validation_selected.csv` 给出按真实验证 RMSE 选出的各模式权重，另有各模型的测试 RMSE 图。只用与 Real-only 完全相同的 seed 集合参与配置选择；缺组时仍汇总已有运行，但不会将不完整组当作可比较的优胜配置。测试集不用于选权重。

汇总逻辑的确定性选择测试：

```powershell
& $python -m experiments.mixup.test_summarize
```

重构复核已在本地完成：6 次冒烟均通过 `verify`，与重构前保存的同配置冒烟结果比较时，checkpoint 指标、曝光统计及 42 个预测/采样数组完全一致。可按以下命令重复核对（旧结果目录仅在本机存在）：

```powershell
& $python -m experiments.mixup.verify --output experiments/results/refactor_mixup_smoke --expected-runs 6
& $python -m experiments.compare_refactor --old experiments/results/mixup_smoke --new experiments/results/refactor_mixup_smoke --expected-runs 6
```

## 4. Exchange 正式初筛

五模型、三个 seed、`λ={0.9,0.75,0.5}`；每模型/seed 的 Real-only 只训练一次，加两种 Mixup 各三个权重，共 **105 runs**。每组 2,000 updates、batch=64、`p_aug=0.5`。使用新的输出目录，勿与冒烟结果混放。完整运行结束再统一查看测试结果。

```powershell
& $python -m experiments.mixup.run --stage screen --data-root $dataRoot --output experiments/results/mixup_exchange
& $python -m experiments.mixup.verify --output experiments/results/mixup_exchange --expected-runs 105
```

七数据集完整网格共 735 runs；可用 `--datasets etth1` 等参数分批运行，完整默认命令如下：

```powershell
& $python -m experiments.mixup.run --stage full --data-root $dataRoot --output experiments/results/mixup_full
& $python -m experiments.mixup.verify --output experiments/results/mixup_full --expected-runs 735
```

可选覆盖：`--models linear gru`、`--seeds 42`、`--weights 0.75`、`--modes real real_synthetic`、`--steps 500`、`--datasets exchange etth1`。改变步数、生成池大小、代码或配置时使用**新输出目录**；运行签名会拒绝复用不一致的已完成任务。

## 方法与输出

每步先从真实训练段抽取 64 个锚点，确定性选出 32 个增强位置。Real-only 不增强；Real–Real 伙伴来自同一真实训练池；Real–Synthetic 伙伴来自固定模拟池。每个窗口分别用其**历史**均值/标准差归一化，然后同一 `λ` 融合完整历史和目标。验证与测试只使用真实数据，按原尺度验证 RMSE 选 `best`，同时保存 `final`。测试预测按真实历史的统计量反变换。仅更新步数与真实锚点曝光量固定；增强组的原样真实曝光较少。

每个 `<dataset>/<model>/<mode>/lambda_<weight>/seed_<seed>/` 保存 `config.json`、`split_manifest.json`、`train_log.csv`、`best.pt`、`final.pt`、`metrics.json`、`predictions.npz`、`exposures.npz`。数据集目录保存真实切分说明、模拟池说明和免训练基线；根目录 `experiment.json` 记录运行环境。`metrics.json` 最后写入，表示该次运行完成。中断后重复同一命令会跳过签名一致的完成项；未完成项从同 seed 重新训练。

冒烟仅验证流程；不依据其测试指标选择模型或 Mixup 权重。真实–真实 Mixup 是判断线性组合本身作用的必要对照；与已有样本级混合结果比较时须注明后者真实窗口曝光量随合成占比变化。
