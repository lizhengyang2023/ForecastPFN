# 周期与相位配对 Mixup 实验

实验协议见 [`real_synthetic_mixup_paired.md`](real_synthetic_mixup_paired.md)。本目录在已完成的随机 Mixup 基础上实现四个新组合：`real_real` / `real_synthetic` × `period_only` / `period_phase`。每个增强位置仍是一个真实训练锚点与一个伙伴的凸组合；仅改变伙伴选择，保持 `λ`、`p_aug`、模型、训练预算与真实验证选择规则。所有命令在 **ForecastPFN 仓库根目录**执行。

## 1. 环境与确定性测试

沿用 [`../mixup/README.md`](../mixup/README.md) 的 Python 环境和数据集根目录。程序需要现有的 NumPy、SciPy、PyTorch、pandas 和 Matplotlib；`scipy.spatial.cKDTree` 用于周期/相位邻居检索。

```powershell
$python = 'D:\software\Anacoda\python.exe'
$dataRoot = 'D:\Code\TimeSeries\datasets\tslib-data\long_term_forecast'
& $python -m experiments.mixup_paired.tests
```

测试覆盖周期/相位估计、反相与无周期回退、真实窗口自身排除、共享锚点和增强位置，以及固定 seed 的完整训练重放。周期特征只读取训练窗口的历史 `X`，不会读取未来 `Y`；模拟池固定且与现有 Mixup 使用相同生成器配置。

## 2. 单机冒烟与自动报告

以下命令用四个 CPU 进程完成 ETTh1 的四种组合，每组合只训练 Linear、seed 42、`λ=0.75`、2 updates。它会先生成小型共享模拟池，再启动四进程，最后自动执行 `verify` 和 `summarize`。该目录只用于流程检查，不用于选择方法。

```powershell
& $python -m experiments.mixup_paired.launch --stage smoke --datasets etth1 `
  --data-root $dataRoot --output experiments/results/mixup_paired_smoke `
  --models linear --steps 2 --synthetic-series 8 `
  --devices cpu cpu cpu cpu
```

四个日志位于 `<output>/logs/`，每个组合自己的训练文件位于 `<output>/<mode>_<strategy>/`。若要按计划执行 50 updates、Linear/GRU 的完整冒烟，去掉 `--models linear --steps 2` 并将 `--synthetic-series` 设为 32；预期 8 runs。

## 3. 四卡正式初筛

默认 `--stage screen` 先运行 ETTh1。每张卡负责一种组合，五模型 × 三 seed × 三权重 = **45 runs/卡**，四卡合计 **180 runs**。默认设备依次为 `cuda:0` 至 `cuda:3`，可用 `--devices` 显式调整。新运行与旧随机 Mixup 的完整结果在报告中并列，但仅在模型、seed 和训练预算相同时计算对照差值。

```powershell
& $python -m experiments.mixup_paired.launch --stage screen --datasets etth1 `
  --data-root $dataRoot --output experiments/results/mixup_paired_etth1 `
  --random-output experiments/results/mixup_full
```

四张卡必须在同一机器上由 CUDA 可见。启动脚本先顺序准备共享模拟池，随后给四个子进程分配不同 GPU 和不同输出目录；任一子进程失败时保留日志，重新执行相同命令可跳过签名一致的已完成运行。`--random-output` 可以指向旧随机实验的原始结果目录，也可以指向只保留 `report/runs.csv` 的归档目录；后者的旧数据/生成池签名无法从 CSV 独立核验，解释对照时应查看原始实验记录。

## 4. 七数据集全量运行

全量为七任务 × 四组合 × 五模型 × 三 seed × 三权重 = **1,260 个新增 runs**，每张卡 315 runs。建议先查看 ETTh1 的实际相位匹配率和验证结果，再决定是否运行全部任务。完整命令：

```powershell
& $python -m experiments.mixup_paired.launch --stage full `
  --data-root $dataRoot --output experiments/results/mixup_paired_full `
  --random-output experiments/results/mixup_full
```

可用 `--datasets etth1 etth2` 分批，每批使用独立输出目录；最后按批汇总。`--steps`、模型、seed、权重、模拟池大小或匹配配置改变时使用新的输出目录。现有随机 Mixup 的全量结果不会重训，也不会写入配对运行目录。

## 5. 手动运行、核验与汇总

如果只运行一个组合，先准备共享资产，再指定模式、策略和设备。以下为 ETTh1 的模拟伙伴加相位配对：

```powershell
& $python -m experiments.mixup_paired.run --stage prepare --datasets etth1 `
  --data-root $dataRoot --output experiments/results/mixup_paired_assets
& $python -m experiments.mixup_paired.run --stage screen --datasets etth1 `
  --data-root $dataRoot --output experiments/results/mixup_paired_manual/real_synthetic_period_phase `
  --assets experiments/results/mixup_paired_assets/assets `
  --mode real_synthetic --strategy period_phase --device cuda:0
& $python -m experiments.mixup_paired.verify `
  --output experiments/results/mixup_paired_manual --expected-runs 45
& $python -m experiments.mixup_paired.summarize `
  --output experiments/results/mixup_paired_manual `
  --random-output experiments/results/mixup_full
```

每次运行保存 `config.json`、`split_manifest.json`、`train_log.csv`、`best.pt`、`final.pt`、`metrics.json`、`predictions.npz` 和 `exposures.npz`。`metrics.json` 中的 `matching` 记录真实相位配对率、仅周期配对率、随机回退率、候选数、周期/相位距离、未来相位漂移、分箱分布与唯一伙伴数；特征缓存和可靠周期占比分别在 `<cell>/features/` 与 `<cell>/<dataset>/feature_diagnostics.json`。

汇总写入 `<output>/report/`：`report.md` 为验证选择和同权重对照，`runs.csv` 为逐 seed 指标，`aggregate.csv` 为跨 seed 汇总，`validation_selected.csv` 为验证选出的权重，`paired_differences.csv` 为同模型、seed、权重下相对旧基线的测试差值，`compact_mase_by_ratio.csv` 与每任务的 `*_compact_mase.png` 展示各伙伴权重的测试 MASE；`*_matching_rates.png` 展示实际配对与回退占比。**测试指标只用于报告；模型与权重由真实验证 RMSE 选择。**若配对运行大多回退，报告中的预测差异不能归因于相位约束。
