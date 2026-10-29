# 统一的受控 Mixup 实验

本目录统一原 `mixup` 与 `mixup_paired` 的运行入口。**新协议**包含五种任务：`real`、`real_real_phase_match`、`real_real_phase_mismatch`、`real_synthetic_phase_match`、`real_synthetic_phase_mismatch`。旧随机配对及带回退的 `period_only` / `period_phase` 结果保留在原输出目录，只作历史参考；新协议须使用新输出目录，不能混合汇总。

## 对照定义

所有任务使用同一真实训练锚点池：锚点历史的周期估计可靠，而且在 Real–Real 和 Real–Synthetic 两个来源中均能找到相位匹配与相位不匹配的伙伴。候选特征只取历史 `X`，不读取未来 `Y`。两组伙伴均须满足相同频率、周期容差和预测跨度漂移限制；匹配组要求圆周相位距离 `≤ π/4`，不匹配组要求 `> π/4`。两组分别从最多 8 个最接近 0 或 π 的候选中按固定 seed 抽取。Real–Real 排除自身及相距不足 `L+H` 的重叠窗口。**没有随机回退，也没有周期不匹配的伙伴。**

共同锚点池会缩小训练数据；`real` 也在此池上重训，因此它与旧 Real-only 的数值不应直接视作同一基线。`<output>/<dataset>/support.json` 记录可靠率、各类候选覆盖和共同锚点比例；如果共同池为空，准备阶段明确失败，不放宽阈值。所有任务共享相同 seed 的模型初始化、锚点采样流和增强位置流。每个 batch 默认 64 个锚点、其中 32 个位置增强；λ 同时作用于标准化后的历史与目标。每个 run 训练 2,000 updates，每 100 步按真实验证集原尺度 RMSE 选择 `best`，完成后评估真实测试集。

## 四卡运行

在仓库根目录执行；先运行测试：

```bash
python -m unittest discover -s experiments/mixup/tests_controlled -v
```

ETTh1 正式筛选：

```bash
python -m experiments.mixup.launch --stage screen --datasets etth1 \
  --data-root /path/to/long_term_forecast \
  --output experiments/results/mixup_controlled_etth1 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

每个可运行数据集为 `5 模型 × 3 seed × (1 Real-only + 4 任务 × 3 λ) = 195 runs`。先运行 ETTh1 并检查 `support.json` 的覆盖率，再决定是否扩展。**本地只准备检查**（合成序列 32 条）显示 ETTh1、ETTh2、Electricity、Traffic、Weather 有共同锚点；Exchange 仅 131/5,202（2.5%），Illness 为零，因其可靠真实周期窗口数为零。正式默认池为 1,024 条合成序列，其余任务覆盖率仍须重新检查；增加合成序列不会改变 Illness 的真实周期可靠率。`--stage full` 默认选择上述五个数据集，共 975 runs；**本轮正式测试与跨数据集汇总排除 Exchange 和 Illness**，将其弱周期问题留待后续独立研究。完整试验计划见 [周期与相位受控 Mixup 计划](../mixup_paired/real_synthetic_mixup_paired.md)。如显式添加其他数据集，若共同池为空会明确失败并保存 `support.json`，不会回退或伪造结果。

五数据集全量实验（在仓库根目录执行；按实际位置修改 `--data-root`）：

```bash
python -m experiments.mixup.launch --stage full \
  --datasets etth1 etth2 electricity traffic weather \
  --data-root /mnt/raid5/lizy/tslib-data/long_term_forecast \
  --output experiments/results/mixup_controlled_full \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

`launch` 在 975 个 run 全部完成后自动执行核验与汇总，报告位于 `experiments/results/mixup_controlled_full/report/`。如需对**已完成的同一输出目录**重新核验或生成报告，可单独运行：

```bash
python -m experiments.mixup.verify \
  --output experiments/results/mixup_controlled_full --expected-runs 975
python -m experiments.mixup.summarize \
  --output experiments/results/mixup_controlled_full
```

`launch` 顺序准备共享合成数据和共同伙伴池，然后按稳定任务序号分片。分片单位是 `(数据集, 模型, seed, λ, 任务)`，每个完整 run 固定在一张 GPU 上；不会拆分单次训练的 batch。每个 worker 独占 `<output>/logs/worker_<id>.log`，日志采用追加模式并含 UTC 启动时间；各 run 独占 `<output>/<dataset>/<model>/<task>/lambda_<λ>/seed_<seed>/`。共享 `assets/` 只在准备阶段写入，worker 只读。启动器等待全部 worker 结束后才统一 `verify` 和 `summarize`。失败时可用**完全相同的命令和输出目录**重跑：已完成的 `metrics.json` 会跳过，未完成 run 从原 seed 重新训练。源码或配置改变时必须使用新输出目录。

只占用一张卡时可用 `--devices cuda:0`；CPU 流程检查可用 `--devices cpu` 并覆盖模型、seed、权重和步数。也可分别调用 `experiments.mixup.run --stage prepare` 与带 `--num-shards N --shard-id I --device cuda:I` 的训练命令，但所有 worker 必须采用完全相同的任务网格参数。

## 结果

统一输出 `report/runs.csv`、`aggregate.csv`、`validation_selected.csv`、`paired_differences.csv`、`report.md`。模型 checkpoint 与 λ 只按验证 RMSE 选择；测试 MAE、RMSE、MASE 仅用于报告。同模型、同 seed、同 λ 的匹配／不匹配对照可从 `paired_differences.csv` 读取。`verify` 检查完整网格、测试预测指标、来源曝光与相同 seed 的初始权重。

汇总还会为每个数据集绘制一张不同模型的 MASE 相对 Real 百分比折线图，横轴为四种受控任务，点上标注按验证 RMSE 选出的最优 λ。变化比例先按相同模型与 seed 计算 `100 × (Mixup 测试 MASE / Real 测试 MASE − 1)`，再跨 seed 求均值；**负值表示改善**。图和作图数据保存在 `report/<dataset>_mase_delta_vs_real.png`、同名 SVG、`report/mase_delta_vs_real.csv`，图集入口为 `report/mase_delta.md`。仅下载了 `report/runs.csv` 和 `report/validation_selected.csv` 时，也能在仓库根目录单独重绘：

```bash
python -m experiments.mixup.plot_mase_delta \
  --output experiments/results/mixup_controlled_full
```

[`report/mase_delta.md`](../results/mixup_controlled_full/report/mase_delta.md) 同时列出按验证集选择 λ 后的任务排名，以及分别固定 λ=0.5、0.75、0.9 时 Real 与四种混合任务的三张排名表。每表包含第一名次数、距离第一名 ≤0.5% 的次数、优于 Real 的次数与增益中位数；Real 基线在三个 λ 下相同。第一名是**描述性排名**，接近第一名的阈值仅用于敏感性检查，不能据此判定显著性。逐组合结果保存在 `report/mase_task_rankings.csv` 和 `report/mase_task_fixed_lambda_rankings.csv`，对应汇总保存在同名前缀的 `*_summary.csv`。

要查看所有 λ，而非只看验证集选中的 λ，运行：

```bash
python -m experiments.mixup.plot_mase_delta \
  --output experiments/results/mixup_controlled_full --all-lambdas
```

这会生成 [`report/mase_delta_all_lambdas.md`](../results/mixup_controlled_full/report/mase_delta_all_lambdas.md) 及每数据集的 PNG/SVG：颜色区分模型，圆点/方点/三角形依次表示 λ=0.5/0.75/0.9；实线连接验证集选中的点，半透明色带连接三个 λ 的最小与最大增益。色带是**参数敏感性范围，不是置信区间**；类别之间的连线也不表示连续变化。对应数据、三个 λ 的增益跨度和固定同一 λ 的相位匹配优势分别保存在 `report/mase_delta_all_lambdas.csv`、`report/mase_lambda_sensitivity.csv`、`report/mase_phase_same_lambda_summary.csv`。

原 `python -m experiments.mixup_paired.launch`、`run`、`verify` 和 `summarize` 命令名转发到统一实现；旧 `--mode`、`--strategy` 和 `--random-output` 参数不再属于新协议。
