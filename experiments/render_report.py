"""Build a self-contained report and canvas from measured experiment outputs."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'results/quick_validation_100'
summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
short = json.loads((OUT.parent/'quick_validation/summary.json').read_text(encoding='utf-8'))
loss = list(csv.DictReader((OUT/'loss.csv').open()))
rows = list(csv.DictReader((OUT/'sample_examples.csv').open()))
examples = []
for sid in dict.fromkeys(r['series_id'] for r in rows):
    series = [r for r in rows if r['series_id']==sid]
    examples.append(dict(id=sid,clean=[float(r['clean']) for r in series],
        observed=[float(r['observed']) for r in series],
        forecast=[float(r['ForecastPFN-small']) for r in series[-10:]],
        seasonal=[float(r['SeasonalNaive-7']) for r in series[-10:]]))
data = dict(summary=summary, short=short, loss=[{k:float(v) if v else None for k,v in r.items()} for r in loss],examples=examples)
template = '''import {H1,H2,Text,Stack,Row,Stat,LineChart,Table,Select,useState,useHostTheme} from "cursor/canvas";
const data = __DATA__;
export default function Report() {
 const theme=useHostTheme();
 const [target,setTarget]=useState("clean");
 const [sample,setSample]=useState("0");
 const s=data.summary;
 const metrics=s.metrics.filter(r=>r.target===target);
 const example=data.examples[Number(sample)];
 return <main style={{background:theme.bg.editor,color:theme.text.primary,padding:24,maxWidth:1100,margin:"auto"}}>
 <Stack gap={20}>
 <H1>ForecastPFN 小规模验证</H1>
 <Text>2026-10-26 · 原仓库生成器与模型模块 · CPU · 随机种子 42 · 独立序列划分</Text>
 <Text>本页报告实际运行数据；这是一轮合成分布内实验，不能据此推断论文模型在真实数据上的效果。</Text>
 <Row gap={28}><Stat label="可训练参数" value={s.parameters.toLocaleString()}/><Stat label="训练 + 验证耗时" value={(s.training_seconds/60).toFixed(2)+" 分钟"}/><Stat label="最佳验证 epoch" value={s.best_epoch}/></Row>
 <H2>训练预算对比</H2>
 <Table headers={["预算","最佳验证 NMSE","测试 clean NMSE","训练及验证秒数"]} rows={[data.short,s].map(r=>[r.epochs+" epochs",r.best_val_loss.toFixed(5),r.metrics.find(m=>m.target==="clean"&&m.model==="ForecastPFN-small")!.nmse.toFixed(5),r.training_seconds.toFixed(1)])}/>
 <Text>先完成 25 epochs；观察到欠拟合后，补充固定 100 epochs，从同一初始化重新训练。两轮均只用验证集选 checkpoint；属于探索性预算比较，测试集没有重新抽样。</Text>
 <H2>学习曲线：scale-normalized MSE</H2>
 <Text>横轴：epoch；纵轴：无量纲 MSE / context scale²。训练为抽样 batch 均值，验证为固定 128 条独立序列 × 10 步。</Text>
 <LineChart categories={data.loss.map(r=>String(r.epoch))} series={[{name:"Validation",data:data.loss.map(r=>r.val_loss)}]} height={230}/>
 <LineChart categories={data.loss.slice(1).map(r=>String(r.epoch))} series={[{name:"Train",data:data.loss.slice(1).map(r=>r.train_loss as number)}]} height={180}/>
 <Text>来源：loss.csv；epoch 0 为训练前验证。上下图纵轴分别自动缩放。</Text>
 <H2>独立测试集：免训练 baseline 对比</H2>
 <Select value={target} onChange={setTarget} options={[{value:"clean",label:"Clean：无噪信号（训练目标）"},{value:"observed",label:"Observed：带噪观测"}]}/>
 <Table headers={["方法","NMSE ↓","MSE ↓","MAE ↓","sMAPE % ↓"]} rows={metrics.map(r=>[r.model,r.nmse.toFixed(5),r.mse.toFixed(5),r.mae.toFixed(5),r.smape.toFixed(2)])}/>
 <Text>来源：metrics.csv；256 条未见测试序列，每条预测最后 10 天。NMSE 使用相同历史 robust scale，不使用测试目标估计归一化参数。</Text>
 <Text>ForecastPFN − seasonal naive 的 NMSE 差（负数更好）：{s.comparison[target as "clean"|"observed"].nmse_difference.toFixed(5)}；按序列配对 bootstrap 95% 区间：{s.comparison[target as "clean"|"observed"].paired_bootstrap_95ci.map(v=>v.toFixed(5)).join(" 至 ")}；逐序列胜率：{(100*s.comparison[target as "clean"|"observed"].series_win_rate).toFixed(1)}%。</Text>
 <H2>采样与预测示例</H2>
 <Select value={sample} onChange={setSample} options={data.examples.map((e,i)=>({value:String(i),label:"测试序列 "+i+"（全局 ID "+e.id+"）"}))}/>
 <Text>横轴：日序号；纵轴：原始合成值。展示固定前六条测试序列，没有按效果挑选。</Text>
 <LineChart categories={example.clean.map((_,i)=>String(i))} series={[{name:"Clean signal",data:example.clean},{name:"Noisy observation",data:example.observed}]} height={240} beginAtZero={false}/>
 <Text>上图为全长 64 天；模型只读取第 18–53 天，预测第 54–63 天。下图只展示这 10 个未来点。</Text>
 <LineChart categories={Array.from({length:10},(_,i)=>String(54+i))} series={[{name:"Clean target",data:example.clean.slice(-10)},{name:"ForecastPFN-small",data:example.forecast},{name:"SeasonalNaive-7",data:example.seasonal}]} height={240} beginAtZero={false}/>
 <Text>来源：sample_examples.csv。生成信号含趋势、日历季节性以及乘性 Weibull 噪声。</Text>
 <H2>设置与解释边界</H2>
 <Text>768 / 128 / 256 条训练 / 验证 / 测试日频序列；每条 64 天，36 天历史，10 天预测。两层原始 TransformerBlock：2 heads × key_dim 18；Adam 0.001；100 epochs × 24 steps × batch 32；仅 point task。最佳 checkpoint 只按验证 loss 选择。</Text>
 <Text>保留原 robust scaler、日历编码、task embedding、GELU 与 ReLU 输出。负 clean 值占比 {(100*s.negative_clean_fraction).toFixed(2)}%；ReLU 对负目标存在表达限制。依赖安装时间不计入训练时间；TensorFlow {s.tensorflow}，Python {s.python}。</Text>
 <Text>本实验只有一个训练 seed、日频合成序列、一个测试预测起点；bootstrap 仅反映测试序列抽样不确定性。后续应增加 seed、训练预算，并加入真实数据与多频率验证。</Text>
 </Stack></main>;
}
'''
(OUT/'ForecastPFN-validation.canvas.tsx').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False)),encoding='utf-8')
clean = {r['model']:r for r in summary['metrics'] if r['target']=='clean'}
observed = {r['model']:r for r in summary['metrics'] if r['target']=='observed'}
delta = 100*(1-clean['ForecastPFN-small']['nmse']/clean['SeasonalNaive-7']['nmse'])
report = f'''# ForecastPFN 小规模验证报告

日期：2026-10-26。环境：conda base Python {summary['python']}，工作区独立 TensorFlow {summary['tensorflow']}，CPU。

## 实验结果

训练正常完成。共 {summary['parameters']:,} 参数；训练及逐 epoch 验证用时 {summary['training_seconds']:.1f} 秒。
验证 NMSE 从 {summary['initial_val_loss']:.6f} 降到 {summary['best_val_loss']:.6f}，最佳 epoch 为 {summary['best_epoch']}。
训练首/末 epoch loss：{float(loss[1]['train_loss']):.6f} / {float(loss[-1]['train_loss']):.6f}。

无噪测试目标：ForecastPFN-small NMSE = {clean['ForecastPFN-small']['nmse']:.6f}，
seasonal naive = {clean['SeasonalNaive-7']['nmse']:.6f}，相对改善 {delta:.2f}%（负数表示退步）。
两者 NMSE 差的按序列配对 bootstrap 95% 区间为 {summary['comparison']['clean']['paired_bootstrap_95ci']}。
ForecastPFN 逐序列胜率为 {summary['comparison']['clean']['series_win_rate']:.1%}。

带噪测试目标：ForecastPFN-small NMSE = {observed['ForecastPFN-small']['nmse']:.6f}，
seasonal naive = {observed['SeasonalNaive-7']['nmse']:.6f}。
所有方法的 MSE、MAE、NMSE、sMAPE 均保存在 metrics.csv，详细交互表见同目录 canvas。

## 结果解读

100-epoch 小模型的 clean NMSE 比 25-epoch 下降约 20.9%，也低于 ContextMean 的 0.160543、
LastValue 的 0.201991、Drift 的 0.212238；但仍是 seasonal naive 的约 3.25 倍。
配对差值置信区间完全大于零，当前样本上 seasonal naive 的优势明确。
带噪观测目标下结论一致，说明差距不只是 clean/noisy 评估口径造成的。

采样图中可见趋势、重复的每周峰谷与幅度不等的乘性噪声。小模型在固定展示的六条测试序列上
主要给出平滑的水平预测，缺乏清晰的周内峰谷；seasonal naive 直接复用上周形状，
与这批强周季节性日频数据匹配。loss 降低说明模型确实在学习，但尚不等于学会季节外推。
缩小模型、短训练预算与单一 point task 都可能影响效果，本实验不能单独识别各因素的因果贡献。
负值/ReLU 不匹配是保留的原始限制，不能据此把全部误差归因于负值。

下一步建议先在相同序列划分下恢复原 attention 宽度，做多 seed 和训练预算对照，
再加入周频、月频以及真实数据；保留 seasonal naive 作为必须比较的强 baseline。

![Learning curve](loss.png)

![Baseline comparison](baselines.png)

![First six test series](samples.png)

## 方法与限制

使用原仓库 generate、TransformerModel、TransformerBlock、robust_scaler。
缩小为两层 2-head / key_dim 18 attention，保留其余模型逻辑；只训练点预测任务。
768 / 128 / 256 条独立日频序列；每条长度 64；输入 36，预测 10。
首轮训练 25 epochs × 24 steps × batch 32 = 19,200 次抽样任务；补充轮 100 epochs = 76,800 次任务；单随机种子 42。
首轮最佳验证 NMSE {short['best_val_loss']:.6f}，测试 clean NMSE {short['metrics'][0]['nmse']:.6f}，训练及验证 {short['training_seconds']:.1f} 秒。
观察到首轮欠拟合后增加固定预算，两轮均从同一初始化开始，且只按验证集选 checkpoint；这是探索性比较，沿用同一测试集。
验证和测试只取每条序列最后 10 点，checkpoint 只按验证 NMSE 选择。
所有 baseline 读取同样的带噪历史；模型训练目标是无噪信号，因此额外报告带噪观测指标。
NMSE 定义为 mean(error² / context_robust_scale²)，不是按方差归一化。

原生成数据中负 clean 值占 {summary['negative_clean_fraction']:.2%}；原模型输出为 ReLU，不能表达负预测。
该限制被保留。未使用真实数据、预训练权重或论文的完整规模；不能将结果当作论文性能复现。
Bootstrap 以整条序列为单位，不能代表训练 seed 的波动。
256 条序列 × 10 步的批量推理用时 {summary['inference_seconds']:.2f} 秒。

复现命令与细节见 ../../README.md；完整数据与预测分别保存为 samples.npz、predictions.npz。
'''
(OUT/'report.md').write_text(report,encoding='utf-8')
print(OUT/'report.md')
