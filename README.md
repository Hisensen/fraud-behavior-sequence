# 行为序列反欺诈 — MEM 异常检测

无监督的金融反欺诈方案：把用户行为事件序列编码后，只用白样本训练
Masked Event Model / 自回归模型学"正常行为语法"，用逐位置预测惊讶度
（交叉熵 → 难度归一化 → 池化）作为异常分。**零黑标签冷启动**，
支持逐笔可疑操作定位。

## 核心结果

| 数据集 | 性质 | 无监督 AUC | 监督 oracle 上限 |
|---|---|---|---|
| rich 合成基准（操作流） | 合成 | **0.984**（AR+时间偏置） | ~0.99 |
| temporal 合成（操作流） | 合成 | 0.959 | 0.999 |
| Ethereum 钓鱼（资金流） | ✅ 真实 | **0.928**（+对手方特征融合） | 0.934 |
| Sparkov（消费流） | 拟真 | 0.886（超过 oracle） | 0.880 |
| TabFormer（消费流） | 拟真 | 0.752（打平天花板） | 0.760 |
| AML（转账） | 拟真 | 0.712（超过 oracle） | 0.695 |
| BankSim / PaySim | 拟真 | 0.736 / 0.561 | 0.891 / 0.610 |

规律：MEM 无监督分数稳定逼近"用全部标签"的 oracle（差 0.03~0.07）；
绝对分高低由数据信号强度决定。

## 报告（按阅读顺序）

1. `EXPERIMENT_MEM.md` — 方法主实验：MEM 设计、熵混淆大坑与修复
2. `VALIDATION_REPORT.md` — 污染鲁棒性 / 少量标签用法 / 首个真实数据验证
3. `OPEN_DATA_SURVEY.md` — 10 个开源反欺诈数据集可行性普查 + 6 个实测
4. `ENCODING_REPORT.md` — 编码消融：统一 Schema + 个人金额基线（最大单项增益）
5. `ARCH_REPORT.md` — 架构探索：AR+时间偏置；按数据形状的三档定稿配置
6. `IMPROVEMENT_REPORT.md` — 提升轮：字段遮罩迁移 / 自清洗(负结果) / 对手方特征与分数融合
7. **`FINAL_SOLUTION.md` — 定稿方案：五层系统架构 + 分阶段落地路线（读这篇就够）**

## 展示页面

本地（docs/，自包含 HTML，浏览器直接打开）：

- **`tutorial.html` — 零基础完整教程（写给完全不了解方案的人：手算例子+全部实验证明，给新人看这篇）**
- `method-explained.html` — 无监督方案详解（每步配真实例子）
- **`tracer.html` — 交互式全程追踪器（选真实账户、点任意事件格子看七站完整记录）**
- `model-io.html` — 模型输入输出维度可视化（点任意事件列看编号→向量→概率的变形全程）
- `datasets.html` — 实验数据档案馆（8个数据集的字段/原文样例/处理链路/结果）
- `pipeline-walkthrough.html` — 处理流程逐步拆解（每步实算演示）
- `results-explained.html` — 结果解读：交互阈值滑块 + AUC 直观化 + oracle 对比

在线（Claude Artifact）：

- [方案全景：五步流水线 + 端到端实例](https://claude.ai/code/artifact/8951e47f-47c6-42ad-9969-3842e01e259c)
- [6 数据集行为序列实样](https://claude.ai/code/artifact/d8de9f20-8dae-4cca-987c-adbdbf59cb6e)
- [流程拆解](https://claude.ai/code/artifact/a5de43a9-a25b-4284-842b-0b3bb5de25d5) ·
  [结果解读](https://claude.ai/code/artifact/181ed083-e95f-48a7-a5b9-e1665a371045)

## 按数据形状的定稿配置

| 数据形状 | 遮罩/目标 | 池化 | 架构 |
|---|---|---|---|
| 操作流（App 日志） | 自回归预测下一事件 | top-k | +时间偏置注意力 |
| 资金流（转账流水） | 整事件遮罩 | mean | 标准 |
| 消费流（卡交易） | 字段级遮罩（遮金额留类别） | top-k | 标准 |

## 真实数据接入（统一流水线）

格式未知的真实数据用 `fraud_pipeline.py` 五步接入（详见 **USAGE.md**）：

```bash
python fraud_pipeline.py inspect  流水.csv                 # ① 自动识别字段 → mapping.json
python fraud_pipeline.py convert  流水.csv -m mapping.json -o data.jsonl
python fraud_pipeline.py profile  data.jsonl               # ③ 质量自检 + oracle 信号摸底
python fraud_pipeline.py run      data.jsonl --shape operation -o outputs/
python fraud_pipeline.py score    新数据.jsonl --model outputs/model.pt
```

端到端演练（模拟银行流水表 13.5 万行）验证：字段全自动识别、
往返转换无损、run 出 AUC 0.982、score 复用训练期归一化统计与阈值。

## 复现

```bash
# 主实验（合成 temporal 数据）
python gen_temporal.py && python mem_experiment.py && python mem_score_v2.py

# 编码消融（rich 基准）
python gen_rich_data.py && python mem_rich.py

# 架构探索
python mem_arch.py && python mem_arch2.py && python mem_arch3.py
```

开源数据实验（`eth_/sparkov_/banksim_/paysim_/aml_/tabformer_*.py`）需先经
kagglehub / GitHub 下载原始数据（脚本内有路径），大文件不入库。

依赖：`torch pandas scikit-learn numpy scipy`（CPU 即可，单实验分钟级）。

## 局限（诚实清单）

- 团伙欺诈的跨账户图信号（共享设备/资金链）是本方案盲区，需图方法补充
- V8 架构优势目前仅在合成基准验证（真实银行操作日志无公开数据）
- 评估仍需少量黑标签"阅卷"；阈值需按白样本分位定期重校
