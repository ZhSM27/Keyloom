# 用新版键型规则重训的规划

状态：规划，未执行。训练语料和完整训练流水线不在本仓库，所有训练都要在仓库外进行；本文只规定标签、版本、阶段、门槛和验收方式。踩音准确度（多压踩重音、无音可踩时放弃）的改动见 [踩音准确度规划](accent-alignment-plan.md)，与本轮的 S4a 合并训练。

## 1. 目标与边界

- 把规划器、角色规划器、密度适配器和生成器的键型监督，从 YAVSRG 片段覆盖率（单小节覆盖 ≥50%）换成 `community-pattern-rules-v2-tech`，并新增第 16 个键型 **Tech**。
- r1 全部保留为冻结基线，新模型使用 r2 修订号，与 r1 并存；没有试听试玩证据就不晋升默认，也不因压力更低而晋升（AGENTS.md）。
- 手部压力观测维持 2/4 拍；标签语义、校准数据和解码前缀同时换版本，旧阈值不复用。

## 2. 标签迁移

| 项目 | r1（旧） | r2（新） |
| --- | --- | --- |
| 标签来源 | 检测器区间覆盖 ≥0.5 | 每小节按规则判定，多标签 `matches` 加一个主标签 `dominant` |
| 词表 | 15 个 `LABELS` | 15 个旧标签，**Tech 追加在末尾**（索引 15），旧索引不变 |
| 无标签小节 | 人类谱约 48% | 约 10%（150 张本机人类谱实测） |
| 版本号 | 无独立版本 | `DEFINITION_SCHEMA` 写入每条样本和每个检查点 |

本机人类谱上的主标签分布：Chordjacks 15.9%、Jumpstream 14.7%、Minijacks 14.0%、Tech 13.3%、Handstream 10.1%、无标签 9.8%；Split Trill 0.3%、Minitrills 0.2%、Trills 0.7% 极少。稀有类需要加权或在报告中单列，不能只看总体准确率。

需要改的代码（全部按 r2 新增，不改 r1 路径）：

- `segment_labels.py`：新增 `LABELS_V2 = (*LABELS, "Tech")`，保留 `LABELS` 供 r1 使用。
- `segment_supervision.PLAN_WIDTH`、`main_style.STYLE_NAMES`：按词表版本取宽度（16 / 17）。
- `arrangement_model.py`：`pattern_head` 输出 16；`condition_width` 17→18。
- `conditional_role_planner.py`：主副嵌入、owner/style 输出头和关系矩阵扩到 16。
- 生成器：plan 投影输入 16；`style_count_allowed` 17×4，Tech 一行从数据统计。
- `style_hierarchy.LOWER_STYLES`：Tech 不设下位型；`density_coordination` 给 Tech 设和弦容量。
- `studio/loader.py` 与 `catalog.json`：新增 r2 架构 ID 和资源；读取时校验词表版本，不匹配直接拒绝。

## 3. 数据准备

1. 用 `keyloom_mania.evaluation.definitions` 给语料的每个小节重新打标签，保存 `matches`（16 维布尔）、`dominant`、四小节短语多数标签和 `DEFINITION_SCHEMA`。
2. 对同一语料输出旧检测器标签与新规则的混淆矩阵，按键型列出变化最大的部分，并人工抽查每类约 30 个小节。
3. 按歌曲（谱面集）划分 train/dev/test，同一首歌的不同难度不得跨集。dev 歌单固定并在训练前登记；test 只在最终验收时使用。
4. 沿用现有过滤条件：长条占比 ≤10%，排除本项目生成的谱面，排除计时段过多的谱面。
5. 重新拟合 `thresholds.npz` 和 compiler 配置，只用 dev 集，并记录拟合数据的摘要。

## 4. 训练阶段

| 阶段 | 内容 | 依赖 | 产物 |
| --- | --- | --- | --- |
| S1 | 规划器 r2：键型头改为 16 维多标签监督，监督目标取 `matches`，不再取覆盖率；密度头目标不变 | 新标签 | `planner/arrangement-r2.pt`、阈值、compiler 配置 |
| S2 | 角色规划器 r2（standard / balanced）：owner/style 头 16 维 | S1 冻结 | `planner/role-*-r2.pt` |
| S3 | 密度适配器 r2：绑定 S1 的 SHA-256 | S1 冻结 | `planner/density-contrast-r2.pt` |
| S4a | 生成器微调：从 music-base r1 出发，plan 投影新增的 Tech 列初始化为 0，使无 Tech 小节的行为与 r1 一致 | S1 计划 | `generator/music-base-r2.pt` |
| S4b | （可选）生成器完整重训，仅在 S4a 证明 Tech 列学不动时进行 | S4a 结论 | 新家族 ID |

S4 之前先修一个解码问题：24 槽网格中 1/16 与 1/12 的相邻槽只差 1 tick，生成器会在两个槽上都放音，评测发现一张生成谱有 517 处 2–10 ms 错位行。这个问题会直接污染 Tech 小节，必须先修。计数前缀要按新解码器重新采样，不沿用 r1 的前缀。

## 5. 评估协议（评测器 v3）

每个阶段都和 r1 在同一 dev 歌单、同一设置、两个固定种子下对比，各项分开报告：

1. **标签质量（S1、S2）**：规划器预测与规则标签在 dev 上的逐类 F1，Tech 和稀有类单列。
2. **键型兑现**：`plan_adherence` 逐键型通过率，以及相对人类上限的兑现度。
3. **自然度**：整谱自然度和键型内自然度。重点看 r1 现在偏离最大的类别：Chordjacks（转移分布 65% 偏离）、Minijacks（48%）、Gluts（36–43%）、Tech（16/28 张偏离）。
4. **硬约束**：错位行、<40 ms 同列连击、短叠审计必须为 0。
5. **计划 vs 导出**：音符数、行数、减键和丢行照常列出。
6. **星级**：只记录，不作为门槛。

晋升条件：在任一自然度维度上都不劣于 r1，键型兑现有提升，硬约束为 0，并且通过试听和试玩。自动分数只用于筛选候选。

## 6. 风险

- **循环评估**：训练标签和兑现评估都用同一套规则，键型兑现会偏乐观。应同时看不依赖标签的整谱自然度、留出集表现和人工试听。
- **阈值未经广泛检验**：规则阈值只在本机 820 张谱上检查过。Tech 和 Minitrills 没有社区标准定义，可能需要按 dev 结果再调；每次调整都要改 `DEFINITION_SCHEMA` 并重打标签。
- **分布变化**：Tech 占 13%，可能挤占原有流类的计划份额，需要检查整曲键型构成是否随音乐变化。

## 7. 开始前需要确认

- 训练语料和训练脚本的位置（不在本仓库）。
- 计算预算：先只做 S1+S4a，还是四个阶段全做。
- 生成器起点：建议先用 music-base r1 微调（S4a）。
