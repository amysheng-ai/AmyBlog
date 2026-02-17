---
title: "Daily AI Papers - 2026年02月17日"
published: 2026-02-17
description: "精选AI论文日报：Agent认知自适应、RLVR温度策略学习、Reasoning模型对抗鲁棒性、Self-Play多样性优化"
tags: [Daily-Papers, Agent, RLVR, Reasoning, LLM-Efficiency]
category: Paper-Digest
draft: false
---

# Daily AI Papers - 2026年02月17日

## 今日预览

今天筛选出 **6篇高质量论文**，涵盖 Agent 认知自适应、RLVR 温度策略学习、Reasoning 模型对抗鲁棒性等核心方向。

**必读推荐**：
- **Think Fast and Slow**: 首次实现 Agent 步级别认知深度自适应，Qwen2.5-7B 在 ALFWorld 达到 82.3% 成功率，超越 GPT-4o 40.3%
- **Look Inward to Explore Outward**: 通过 Hierarchical RL 从 LLM 内部状态学习温度策略，为 RLVR 探索-利用权衡提供新思路
- **R-Diverse**: 揭示 Self-Play 中的 Diversity Illusion 问题，提出 Memory-Augmented Penalty 和 Skill-Aware Measurement

---

## 论文详解

### 1. Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents

#### Meta
- **Title**: Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents
- **Link**: [arXiv:2602.12662](https://arxiv.org/abs/2602.12662)
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: Agent, Reasoning, Cognitive-Adaptation
- **推荐度**: ⭐⭐⭐ 必读
- **TL;DR**: 基于 ACT-R 认知理论，实现 Agent 在步级别动态调整认知深度，在简单步骤快速响应，复杂步骤深度思考

#### Problem & Contribution
- **解决的问题**: 现有 LLM Agent 要么全程不思考（非思考模型），要么全程深度思考（思考模型），无法根据任务步骤的认知需求动态调整，导致效率低下
- **核心想法/方法一句话**: 提出 CogRouter 框架，通过 Cognition-aware SFT 和 Cognition-aware Policy Optimization 训练 Agent 在每一步选择四种认知层次之一（从本能反应到战略规划）
- **主要贡献**:
  1. 首个步级别认知深度自适应的 Agent 框架，基于 ACT-R 理论设计四级认知层次
  2. 提出 Cognition-aware Policy Optimization (CoPO)，通过置信度感知优势重加权实现步级别信用分配
  3. 在 ALFWorld 和 ScienceWorld 达到 SOTA，使用 62% 更少 token

#### Method
- **方法结构/流程**:
  1. **认知层次定义**: 基于 ACT-R 理论设计四级认知模式（L1 本能反应 → L4 战略规划）
  2. **CoSFT 阶段**: 使用人工标注的认知标签进行监督微调，建立稳定的层次特定行为模式
  3. **CoPO 阶段**: 通过置信度感知优势重加权，优化温度策略选择
  4. **核心洞察**: 适当的认知深度应最大化动作的置信度

- **关键设计**:
  - 四级认知层次：L1（无思考）、L2（轻量思考）、L3（深度思考）、L4（战略推理）
  - 置信度感知优势重加权：高置信度动作获得更高优势权重
  - 残差连接保证梯度流动，AdamW 动量为未选中层提供隐式更新

- **训练/推理成本**:
  - 模型: Qwen2.5-7B
  - 数据集: ALFWorld, ScienceWorld
  - Token 效率: 比基线减少 62% token 使用

#### Evidence
- **Benchmark / setting**: ALFWorld (室内任务), ScienceWorld (科学实验)
- **对比对象**: GPT-4o, OpenAI-o3, GRPO, ReAct
- **关键结果**:
  - ALFWorld 成功率: 82.3% (Qwen2.5-7B)
  - 相比 GPT-4o 提升: +40.3%
  - 相比 OpenAI-o3 提升: +18.3%
  - 相比 GRPO 提升: +14.0%

#### Takeaways
- **可以迁移到什么场景**: 长程任务规划、多步决策、资源受限的端侧 Agent
- **风险/注意点**: 认知层次标签获取成本较高，需要人工标注或自动标注策略
- **下一步动作**: 尝试在本地 Qwen2.5-7B 上复现，探索无标签的层次发现方法

---

### 2. Look Inward to Explore Outward: Learning Temperature Policy from LLM Internal States via Hierarchical RL

#### Meta
- **Title**: Look Inward to Explore Outward: Learning Temperature Policy from LLM Internal States via Hierarchical RL
- **Link**: [arXiv:2602.13035](https://arxiv.org/abs/2602.13035)
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: RLVR, Temperature-Policy, Hierarchical-RL
- **推荐度**: ⭐⭐⭐ 必读
- **TL;DR**: 提出 Introspective LLM，通过 Hierarchical RL 从隐藏状态学习采样温度策略，实现 RLVR 中的自适应探索-利用权衡

#### Problem & Contribution
- **解决的问题**: RLVR 中采样温度控制探索-利用权衡，但现有方法使用静态温度或启发式调整，与任务级奖励解耦
- **核心想法/方法一句话**: 在每个解码步骤，模型基于隐藏状态选择温度，联合优化温度策略和 token 策略
- **主要贡献**:
  1. 首个从 LLM 内部状态学习温度策略的 Hierarchical RL 框架
  2. 温度选择与 token 采样联合优化，实现细粒度探索控制
  3. 数学推理任务上超越固定温度和启发式基线

#### Method
- **方法结构/流程**:
  1. **上层策略**: 基于当前隐藏状态选择温度
  2. **下层策略**: 从选定温度对应的分布中采样 token
  3. **联合优化**: 使用坐标上升法从下游奖励联合优化两层策略
  4. **可解释性**: 学习到的温度策略与推理不确定性对齐

- **关键设计**:
  - 温度作为高层动作，影响低层 token 采样
  - 坐标上升优化：交替优化温度策略和 token 策略
  - 内部状态（隐藏层）作为温度选择的观测

#### Evidence
- **Benchmark / setting**: 数学推理 benchmark
- **对比对象**: 固定温度、启发式温度调整
- **关键结果**:
  - 学习到的温度策略超越所有固定温度基线
  - 展现出与推理不确定性对齐的可解释探索行为

#### Takeaways
- **可以迁移到什么场景**: RLVR 训练、多步推理任务、需要自适应探索的生成任务
- **风险/注意点**: 增加了一层策略学习，训练复杂度提升
- **下一步动作**: 结合到现有的 RLVR 代码库中，测试在 GSM8K 上的效果

---

### 3. R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training

#### Meta
- **Title**: R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training
- **Link**: [arXiv:2602.13103](https://arxiv.org/abs/2602.13103)
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: Self-Play, RL, Reasoning, Diversity
- **推荐度**: ⭐⭐⭐ 必读
- **TL;DR**: 揭示 Self-Play 中的 Diversity Illusion 问题，提出 Memory-Augmented Penalty 和 Skill-Aware Measurement，在 10 个 benchmark 上持续超越 R-Zero

#### Problem & Contribution
- **解决的问题**: Self-Play（如 R-Zero）在迭代过程中出现非持续性改进，早期收益随继续训练而退化
- **核心想法/方法一句话**: Diversity Illusion 表现为训练信号看似多样但坍缩为重复模式，包括局部多样性幻觉（仅 batch 内多样）和表面多样性幻觉（问题表面不同但推理技能相同）
- **主要贡献**:
  1. 首次系统分析 Self-Play 中的 Diversity Illusion 问题
  2. 提出 Memory-Augmented Penalty (MAP)，使用持久记忆库跨迭代阻止重复
  3. 提出 Skill-Aware Measurement (SAM)，基于推理技能而非问题表面评估多样性

#### Method
- **方法结构/流程**:
  1. **Challenger-Solver 循环**: Challenger 生成针对 Solver 能力的问题，Solver 在生成数据上优化
  2. **MAP**: 维护跨迭代的记忆库，惩罚重复出现的问题模式
  3. **SAM**: 评估问题所需的推理技能组合，确保技能级多样性
  4. **对齐训练**: MAP 和 SAM 协同工作，确保持续改进

- **关键设计**:
  - 局部多样性幻觉：仅 batch 内强制多样，导致跨迭代模式循环
  - 表面多样性幻觉：问题表述不同但核心推理技能相同
  - 技能感知：基于推理技能而非表面特征评估多样性

#### Evidence
- **Benchmark / setting**: 10 个数学和通用推理 benchmark
- **对比对象**: R-Zero 及其他 Self-Play 方法
- **关键结果**:
  - 在更多迭代中保持改进
  - 在 10 个 benchmark 上持续超越 prior self-play 方法
  - 代码开源: https://github.com/Gengsheng-Li/R-Diverse

#### Takeaways
- **可以迁移到什么场景**: Self-Play 训练、数据生成、课程学习
- **风险/注意点**: 记忆库和技能评估增加计算开销
- **下一步动作**: 阅读代码实现，探索在其他推理任务上的应用

---

### 4. Consistency of Large Reasoning Models Under Multi-Turn Attacks

#### Meta
- **Title**: Consistency of Large Reasoning Models Under Multi-Turn Attacks
- **Link**: [arXiv:2602.13093](https://arxiv.org/abs/2602.13093)
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: Reasoning, Safety, Adversarial-Robustness
- **推荐度**: ⭐⭐⭐ 必读
- **TL;DR**: 系统评估 9 个前沿推理模型在多轮对抗攻击下的鲁棒性，发现推理能力提供有意义但不完整的鲁棒性，识别出 5 种失效模式

#### Problem & Contribution
- **解决的问题**: 大型推理模型在复杂任务上表现优异，但其在多轮对抗压力下的鲁棒性尚未充分探索
- **核心想法/方法一句话**: 评估 9 个前沿推理模型在多轮对抗攻击下的表现，通过轨迹分析识别失效模式
- **主要贡献**:
  1. 首个针对推理模型的多轮对抗攻击系统评估
  2. 识别 5 种失效模式：Self-Doubt、Social Conformity、Suggestion Hijacking、Emotional Susceptibility、Reasoning Fatigue
  3. 发现 CARG 对推理模型失效，随机置信度嵌入反而优于目标提取

#### Method
- **方法结构/流程**:
  1. **攻击类型**: 误导性建议、社会压力等
  2. **模型评估**: 9 个前沿推理模型 vs 指令微调基线
  3. **轨迹分析**: 分析失败案例的推理轨迹
  4. **防御测试**: 测试 CARG (Confidence-Aware Response Generation) 有效性

- **关键发现**:
  - 推理模型显著优于指令微调基线
  - 所有模型都表现出独特的脆弱性特征
  - 误导性建议普遍有效，社会压力效果因模型而异
  - Self-Doubt 和 Social Conformity 占失败的 50%

#### Evidence
- **Benchmark / setting**: 多轮对抗攻击场景
- **对比对象**: 9 个推理模型 vs 指令微调基线
- **关键结果**:
  - 推理模型比基线更鲁棒，但仍存在显著漏洞
  - 5 种失效模式被识别并量化
  - CARG 对推理模型失效（过度自信导致）
  - 随机置信度嵌入 > 目标提取

#### Takeaways
- **可以迁移到什么场景**: 安全对齐、对抗训练、红队测试
- **风险/注意点**: 推理能力≠对抗鲁棒性，需要重新设计基于置信度的防御
- **下一步动作**: 关注基于推理轨迹的防御机制研究

---

### 5. BrowseComp-V^3: A Visual, Vertical, and Verifiable Benchmark for Multimodal Browsing Agents

#### Meta
- **Title**: BrowseComp-$V^3$: A Visual, Vertical, and Verifiable Benchmark for Multimodal Browsing Agents
- **Link**: [arXiv:2602.12876](https://arxiv.org/abs/2602.12876)
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: Agent, Multimodal, Benchmark, Web-Browsing
- **推荐度**: ⭐⭐ 可选
- **TL;DR**: 提出新的多模态浏览 Agent benchmark，强调深度、多层次、跨模态推理，SOTA 模型仅达 36% 准确率

#### Problem & Contribution
- **解决的问题**: 现有 MLLM 浏览 benchmark 在任务复杂度、证据可访问性、评估粒度方面存在局限
- **核心想法/方法一句话**: 300 个跨领域精心设计的难题，要求文本和视觉模态的深度多跳推理，所有证据可公开搜索
- **主要贡献**:
  1. 300 个跨领域高难度问题，强调深度多级跨模态推理
  2. 专家验证的子目标驱动过程评估机制
  3. 提出 OmniSeeker 统一多模态浏览 Agent 框架

#### Method
- **方法结构/流程**:
  1. **数据构建**: 300 个问题，证据跨页面交错分布
  2. **验证机制**: 所有证据必须可公开搜索，确保可复现
  3. **评估**: 最终答案准确性 + 子目标驱动过程评估
  4. **Agent 设计**: OmniSeeker 集成多种搜索和视觉感知工具

#### Evidence
- **Benchmark / setting**: 300 个跨领域问题
- **对比对象**: SOTA MLLM
- **关键结果**:
  - SOTA 模型准确率仅 36%
  - 揭示多模态信息整合和细粒度感知的关键瓶颈

#### Takeaways
- **可以迁移到什么场景**: Web Agent 评估、多模态推理研究
- **风险/注意点**: Benchmark 难度极高，可能不适合初级模型评估
- **下一步动作**: 关注后续基于此 benchmark 的 Agent 改进工作

---

### 6. LCSB: Layer-Cyclic Selective Backpropagation for Memory-Efficient On-Device LLM Fine-Tuning

#### Meta
- **Title**: LCSB: Layer-Cyclic Selective Backpropagation for Memory-Efficient On-Device LLM Fine-Tuning
- **Link**: [arXiv:2602.13073](https://arxiv.org/abs/2602.13073)
- **Venue**: arXiv preprint (under review)
- **Date**: 2026-02-13
- **Tags**: Efficient-LLM, On-Device, Memory-Optimization, LoRA
- **推荐度**: ⭐⭐ 可选
- **TL;DR**: 提出 Layer-Cyclic Selective Backpropagation，每步仅计算部分层的梯度，通过残差连接和 AdamW 动量保证收敛，实现 1.40x 加速且质量退化 <2%

#### Problem & Contribution
- **解决的问题**: 端侧 LLM 微调内存受限，MeBP 需要反向计算所有层，权重解压占 32-42% 反向时间
- **核心想法/方法一句话**: 每步仅选择部分层计算梯度，残差连接保证梯度流动，AdamW 动量为未选中层提供隐式更新
- **主要贡献**:
  1. 首个层循环选择性反向传播方法，实现内存-效率权衡
  2. 理论解释：LCSB 等价于 LoRA 参数空间的块坐标下降
  3. 4-bit 量化设置下展现卓越稳定性（完整反向传播发散的模型 LCSB 能稳定收敛）

#### Method
- **方法结构/流程**:
  1. **层选择**: 每步循环选择部分层计算梯度
  2. **梯度流动**: 残差连接保证未选中层的梯度通过恒等路径传播
  3. **隐式更新**: AdamW 动量为未选中层提供隐式参数更新
  4. **理论分析**: 证明 LCSB 等价于 LoRA 空间的块坐标下降

- **关键设计**:
  - 循环层选择策略
  - 残差连接 + AdamW 动量的隐式更新机制
  - 4-bit 量化下的隐式正则化效应

#### Evidence
- **Benchmark / setting**: 5 个模型，3 个任务
- **对比对象**: 完整反向传播 (MeBP), MeZO
- **关键结果**:
  - 最高 1.40x 加速
  - 质量退化 <2%
  - 4-bit 量化下：3B 模型完整反向传播发散，LCSB 稳定收敛
  - MeZO 梯度估计与真实梯度余弦相似度 ≈0.001

#### Takeaways
- **可以迁移到什么场景**: 端侧 LLM 微调、内存受限环境、LoRA 训练
- **风险/注意点**: 层选择策略需要针对不同模型调优
- **下一步动作**: 尝试集成到现有端侧训练框架

---

## 总结

| 论文 | 推荐度 | TL;DR | 下一步 |
|------|--------|-------|--------|
| Think Fast and Slow | ⭐⭐⭐ | 步级别认知深度自适应，82.3% ALFWorld 成功率 | 复现或探索无标签层次发现 |
| Look Inward to Explore Outward | ⭐⭐⭐ | Hierarchical RL 学习温度策略，自适应探索-利用 | 集成到 RLVR 代码库 |
| R-Diverse | ⭐⭐⭐ | 揭示 Diversity Illusion，MAP+SAM 持续改进 | 阅读代码，应用到其他任务 |
| Consistency of Large Reasoning Models | ⭐⭐⭐ | 推理模型对抗鲁棒性系统评估，5种失效模式 | 关注推理轨迹防御机制 |
| BrowseComp-V^3 | ⭐⭐ | 多模态浏览 Agent benchmark，SOTA 仅 36% | 关注后续 Agent 改进工作 |
| LCSB | ⭐⭐ | 层循环选择性反向传播，1.40x 加速 | 集成到端侧训练框架 |

**今日趋势观察**：
1. **Agent 认知自适应成为热点**：从固定思考模式转向动态认知深度调整，显著提升效率
2. **RLVR 探索机制精细化**：从静态温度转向基于内部状态的自适应温度策略
3. **Self-Play 多样性问题被重视**：Diversity Illusion 的揭示为持续改进提供新方向

---

*Curated by Amy 🤖*
