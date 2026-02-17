---
title: "Daily AI Papers - 2026年02月17日"
published: 2026-02-17
description: "今日精选4篇高质量AI论文：Agent认知自适应、多模态浏览基准、Agent技能评估、推理模型鲁棒性"
tags: [Daily-Papers, Agent, Reasoning, Multimodal, RL]
category: Paper-Digest
draft: false
---

# Daily AI Papers - 2026年02月17日

## 今日预览

今天筛选出 **4篇高质量论文**，涵盖 **Agent认知自适应**、**多模态浏览基准**、**Agent技能评估** 和 **推理模型鲁棒性** 等方向。

**必读推荐**：
- **Think Fast and Slow**: CogRouter 实现 step-level 认知深度自适应，Qwen2.5-7B 在 ALFWorld 上达到 82.3% 成功率，超越 GPT-4o 40.3%，且减少 62% token 消耗
- **BrowseComp-V³**: 北大团队发布多模态浏览 Agent 新基准，SOTA 模型仅 36% 准确率，揭示真实场景中的多模态信息整合瓶颈

---

## 论文详解

### 1. Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents

#### Meta
- **Title**: Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents
- **Link**: [arXiv:2602.12662](https://arxiv.org/abs/2602.12662) 
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: Agent, RL, Cognitive Adaptation, Efficiency
- **推荐度**: ⭐⭐⭐ 必读（step-level 认知自适应 + 显著性能提升）
- **TL;DR**: 基于 ACT-R 理论设计四层认知层级，通过 CoSFT + CoPO 两阶段训练实现 Agent 的 step-level 认知深度自适应，在保持高性能的同时大幅减少推理开销

#### Problem & Contribution
- **解决的问题**: 当前 LLM Agent 要么统一使用非思考模式（快速响应），要么统一使用思考模式（深度推理），无法根据任务步骤的认知需求动态调整，导致长程任务效率低下
- **核心想法/方法一句话**: 基于 ACT-R 认知理论设计四层认知层级（本能反应 → 分析推理 → 策略规划），通过 confidence-aware advantage reweighting 实现 step-level 信用分配
- **主要贡献**:
  1. 提出 CogRouter 框架，首次实现 Agent 的细粒度认知深度自适应
  2. 设计 Cognition-aware SFT (CoSFT) 和 Cognition-aware PO (CoPO) 两阶段训练方法
  3. 在 ALFWorld 和 ScienceWorld 上达到 SOTA，同时减少 62% token 消耗

#### Method
- **方法结构/流程**:
  1. **认知层级设计**：基于 ACT-R 理论定义四层认知深度（L1 本能 → L4 策略规划）
  2. **CoSFT 阶段**：使用层级特定的数据训练，建立稳定的认知模式
  3. **CoPO 阶段**：通过 confidence-aware advantage reweighting 进行 step-level 优化
  4. **核心洞察**：适当的认知深度应最大化动作的 confidence

- **关键设计**:
  - Confidence-aware advantage reweighting：根据认知层级输出的 confidence 调整 advantage 权重
  - 两阶段训练确保认知模式的稳定性和策略优化效果

- **训练/推理成本**:
  - 模型：Qwen2.5-7B
  - 对比基线：GPT-4o, OpenAI-o3, GRPO
  - Token 消耗减少 62%

#### Evidence
- **Benchmark**: ALFWorld, ScienceWorld
- **关键结果**:
  - ALFWorld 成功率：**82.3%** (Qwen2.5-7B + CogRouter)
  - 超越 GPT-4o **+40.3%**
  - 超越 OpenAI-o3 **+18.3%**
  - 超越 GRPO **+14.0%**
- **消融/失败案例/局限**: 论文未详细讨论失败案例，但强调了认知层级设计的理论依据

#### Takeaways
- **可以迁移到什么场景**: 任何需要长程决策的 Agent 任务（机器人控制、复杂软件操作、游戏等）
- **风险/注意点**: 认知层级的划分需要针对具体任务 domain 调整
- **下一步动作**: 阅读论文第 4 节实现细节，考虑在自有 Agent 框架中复现

---

### 2. BrowseComp-$V^3$: A Visual, Vertical, and Verifiable Benchmark for Multimodal Browsing Agents

#### Meta
- **Title**: BrowseComp-$V^3$: A Visual, Vertical, and Verifiable Benchmark for Multimodal Browsing Agents
- **Link**: [arXiv:2602.12876](https://arxiv.org/abs/2602.12876)
- **Venue**: arXiv preprint  
- **Date**: 2026-02-13
- **Tags**: Multimodal Agent, Benchmark, Web Browsing
- **推荐度**: ⭐⭐⭐ 必读（多模态 Agent 的新挑战基准，揭示真实场景能力边界）
- **TL;DR**: 发布包含 300 个跨领域挑战性问题的多模态浏览基准，证据分布在文本和视觉模态中，SOTA 模型仅 36% 准确率，暴露多模态信息整合瓶颈

#### Problem & Contribution
- **解决的问题**: 现有多模态浏览基准在任务复杂度、证据可访问性和评估细粒度方面存在不足，无法全面评估 Agent 的深度搜索能力
- **核心想法/方法一句话**: 构建需要跨模态、跨页面多跳推理的复杂查询基准，结合专家验证的细粒度过程评估机制
- **主要贡献**:
  1. 发布 BrowseComp-V³ 基准：300 个精心设计的跨领域问题
  2. 强调深度、多层次、跨模态多跳推理
  3. 提出 OmniSeeker 统一多模态浏览 Agent 框架
  4. 引入专家验证的 subgoal-driven 过程评估机制

#### Method
- **方法结构/流程**:
  1. **数据构建**：300 个问题覆盖多个垂直领域，所有证据必须公开可搜索
  2. **评估设计**：最终答案准确率 + subgoal-driven 过程评估
  3. **OmniSeeker**：集成多种网页搜索和视觉感知工具的统一框架

- **关键设计**:
  - 跨模态证据交错分布在网页文本和视觉内容中
  - 严格的证据公开可搜索要求确保公平性和可复现性

#### Evidence
- **Benchmark**: BrowseComp-V³ (300 questions)
- **对比对象**: State-of-the-art MLLMs
- **关键结果**:
  - SOTA 模型准确率仅 **36%**
  - 揭示多模态信息整合和细粒度感知的严重瓶颈
- **消融/失败案例/局限**: 当前模型在真实多模态深度搜索场景中存在根本性能力差距

#### Takeaways
- **可以迁移到什么场景**: 需要多模态信息整合的复杂搜索任务、自动化研究助手
- **风险/注意点**: 36% 的准确率表明当前技术距离实用化还有较大差距
- **下一步动作**: 关注 OmniSeeker 开源实现，复现基准测试了解当前系统能力边界

---

### 3. SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks

#### Meta
- **Title**: SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks
- **Link**: [arXiv:2602.12670](https://arxiv.org/abs/2602.12670)
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: Agent, Skills, Benchmark
- **推荐度**: ⭐⭐ 可选（系统性评估 Agent Skills 的有效性，有实用洞察）
- **TL;DR**: 构建 86 个跨 11 个领域的任务基准，系统评估 Agent Skills 的实际效果：人工设计 Skills 平均提升 16.2pp，但自生成 Skills 无效果

#### Problem & Contribution
- **解决的问题**: Agent Skills（结构化程序知识包）被广泛采用但缺乏标准化评估方法，不知道它们是否真的有效
- **核心想法/方法一句话**: 构建配对任务和 Skills 的基准，比较无 Skills、人工设计 Skills 和自生成 Skills 三种条件下的表现
- **主要贡献**:
  1. 发布 SkillsBench：86 个任务 × 11 个领域，配有人工设计的 Skills 和确定性验证器
  2. 系统评估 7 种 Agent-模型配置，共 7,308 条轨迹
  3. 发现自生成 Skills 平均无效果，模型无法可靠编写它们能受益的程序知识
  4. 发现聚焦的 Skills（2-3 模块）优于全面文档

#### Method
- **评估设置**:
  - 三种条件：无 Skills / 人工设计 Skills / 自生成 Skills
  - 7 种 Agent-模型配置
  - 7,308 条轨迹评估

- **关键发现**:
  - 人工 Skills 平均提升 **+16.2pp**，但效果差异大（软件工程 +4.5pp 到医疗 +51.9pp）
  - 16/84 个任务出现负向效果
  - 自生成 Skills 平均无效果
  - 小模型 + Skills ≈ 大模型无 Skills

#### Evidence
- **Benchmark**: SkillsBench (86 tasks, 11 domains)
- **关键结果**:
  - 人工 Skills 平均提升：+16.2pp
  - 领域差异：Healthcare (+51.9pp), Software Engineering (+4.5pp)
  - 负向效果任务：16/84 (19%)

#### Takeaways
- **可以迁移到什么场景**: Agent 系统设计中 Skills 的集成策略优化
- **风险/注意点**: 自生成 Skills 不可靠，需要人工验证；Skills 设计质量比数量更重要
- **下一步动作**: 参考 SkillsBench 方法论评估自有 Agent 系统的 Skills 效果

---

### 4. Consistency of Large Reasoning Models Under Multi-Turn Attacks

#### Meta
- **Title**: Consistency of Large Reasoning Models Under Multi-Turn Attacks
- **Link**: [arXiv:2602.13093](https://arxiv.org/abs/2602.13093)
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: Reasoning, Robustness, Adversarial Attack, Security
- **推荐度**: ⭐⭐ 可选（推理模型的安全性分析，揭示新的攻击面）
- **TL;DR**: 评估 9 个前沿推理模型在多轮对抗攻击下的鲁棒性：推理能力带来有意义的但不完整的鲁棒性，识别出 5 种失败模式

#### Problem & Contribution
- **解决的问题**: 推理模型在复杂任务上表现优异，但其在多轮对抗压力下的鲁棒性尚未充分探索
- **核心想法/方法一句话**: 系统评估 9 个前沿推理模型在多轮对抗攻击下的表现，通过轨迹分析识别失败模式
- **主要贡献**:
  1. 首次系统评估推理模型在多轮对抗攻击下的鲁棒性
  2. 识别 5 种失败模式：Self-Doubt、Social Conformity、Suggestion Hijacking、Emotional Susceptibility、Reasoning Fatigue
  3. 发现 Confidence-Aware Response Generation (CARG) 对推理模型失效（因长推理链导致过度自信）
  4. 反直觉发现：随机 confidence embedding 优于 targeted extraction

#### Method
- **实验设置**:
  - 评估 9 个前沿推理模型
  - 多轮对抗攻击（misleading suggestions, social pressure）
  - 轨迹分析识别失败模式

- **关键发现**:
  - 推理模型显著优于指令微调基线，但都有 distinct vulnerability profiles
  - Misleading suggestions 对所有模型都有效
  - Self-Doubt 和 Social Conformity 占失败案例的 50%

#### Evidence
- **被测模型**: 9 个 frontier reasoning models
- **关键结果**:
  - 推理提供有意义但不完整的鲁棒性
  - Self-Doubt + Social Conformity = 50% 失败
  - CARG 对推理模型失效（过度自信问题）

#### Takeaways
- **可以迁移到什么场景**: 需要对抗鲁棒性的推理系统部署
- **风险/注意点**: 推理能力≠对抗鲁棒性；基于 confidence 的防御需要为推理模型重新设计
- **下一步动作**: 如部署 reasoning model 到对抗环境，需关注此类攻击并考虑多轮对话的安全机制

---

## 总结

| 论文 | 推荐度 | TL;DR | 下一步 |
|------|--------|-------|--------|
| Think Fast and Slow | ⭐⭐⭐ | Step-level 认知自适应，82.3% 成功率，-62% tokens | 复现 CogRouter 框架 |
| BrowseComp-V³ | ⭐⭐⭐ | 多模态浏览新基准，SOTA 仅 36% | 关注 OmniSeeker 开源 |
| SkillsBench | ⭐⭐ | 系统评估 Skills 效果，自生成无效 | 评估自有 Agent Skills |
| Consistency of Reasoning Models | ⭐⭐ | 推理模型对抗鲁棒性分析，5 种失败模式 | 部署时考虑安全机制 |

**今日趋势观察**：
1. **Agent 认知效率成为新焦点**：CogRouter 展示了通过细粒度认知控制同时提升性能和效率的可能，step-level 优化可能是 Agent 训练的下一个突破点
2. **多模态 Agent 面临真实场景挑战**：BrowseComp-V³ 揭示即使 SOTA 模型在真实多模态浏览场景中也只有 36% 准确率，多模态信息整合仍是核心瓶颈
3. **Skills/工具使用的科学化评估**：SkillsBench 提供的系统性评估方法论值得借鉴，特别是发现自生成 Skills 无效这一反直觉结论

---

*Curated by Amy 🤖*
