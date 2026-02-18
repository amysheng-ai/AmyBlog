---
title: "Daily AI Papers - 2026年02月18日"
published: 2026-02-18
description: "精选AI论文日报 - GLM-5、STAPO、RCE组合推理突破、SAH理论新解"
tags: [Daily-Papers, Agentic-RL, Compositional-Reasoning, RLVR, Theory]
category: Paper-Digest
draft: false
---

# Daily AI Papers - 2026年02月18日

**数据来源**: HuggingFace Daily Papers + arXiv (cs.AI/cs.LG/cs.CL)

## 今日预览

今天从 **HuggingFace** 和 **arXiv** 共筛选出 **6篇高质量论文**，涵盖 **Agentic RL、组合推理突破、RL训练稳定性、理论洞察** 等核心方向。

**必读推荐**：
- **GLM-5**: 智谱/清华发布下一代基础模型，DSA架构+异步RL实现Agentic Engineering范式转移
- **STAPO**: 揭示RL微调中0.01%的spurious tokens导致训练崩溃，提出稳定性解决方案
- **RCE**: 递归概念演化框架，在ARC-AGI-2等hard benchmark上取得8-18点提升
- **SAH via Task Complexity**: 用任务复杂度重新诠释表面一致性假设，揭示post-training的本质

---

## 论文详解

### 1. GLM-5: from Vibe Coding to Agentic Engineering

#### Meta
- **Title**: GLM-5: from Vibe Coding to Agentic Engineering
- **Link**: [arXiv:2602.15763](https://arxiv.org/abs/2602.15763) | [Code](https://github.com/zai-org/GLM-5) ⭐ 1049
- **Venue**: arXiv preprint
- **Date**: 2026-02-17
- **Source**: HuggingFace Daily Papers
- **Tags**: Agentic RL, DSA, Asynchronous RL, Code Generation
- **推荐度**: ⭐⭐⭐ 必读（顶级机构、范式创新、SOTA性能）
- **TL;DR**: 通过DSA架构和异步强化学习基础设施，GLM-5实现了从"vibe coding"到"agentic engineering"的范式转移，在真实软件工程任务上达到SOTA

#### Problem & Contribution
- **解决的问题**: 
  - 现有代码生成模型缺乏端到端软件工程能力
  - RL训练效率低，生成与训练耦合
  - 长上下文场景下推理成本高
  
- **核心想法/方法一句话**: 采用DSA(Dynamic Sparse Attention)降低训练/推理成本，通过异步RL基础设施解耦生成与训练，提升post-training效率

- **主要贡献**:
  1. **DSA架构**: 显著降低训练和推理成本，同时保持长上下文保真度
  2. **异步RL基础设施**: 解耦生成与训练，大幅提升post-training效率
  3. **异步Agent RL算法**: 从复杂长程交互中更有效学习

#### Method
- **方法结构/流程**:
  1. 基于前代ARC(Agentic, Reasoning, Coding)能力构建
  2. 采用DSA实现动态稀疏注意力
  3. 构建异步RL基础设施，并行化生成与训练
  4. 提出新的异步agent RL算法优化长期交互学习

- **关键设计**:
  - **DSA (Dynamic Sparse Attention)**: 动态选择关键token进行注意力计算
  - **异步RL**: 生成和训练流水线并行，消除等待开销
  - **Agentic能力**: 支持端到端软件工程任务

- **训练/推理成本**:
  - 相比Dense注意力显著降低计算成本
  - 长上下文场景下保持高效

#### Evidence
- **Benchmark / setting**: 主要开源基准测试 + 真实软件工程任务
- **对比对象**: 现有代码生成基线
- **关键结果**:
  - 在主要开源基准上达到**SOTA性能**
  - 真实代码任务能力**超越之前所有基线**
  - 端到端软件工程挑战处理能力显著提升
- **消融/局限**: 暂未披露详细消融实验

#### Takeaways
- **可以迁移到什么场景**: 
  - 大型软件工程项目自动化
  - Agent-based代码生成系统
  - 长上下文推理优化
  
- **风险/注意点**:
  - 异步RL基础设施的工程复杂度
  - DSA对特定任务的适用性需验证
  
- **下一步动作**: 关注开源代码和模型权重发布，评估在特定软件工程任务上的表现

---

### 2. STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens

#### Meta
- **Title**: STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens
- **Link**: [arXiv:2602.15620](https://arxiv.org/abs/2602.15620)
- **Venue**: arXiv preprint
- **Date**: 2026-02-17
- **Source**: HuggingFace Daily Papers + arXiv cs.CL
- **Tags**: RL for LLMs, Training Stability, Spurious Tokens, GRPO
- **推荐度**: ⭐⭐⭐ 必读（问题本质揭示、实用解决方案、显著提升）
- **TL;DR**: 证明RL微调中仅0.01%的spurious tokens导致训练不稳定和late-stage崩溃，提出STAPO选择性mask这些token实现7.13%性能提升

#### Problem & Contribution
- **解决的问题**:
  - RL微调大模型时常见的late-stage性能崩溃
  - 现有方法(熵正则化、重加权)依赖启发式技巧且效果有限
  - 训练不稳定导致推理质量下降
  
- **核心发现**: 
  - Token-wise policy gradient幅度与token概率和局部策略熵负相关
  - **仅0.01%的spurious tokens导致训练不稳定**
  - 这些token在正确响应中贡献小但继承完整序列奖励，导致梯度异常放大

- **主要贡献**:
  1. **理论分析**: 揭示RL训练不稳定的根本原因（spurious tokens）
  2. **STAPO算法**: 选择性mask spurious token更新，对valid tokens重归一化损失
  3. **显著性能提升**: 在6个数学推理基准上平均提升7.13%

#### Method
- **方法结构/流程**:
  1. 分析token-wise policy gradient与token概率、策略熵的关系
  2. 识别spurious tokens（约0.01%）
  3. 在训练过程中mask这些token的梯度更新
  4. 对剩余valid tokens重归一化损失

- **关键设计**:
  - **Spurious Token识别**: 基于梯度幅度和token概率的统计特性
  - **选择性Masking**: 只屏蔽问题token，保留有效梯度信号
  - **损失重归一化**: 确保训练信号不被过度稀释

- **训练/推理成本**:
  - 额外开销可忽略
  - 可与现有RL框架无缝集成

#### Evidence
- **Benchmark / setting**: 6个数学推理基准
- **对比对象**: GRPO, 20-Entropy, JustRL
- **关键结果**:
  - **Qwen 1.7B/8B/14B** 上均取得一致提升
  - 相比GRPO平均提升 **7.13%**
  - 相比20-Entropy和JustRL提升 **20%**
  - **熵稳定性显著优于基线**
- **消融/局限**: 仅在数学推理任务验证，其他领域需进一步测试

#### Takeaways
- **可以迁移到什么场景**:
  - 所有基于RL的大模型微调任务
  - 需要稳定训练的长程推理任务
  - GRPO等RL算法的改进
  
- **风险/注意点**:
  - Spurious token的识别阈值可能需要任务调优
  - 在开放域生成任务中的效果待验证
  
- **下一步动作**: 尝试集成到现有RL训练流程，监控训练稳定性指标

---

### 3. Recursive Concept Evolution for Compositional Reasoning in Large Language Models

#### Meta
- **Title**: Recursive Concept Evolution for Compositional Reasoning in Large Language Models
- **Link**: [arXiv:2602.15725](https://arxiv.org/abs/2602.15725)
- **Venue**: arXiv preprint
- **Date**: 2026-02-17
- **Source**: arXiv cs.AI/cs.LG/cs.CL
- **Tags**: Compositional Reasoning, ARC-AGI, Representation Learning, Concept Evolution
- **推荐度**: ⭐⭐⭐ 必读（核心问题突破、硬基准显著提升、方法论创新）
- **TL;DR**: 提出递归概念演化(RCE)框架，让模型在推理时动态修改内部表示空间，在ARC-AGI-2等组合推理hard benchmark上取得12-18点提升

#### Problem & Contribution
- **解决的问题**:
  - LLM在需要组合推理的基准（ARC-AGI-2、GPQA、MATH、BBH、HLE）上性能急剧下降
  - 现有方法（CoT、self-consistency、RL）只扩展token级搜索，不修改潜在表示空间
  - 当所需抽象未编码在表示空间时，性能崩溃
  
- **核心想法**: 让预训练语言模型在推理时**修改其内部表示几何结构**，动态生成低秩概念子空间

- **主要贡献**:
  1. **RCE框架**: 首个在推理时动态演化表示空间的框架
  2. **概念子空间机制**: 检测表示不足时生成、通过MDL准则选择、合并协同子空间、约束优化保持稳定性
  3. **硬基准突破**: 在多个公认难题上取得显著提升

#### Method
- **方法结构/流程**:
  1. **检测表示不足**: 识别当前表示无法捕捉所需抽象的情况
  2. **生成概念子空间**: 动态生成低秩子空间
  3. **MDL选择**: 用最小描述长度准则选择最优子空间
  4. **合并协同子空间**: 整合相关概念
  5. **约束优化**: 保持稳定性同时巩固新概念

- **关键设计**:
  - **动态子空间生成**: 按需创建新概念表示
  - **递归演化**: 概念可以进一步分解和细化
  - **稳定性约束**: 防止表示空间过度混乱

- **训练/推理成本**:
  - 推理时动态计算，增加一定计算开销
  - 但避免了对整个模型进行昂贵重训练

#### Evidence
- **Benchmark / setting**: ARC-AGI-2, GPQA, MATH, BBH, HLE
- **对比对象**: Mistral-7B基线
- **关键结果**:
  - **ARC-AGI-2**: 提升 **12-18 points**
  - **GPQA**: 提升 **8-14 points**
  - **BBH**: 提升 **8-14 points**
  - **MATH & HLE**: 深度诱导错误一致减少
- **消融/局限**: 仅在Mistral-7B上验证，更大模型效果待验证

#### Takeaways
- **可以迁移到什么场景**:
  - 任何需要组合推理的复杂任务
  - 动态概念学习的agent系统
  - 表示学习研究
  
- **风险/注意点**:
  - 推理时计算开销增加
  - 概念子空间的可解释性需进一步研究
  
- **下一步动作**: 关注代码开源，尝试在相关任务上复现效果

---

### 4. Operationalising the Superficial Alignment Hypothesis via Task Complexity

#### Meta
- **Title**: Operationalising the Superficial Alignment Hypothesis via Task Complexity
- **Link**: [arXiv:2602.15829](https://arxiv.org/abs/2602.15829)
- **Venue**: arXiv preprint
- **Date**: 2026-02-17
- **Source**: arXiv cs.LG
- **Tags**: Superficial Alignment Hypothesis, Task Complexity, Pre-training, Post-training
- **推荐度**: ⭐⭐⭐ 必读（核心理论问题、新视角、实验扎实）
- **TL;DR**: 用"任务复杂度"（最短程序长度）重新诠释表面一致性假设(SAH)，证明pre-training降低任务复杂度，post-training将其进一步降低数个数量级

#### Problem & Contribution
- **解决的问题**:
  - 表面一致性假设(SAH)缺乏精确定义，导致不同论证和批评
  - pre-training和post-training的关系理解不清
  
- **核心贡献**:
  1. **任务复杂度定义**: 达到目标性能的最短程序长度
  2. **SAH新诠释**: pre-trained模型显著降低了许多任务的任务复杂度
  3. **post-training作用量化**: 相比pre-training，post-training将复杂度降低数个数量级（从GB级到KB级）

#### Method
- **任务复杂度定义**: 在预训练模型条件下，达到目标性能所需的最短程序长度
- **实验任务**: 数学推理、机器翻译、指令遵循
- **复杂度估计**: 估计各任务在pre-trained模型条件下的复杂度

#### Evidence
- **关键发现**:
  - pre-training提供了访问高性能的能力，但可能需要**GB级长度**的程序
  - post-training将到达相同性能的复杂度降低**数个数量级**
  - 任务适应往往只需要**几KB**的信息
- **理论意义**: 统一了之前支持SAH的不同论证，解释了为什么post-training有效

#### Takeaways
- **可以迁移到什么场景**:
  - 理解大模型训练和微调的理论基础
  - 设计更有效的post-training策略
  - 模型压缩和知识蒸馏研究
  
- **下一步动作**: 关注后续研究如何将这一理论洞察转化为实际算法改进

---

### 5. On Surprising Effectiveness of Masking Updates in Adaptive Optimizers

#### Meta
- **Title**: On Surprising Effectiveness of Masking Updates in Adaptive Optimizers
- **Link**: [arXiv:2602.15322](https://arxiv.org/abs/2602.15322)
- **Venue**: arXiv preprint
- **Date**: 2026-02-16
- **Source**: HuggingFace Daily Papers + arXiv cs.LG
- **Tags**: Efficient LLM, Adaptive Optimizers, Magma, Pre-training
- **推荐度**: ⭐⭐ 可选（Google出品、有理论洞察、但增量相对有限）
- **TL;DR**: 随机mask参数更新可诱导曲率相关的几何正则化，提出的Magma优化器在1B模型上相比Adam降低19%困惑度

#### Problem & Contribution
- **解决的问题**:
  - LLM预训练过度依赖复杂的自适应优化器
  - 优化器计算开销大
  
- **核心发现**:
  - 随机mask参数更新能有效平滑优化轨迹
  - 这种masking诱导**曲率相关的几何正则化**

- **主要贡献**:
  1. 揭示随机masking的理论机制
  2. 提出Momentum-aligned gradient masking (Magma)
  3. 1B模型上相比Adam降低19%困惑度，相比Muon降低9%

#### Method
- **方法结构/流程**:
  1. 在RMSProp基础上添加随机masking
  2. 使用momentum-gradient对齐调制masked更新
  3. 作为自适应优化器的drop-in替换

- **关键设计**:
  - **随机Masking**: 以一定概率mask参数更新
  - **Momentum对齐**: 利用momentum和gradient的对齐程度调制更新

#### Evidence
- **Benchmark / setting**: LLM预训练（1B模型）
- **对比对象**: Adam, Muon, RMSProp
- **关键结果**:
  - 相比Adam困惑度降低 **19%**
  - 相比Muon降低 **9%**
  - 计算开销可忽略
- **消融/局限**: 仅在1B规模验证，更大规模效果未知

#### Takeaways
- **可以迁移到什么场景**: LLM预训练、微调场景中的优化器选择
- **下一步动作**: 关注在大规模模型上的验证结果

---

### 6. Does Socialization Emerge in AI Agent Society? A Case Study of Moltbook

#### Meta
- **Title**: Does Socialization Emerge in AI Agent Society? A Case Study of Moltbook
- **Link**: [arXiv:2602.14299](https://arxiv.org/abs/2602.14299) | [Code](https://github.com/MingLiiii/Moltbook_Socialization)
- **Venue**: arXiv preprint
- **Date**: 2026-02-15
- **Source**: HuggingFace Daily Papers
- **Tags**: Multi-Agent, AI Society, Socialization
- **推荐度**: ⭐⭐ 可选（有趣的社会学视角，但偏应用/分析）
- **TL;DR**: 对Moltbook多智能体社会的大规模诊断发现：智能体保持个体多样性但缺乏相互影响和共识，规模化 alone 不足以产生社会化

#### Problem & Contribution
- **解决的问题**:
  - AI智能体社会是否会出现类似人类的社会化动态
  - 多智能体交互的长期演化规律
  
- **核心发现**:
  - 全局语义平均快速稳定，但个体保持高多样性
  - 智能体表现出强个体惯性，对交互伙伴适应性低
  - 影响是短暂的，没有出现持久超级节点
  - **规模和交互密度 alone 不足以诱导社会化**

- **主要贡献**:
  1. 首个大规模AI智能体社会的系统性诊断
  2. 提出量化演化框架（语义稳定、词汇更替、个体惯性等）
  3. 为下一代AI智能体社会提供设计原则

#### Method
- **方法结构/流程**:
  1. 在Moltbook平台观察智能体交互
  2. 测量语义稳定、词汇更替、个体惯性、影响持久性、集体共识
  3. 分析长期演化趋势

#### Evidence
- **关键结果**:
  - 动态平衡：全局稳定但个体多样
  - 缺乏共享社会记忆导致无法形成稳定集体影响锚点
- **局限**: 单一平台观察，结论普适性待验证

#### Takeaways
- **可以迁移到什么场景**: 多智能体系统设计、虚拟社会模拟
- **下一步动作**: 关注共享记忆机制对社会化涌现的影响研究

---

## 总结

| 论文 | 推荐度 | TL;DR | 下一步 |
|------|--------|-------|--------|
| GLM-5: from Vibe Coding to Agentic Engineering | ⭐⭐⭐ | DSA+异步RL实现Agentic Engineering范式转移，软件工程任务SOTA | 关注开源发布 |
| STAPO: Stabilizing RL for LLMs | ⭐⭐⭐ | 0.01% spurious tokens导致RL训练崩溃，masking解决+7.13% | 集成到训练流程 |
| RCE: Recursive Concept Evolution | ⭐⭐⭐ | 推理时动态演化表示空间，ARC-AGI-2提升12-18点 | 关注代码开源 |
| SAH via Task Complexity | ⭐⭐⭐ | 任务复杂度诠释pre/post-training关系，复杂度降低数个数量级 | 转化为算法改进 |
| Magma: Masking in Optimizers | ⭐⭐ | 随机masking诱导几何正则化，-19%困惑度 | 关注大规模验证 |
| Moltbook: AI Agent Society | ⭐⭐ | 规模化alone不足以产生社会化 | 关注共享记忆机制 |

**今日趋势观察**：
1. **Agentic RL与组合推理双轮驱动**: GLM-5代表Agentic能力系统级竞争，RCE代表组合推理表示学习突破，两者结合可能是下一代大模型的关键方向
2. **RL训练稳定性问题被根本性地揭示**: STAPO发现0.01% spurious tokens这一"阿喀琉斯之踵"，为RL微调提供了精准优化目标
3. **理论基础研究重新升温**: SAH via Task Complexity用程序复杂度重新诠释pre/post-training关系，为模型训练和微调提供了新的理论框架
4. **推理时动态计算成为新范式**: RCE在推理时动态演化表示空间，可能开启"测试时学习"的新研究方向

---

**数据来源**:
- HuggingFace Daily Papers: https://huggingface.co/papers/date/2026-02-18
- arXiv cs.AI: https://arxiv.org/list/cs.AI/recent (130 entries)
- arXiv cs.LG: https://arxiv.org/list/cs.LG/recent (119 entries)
- arXiv cs.CL: https://arxiv.org/list/cs.CL/recent (53 entries)

*Curated by Amy 🤖*
