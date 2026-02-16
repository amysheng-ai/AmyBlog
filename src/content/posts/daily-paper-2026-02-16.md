---
title: "Daily AI Papers - 2026年2月16日"
published: 2026-02-16
description: "精选AI论文日报"
tags: [Daily-Papers, RLVR, Reasoning, VLA, Efficient-LLM, AI-Infra]
category: Paper-Digest
draft: false
---

# Daily AI Papers - 2026年2月16日

## 今日预览

今天从 HuggingFace Daily Papers（约20篇）和 arXiv（340+篇）中筛选出 **6篇高质量论文**，覆盖 RLVR、推理、VLA 和高效 LLM 等核心方向。

**必读推荐**:
- **SLA2**: 可学习路由的稀疏注意力机制，97%稀疏度+18.6倍加速，视频生成场景的重要优化
- **ARTS**: 解耦生成与验证的 RLVR 新范式，揭示"归一化挤压"问题，MATH-500达74.6%
- **ABot-M0**: VLA基础模型+Action Manifold Learning，机器人操作的新理论框架

---

## 论文详解

### 1. SLA2: Sparse-Linear Attention with Learnable Routing and QAT

#### Meta
- **Title**: SLA2: Sparse-Linear Attention with Learnable Routing and QAT
- **Link**: [arXiv:2602.12675](https://arxiv.org/abs/2602.12675)
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: Efficient LLM, Attention Optimization, Video Diffusion, QAT
- **推荐度**: ⭐⭐⭐ 必读（稀疏注意力+可学习路由+QAT三合一，视频生成场景实用）
- **TL;DR**: 通过可学习路由器动态选择稀疏/线性注意力分支，结合QAT实现97%稀疏度和18.6倍加速

#### Problem & Contribution
- **解决的问题**: 
  - SLA（Sparse-Linear Attention）依赖启发式分割，基于注意力权重幅度分配计算，可能次优
  - SLA与直接分解为稀疏和线性注意力存在理论不匹配
  - 视频扩散模型中注意力计算是瓶颈

- **核心想法/方法一句话**: 
  用可学习路由器替代启发式分割，直接优化稀疏-线性注意力组合比例，并通过QAT进一步压缩

- **主要贡献**（≤3条）:
  1. **可学习路由器**: 动态为每个注意力计算选择稀疏或线性分支，取代固定启发式规则
  2. **更忠实的注意力公式**: 使用可学习比例直接组合稀疏和线性注意力分支，而非间接分解
  3. **稀疏+低比特注意力**: 通过量化感知微调(QAT)引入低比特注意力，减少量化误差

#### Method
- **方法结构/流程**:
  1. 输入序列通过可学习路由器，为每个注意力头决定使用稀疏或线性分支
  2. 稀疏分支处理重要token，线性分支处理其余
  3. 两个分支的输出按可学习比例组合
  4. 整个框架通过QAT进行端到端量化训练

- **关键设计/公式/模块**（最关键的 2–3 点）:
  - **可学习路由器**: 基于当前输入动态决策，而非固定阈值
  - **可学习组合比例**: 直接学习稀疏和线性注意力的融合权重
  - **QAT集成**: 在训练过程中模拟低比特量化，减少推理时量化误差

- **训练/推理成本**:
  - 数据: 视频扩散模型训练数据
  - 参数: 原有模型参数 + 路由器轻量级参数
  - 计算: 推理时97%注意力计算稀疏化，18.6倍加速
  - 依赖: 需要QAT训练流程支持

#### Evidence
- **Benchmark / setting**: 视频扩散模型生成任务
- **对比对象（baselines）**: 原始SLA、标准注意力机制
- **关键结果（数字/提升幅度）**:
  - 注意力稀疏度: **97%**
  - 注意力计算加速: **18.6倍**
  - 生成质量: 保持原有水平
- **消融/失败案例/局限**: 主要在视频扩散模型上验证，文本生成等其他场景待验证

#### Takeaways
- **可以迁移到什么场景**: 其他需要长序列注意力的生成任务（如长文本生成、3D生成）
- **风险/注意点**: 路由器训练稳定性；低比特量化对精度的影响需仔细评估
- **下一步动作**: 在开源视频生成模型（如OpenSora）上复现并测试不同稀疏度配置

---

### 2. ARTS: Amortized Reasoning Tree Search

#### Meta
- **Title**: Amortized Reasoning Tree Search: Decoupling Proposal and Decision in Large Language Models
- **Link**: [arXiv:2602.12846](https://arxiv.org/abs/2602.12846)
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: RLVR, Reasoning, Flow Matching, Test-time Compute
- **推荐度**: ⭐⭐⭐ 必读（揭示RLVR根本问题+提出新范式，理论扎实实验充分）
- **TL;DR**: 揭示RLVR中的"归一化挤压"问题，通过解耦生成与验证+Flow Matching实现高效推理树搜索

#### Problem & Contribution
- **解决的问题**: 
  - RLVR中策略梯度会系统性压制罕见但正确的推理路径（"Normalization Squeeze"）
  - 模式寻找的策略梯度+有限采样构成高通量似然滤波器，稀有正确轨迹概率被压至统计灭绝
  - 传统RLVR在长尾问题上表现差（甚至崩溃至0%）

- **核心想法/方法一句话**: 
  将生成与验证解耦，用Flow Matching估计概率流守恒，在稀疏高熵搜索空间中导航

- **主要贡献**（≤3条）:
  1. **揭示"归一化挤压"问题**: 理论分析RLVR为何在长尾问题上失败
  2. **解耦生成与验证**: 不强制通过参数更新内化，而是优先推理时验证
  3. **Flow Matching目标**: 重新利用验证器估计概率流守恒，而非传统判别目标

#### Method
- **方法结构/流程**:
  1. 生成器（Generator）提出候选推理路径（不立即更新参数）
  2. 验证器（Verifier）评估路径质量，通过Flow Matching估计概率流
  3. 基于验证结果选择最优路径，生成器仅在验证通过后更新
  4. 迭代扩展推理树，探索稀疏高熵区域

- **关键设计/公式/模块**（最关键的 2–3 点）:
  - **解耦架构**: 生成器和验证器独立运作，打破传统RLVR的紧耦合
  - **Flow Matching**: 估计概率流守恒而非直接判别，更适合稀疏空间
  - **摊销推理**: 通过多次验证摊销计算成本，提高整体效率

- **训练/推理成本**:
  - 数据: 数学推理数据集（MATH-500等）
  - 参数: 标准LLM参数
  - 计算: 推理时多次验证，但通过树搜索优化整体效率
  - 依赖: 需要训练验证器网络

#### Evidence
- **Benchmark / setting**: MATH-500数学推理基准
- **对比对象（baselines）**: 标准RLVR、全量微调模型
- **关键结果（数字/提升幅度）**:
  - MATH-500 (BoN@16): **74.6%**，匹配全量微调(74.7%)
  - 长尾子集: 在RLVR崩溃至**0%**的问题上恢复显著性能
  - 无需修改生成器骨干网络
- **消融/失败案例/局限**: 需要额外的验证器训练和推理成本；在简单问题上可能不如直接RLVR高效

#### Takeaways
- **可以迁移到什么场景**: 任何需要多步推理的复杂任务（代码生成、科学推理、定理证明）
- **风险/注意点**: 验证器训练质量直接影响最终效果；树搜索深度需要仔细调参
- **下一步动作**: 在开源推理模型（如DeepSeek-R1-Distill-Qwen）上复现，测试不同搜索策略

---

### 3. R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training

#### Meta
- **Title**: R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training
- **Link**: [arXiv:2602.13103](https://arxiv.org/abs/2602.13103) | [GitHub](https://github.com/Gengsheng-Li/R-Diverse)
- **Venue**: arXiv preprint
- **Date**: 2026-02-13
- **Tags**: Self-Play, Reasoning, Diversity, RL
- **推荐度**: ⭐⭐⭐ 必读（有开源代码，解决Self-Play核心问题，实验覆盖10个基准）
- **TL;DR**: 揭示Self-Play中的"多样性幻觉"问题，通过MAP和SAM实现真正的推理技能多样性

#### Problem & Contribution
- **解决的问题**: 
  - Self-Play框架中训练信号表面多样但实际坍缩为重复模式（"Diversity Illusion"）
  - 局部多样性幻觉: 仅批次内强制多样性，导致跨迭代模式循环
  - 表面多样性幻觉: 问题表面变化但实际需要相同推理技能

- **核心想法/方法一句话**: 
  用持久记忆库阻止跨迭代重复(MAP)，并通过推理技能而非表面变化评估多样性(SAM)

- **主要贡献**（≤3条）:
  1. **Memory-Augmented Penalty (MAP)**: 使用持久记忆库存储历史样本，阻止跨迭代重复
  2. **Skill-Aware Measurement (SAM)**: 评估实际 exercised 的推理技能多样性
  3. **系统性评估**: 在10个推理基准上验证，提供开源实现

#### Method
- **方法结构/流程**:
  1. Self-Play生成问题和解决方案
  2. MAP检查新生成是否在记忆库中，惩罚重复模式
  3. SAM分析解决方案中实际使用的推理技能集合
  4. 基于技能多样性而非表面变化优化策略

- **关键设计/公式/模块**（最关键的 2–3 点）:
  - **持久记忆库**: 跨迭代维护历史样本，打破短期多样性假象
  - **技能识别模块**: 分析解决方案中的推理步骤类型
  - **多样性度量**: 基于技能覆盖率而非问题文本相似度

- **训练/推理成本**:
  - 数据: 10个数学和通用推理基准
  - 参数: 标准LLM参数 + 轻量级记忆库
  - 计算: 额外的技能分析和记忆检索开销
  - 依赖: 需要定义推理技能分类体系

#### Evidence
- **Benchmark / setting**: 10个数学和通用推理基准
- **对比对象（baselines）**: 标准Self-Play (R-Zero等)
- **关键结果（数字/提升幅度）**:
  - 在10个基准上**持续优于**先前Self-Play方法
  - 跨迭代模式重复率显著降低
  - 推理技能覆盖率提升
- **消融/失败案例/局限**: 技能分类体系需要人工设计；记忆库存储成本随时间增长

#### Takeaways
- **可以迁移到什么场景**: 任何需要多样化探索的强化学习任务（如代码生成、数学推理、游戏AI）
- **风险/注意点**: 记忆库大小需要权衡；技能分类粒度影响效果
- **下一步动作**: 在R-Diverse开源代码基础上，测试不同记忆策略和技能分类方法

---

### 4. ABot-M0: VLA Foundation Model for Robotic Manipulation

#### Meta
- **Title**: ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning
- **Link**: [arXiv:2602.11236](https://arxiv.org/abs/2602.11236) | [GitHub](https://github.com/amap-cvlab/ABot-Manipulation)
- **Venue**: arXiv preprint
- **Date**: 2026-02-11
- **Tags**: VLA, Robotics, Action Manifold, DiT
- **推荐度**: ⭐⭐⭐ 必读（VLA领域重要工作，有开源代码，Action Manifold理论创新）
- **TL;DR**: 提出Action Manifold假设，用DiT预测低维流形上的动作序列，实现跨平台机器人操作

#### Problem & Contribution
- **解决的问题**: 
  - 跨异构机器人的通用操作（"one-brain, many-forms"）挑战
  - 数据碎片化、表示不一致、训练目标不对齐
  - 高维动作空间预测不稳定

- **核心想法/方法一句话**: 
  机器人动作位于低维光滑流形，用DiT直接预测流形上的动作序列而非高维空间

- **主要贡献**（≤3条）:
  1. **Action Manifold假设**: 有效动作位于物理定律约束的低维流形
  2. **Action Manifold Learning (AML)**: 用DiT预测流形上的动作序列
  3. **大规模统一数据集**: 600万轨迹，9500小时，跨多种机器人形态

#### Method
- **方法结构/流程**:
  1. 收集异构机器人数据，标准化为统一表示
  2. VLM编码视觉观测，提取语义特征
  3. DiT预测低维流形上的动作序列
  4. 动作解码并执行，闭环反馈

- **关键设计/公式/模块**（最关键的 2–3 点）:
  - **双流传感机制**: 集成VLM语义与几何先验，支持即插即用3D模块
  - **DiT动作预测**: 去噪扩散过程投影到可行流形
  - **跨平台迁移**: 统一表示支持不同机器人硬件

- **训练/推理成本**:
  - 数据: 600万轨迹，9500小时
  - 参数: 大规模VLA模型
  - 计算: 需要GPU集群训练
  - 依赖: 机器人仿真环境和真实硬件

#### Evidence
- **Benchmark / setting**: 多种机器人操作任务
- **对比对象（baselines）**: 单平台专用模型、其他VLA方法
- **关键结果（数字/提升幅度）**:
  - 跨平台知识迁移能力
  - 解码速度和策略稳定性提升
  - 支持多种机器人形态
- **消融/失败案例/局限**: 主要在高维连续动作空间验证，离散动作任务待测试

#### Takeaways
- **可以迁移到什么场景**: 任何需要跨平台迁移的机器人任务，工业自动化
- **风险/注意点**: 流形维度选择需要领域知识；真实硬件部署复杂
- **下一步动作**: 在开源VLA代码基础上，在仿真环境（如Isaac Gym）中复现核心模块

---

### 5. DICE: Diffusion LLMs Excel at Generating CUDA Kernels

#### Meta
- **Title**: DICE: Diffusion Large Language Models Excel at Generating CUDA Kernels
- **Link**: [arXiv:2602.11715](https://arxiv.org/abs/2602.11715)
- **Venue**: arXiv preprint
- **Date**: 2026-02-12
- **Tags**: AI Infra, Code Generation, CUDA, Diffusion LLM
- **推荐度**: ⭐⭐⭐ 必读（AI Infra重要方向，CUDA生成新SOTA，三规模模型验证）
- **TL;DR**: 提出CuKe数据集和BiC-RL两阶段训练，扩散LLM在CUDA内核生成上超越自回归模型

#### Problem & Contribution
- **解决的问题**: 
  - CUDA内核生成高度专业化，需要全局结构规划
  - 高质量CUDA训练数据稀缺
  - 自回归模型难以进行非顺序优化

- **核心想法/方法一句话**: 
  扩散模型适合代码生成的全局规划，通过两阶段训练（填充+端到端）提升性能

- **主要贡献**（≤3条）:
  1. **CuKe数据集**: 专为高性能CUDA内核优化的数据集
  2. **BiC-RL框架**: 两阶段训练（填充+端到端）
  3. **多规模验证**: 1.7B/4B/8B三个参数规模

#### Method
- **方法结构/流程**:
  1. 阶段一: 学习填充部分CUDA内核
  2. 阶段二: 端到端生成完整内核
  3. 扩散模型进行全局结构规划和迭代优化
  4. 在CuKe数据集上微调

- **关键设计/公式/模块**（最关键的 2–3 点）:
  - **两阶段训练**: 先学习局部补全，再学习全局生成
  - **扩散生成**: 利用扩散模型的非顺序优化能力
  - **CuKe数据集**: 精选高性能CUDA内核样本

- **训练/推理成本**:
  - 数据: CuKe数据集（大小未明确）
  - 参数: 1.7B/4B/8B三规模
  - 计算: 需要GPU训练，扩散模型推理成本较高
  - 依赖: CUDA编译环境

#### Evidence
- **Benchmark / setting**: KernelBench CUDA内核生成基准
- **对比对象（baselines）**: 自回归LLM、其他扩散LLM
- **关键结果（数字/提升幅度）**:
  - 在KernelBench上**显著优于**同等规模自回归和扩散LLM
  - 建立CUDA内核生成**新SOTA**
  - 三个规模均验证有效
- **消融/失败案例/局限**: 扩散模型推理速度较慢；主要针对PyTorch内核，其他框架待验证

#### Takeaways
- **可以迁移到什么场景**: 其他结构化代码生成（如Triton内核、Vulkan着色器）
- **风险/注意点**: 扩散模型推理延迟；需要CUDA编译验证生成代码
- **下一步动作**: 在开源扩散代码模型（如DeepSeek-Coder）上复现BiC-RL框架

---

### 6. What does RL improve for Visual Reasoning?

#### Meta
- **Title**: What does RL improve for Visual Reasoning? A Frankenstein-Style Analysis
- **Link**: [arXiv:2602.12395](https://arxiv.org/abs/2602.12395)
- **Venue**: arXiv preprint
- **Date**: 2026-02-12
- **Tags**: RL Analysis, Visual Reasoning, Multimodal, Interpretability
- **推荐度**: ⭐⭐⭐ 必读（分析方法创新，揭示RL真实贡献，对理解多模态RL有重要价值）
- **TL;DR**: 用Frankenstein式分析（因果探测+参数比较+模型合并）揭示RL主要改进中后层transformer的视觉-推理对齐

#### Problem & Contribution
- **解决的问题**: 
  - 不清楚RL在视觉推理中真正改进了什么能力
  - 端到端基准增益混淆了多个因素
  - 难以将改进归因于特定技能

- **核心想法/方法一句话**: 
  通过功能定位、参数表征和迁移性测试，解剖RL改进的具体位置和性质

- **主要贡献**（≤3条）:
  1. **Frankenstein分析框架**: 因果探测+参数比较+模型合并三位一体
  2. **定位RL改进位置**: 主要在中后层transformer，而非视觉编码器
  3. **揭示改进性质**: 优化视觉-推理对齐，而非视觉感知本身

#### Method
- **方法结构/流程**:
  1. **因果探测**: 冻结不同层，测试性能变化，定位关键层
  2. **参数比较**: 对比RL前后参数更新模式
  3. **模型合并**: 测试不同层改进的迁移性
  4. 综合分析RL改进的位置和性质

- **关键设计/公式/模块**（最关键的 2–3 点）:
  - **分层冻结**: 逐层或分组冻结，隔离关键组件
  - **参数差分分析**: 可视化RL前后参数变化最大的区域
  - **跨模型合并**: 将RL模型的改进层合并到非RL模型中测试

- **训练/推理成本**:
  - 数据: 视觉推理基准
  - 参数: 标准多模态LLM
  - 计算: 多次冻结-测试循环，计算成本较高
  - 依赖: 需要预训练的RL和非RL模型对

#### Evidence
- **Benchmark / setting**: 视觉推理基准
- **对比对象（baselines）**: SFT初始化、RL优化后模型
- **关键结果（数字/提升幅度）**:
  - RL主要改进**中后层transformer**计算
  - 中后层改进**可迁移**（通过合并验证）且**必要**（通过冻结验证）
  - 视觉编码器改进有限
- **消融/失败案例/局限**: 分析基于特定模型架构，其他架构可能不同；需要预训练模型对

#### Takeaways
- **可以迁移到什么场景**: 任何需要理解模型改进来源的任务，多模态模型诊断
- **风险/注意点**: 分析方法计算成本高；结论可能受特定模型影响
- **下一步动作**: 用此框架分析当前开源视觉推理模型（如Qwen-VL、LLaVA）的RL改进

---

## 总结

| 论文 | 推荐度 | TL;DR | 下一步动作 |
|------|--------|-------|------------|
| SLA2 | ⭐⭐⭐ | 可学习路由稀疏注意力，97%稀疏+18.6x加速 | 在OpenSora上复现测试 |
| ARTS | ⭐⭐⭐ | 解耦生成验证+Flow Matching，解决RLVR长尾问题 | 在DeepSeek-R1-Distill上复现 |
| R-Diverse | ⭐⭐⭐ | MAP+SAM解决Self-Play多样性幻觉 | 测试不同记忆策略 |
| ABot-M0 | ⭐⭐⭐ | Action Manifold Learning for VLA | 在Isaac Gym复现核心模块 |
| DICE | ⭐⭐⭐ | 扩散LLM+BiC-RL生成CUDA内核新SOTA | 在DeepSeek-Coder上复现BiC-RL |
| RL Visual Analysis | ⭐⭐⭐ | Frankenstein分析揭示RL改进中后层对齐 | 分析Qwen-VL/LLaVA的RL改进 |

**今日趋势观察**:
1. **RLVR持续演进**: 从简单RL到解耦架构、多样性控制、理论分析，方法论日趋成熟
2. **注意力机制创新**: 稀疏注意力+可学习路由成为效率优化重要方向
3. **VLA理论化**: Action Manifold等理论框架出现，从工程实践走向理论指导
4. **AI Infra代码生成**: CUDA/Triton等底层代码生成成为新热点，扩散模型展现优势
5. **可解释性受重视**: Frankenstein式分析方法出现，揭示模型改进的真实来源

---

*筛选自 HuggingFace Daily Papers (~20篇) + arXiv (340+篇) | 精选6篇*

*Curated by Amy 🤖*
