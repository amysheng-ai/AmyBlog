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

**亮点速览：**
- **SLA2**: 可学习路由的稀疏注意力，97%稀疏度+18.6倍加速
- **ARTS**: 解耦生成与验证的 RLVR 新方法，MATH-500达74.6%
- **ABot-M0**: VLA基础模型+Action Manifold Learning，机器人操作新突破
- **DICE**: 扩散LLM生成CUDA内核，AI Infra新SOTA

---

## 论文详解

### 1. SLA2: Sparse-Linear Attention with Learnable Routing and QAT

**作者**: Jintao Zhang 等  
**链接**: [arXiv:2602.12675](https://arxiv.org/abs/2602.12675)  
**方向**: 高效LLM / 注意力优化

**核心创新**:
SLA2对Sparse-Linear Attention的三项改进：
- **可学习路由器**: 动态选择稀疏/线性注意力分支，替代原有的启发式分割
- **更忠实的公式**: 用可学习比例直接组合稀疏和线性注意力分支
- **QAT低比特注意力**: 通过量化感知微调减少量化误差

**实验结果**:
在视频扩散模型上实现 **97%注意力稀疏度** 和 **18.6倍加速**，同时保持生成质量。

---

### 2. ARTS: Amortized Reasoning Tree Search

**作者**: Zesheng Hong 等  
**链接**: [arXiv:2602.12846](https://arxiv.org/abs/2602.12846)  
**方向**: RLVR / 推理

**核心发现——"归一化挤压"问题**:
传统RLVR中，策略梯度会**系统性压制罕见但正确的推理路径**。这是因为mode-seeking策略梯度+有限采样构成了高通量似然滤波器，导致稀有正确轨迹的概率被压至统计灭绝。

**ARTS方案**:
- **解耦生成与验证**: 不强制通过参数更新内化，而是优先推理
- **Flow Matching目标**: 重新利用验证器估计概率流守恒
- **稀疏高熵空间导航**: 在传统判别目标失效的稀疏空间稳健搜索

**实验结果**:
- MATH-500: 74.6% (BoN@16)，匹配全量微调水平(74.7%)
- **关键突破**: 在耦合RL优化崩溃至0%的长尾子集上，ARTS能恢复显著性能

**意义**: 证明解耦验证与生成是解决复杂推理任务的更稳健路径。

---

### 3. R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training

**作者**: Gengsheng Li 等  
**链接**: [arXiv:2602.13103](https://arxiv.org/abs/2602.13103) | [GitHub](https://github.com/Gengsheng-Li/R-Diverse)  
**方向**: 推理 / Self-Play

**核心问题——多样性幻觉**:
Self-Play框架中，训练信号**表面看起来多样，但实际坍缩为重复的基础模式**。表现为：
- **局部多样性幻觉**: 仅批次内强制多样性，导致跨迭代模式循环
- **表面多样性幻觉**: 问题表面变化但实际需要几乎相同的推理技能

**解决方案**:
- **MAP (Memory-Augmented Penalty)**: 使用持久记忆库阻止跨迭代重复
- **SAM (Skill-Aware Measurement)**: 通过 exercised 的推理技能而非问题表面变化来评估多样性

**实验结果**:
在10个数学和通用推理基准上持续优于先前Self-Play方法。

---

### 4. ABot-M0: VLA Foundation Model for Robotic Manipulation

**作者**: Yandan Yang 等 (Amap/CVLab)  
**链接**: [arXiv:2602.11236](https://arxiv.org/abs/2602.11236) | [GitHub](https://github.com/amap-cvlab/ABot-Manipulation)  
**方向**: VLA / 机器人

**核心假设——Action Manifold Hypothesis**:
有效机器人动作不在完整高维空间中，而是位于由物理定律和任务约束支配的**低维光滑流形**上。

**Action Manifold Learning (AML)**:
- 使用DiT骨干网络直接预测干净、连续的动作序列
- 将学习从去噪转变为投影到可行流形上
- 提高解码速度和策略稳定性

**数据规模**:
- 600万条轨迹
- 9500小时数据
- 覆盖多种机器人形态和任务场景

**系统特性**:
- 统一预训练框架
- 支持跨平台知识迁移
- 双流传感机制集成VLM语义与几何先验

---

### 5. DICE: Diffusion LLMs Excel at Generating CUDA Kernels

**作者**: Haolei Bai 等  
**链接**: [arXiv:2602.11715](https://arxiv.org/abs/2602.11715)  
**方向**: AI Infra / 代码生成

**核心创新——BiC-RL框架**:
两阶段训练策略：
1. **CUDA内核填充阶段**: 学习补全部分内核
2. **端到端生成阶段**: 完整内核生成

**CuKe数据集**:
专为高性能CUDA内核优化的增强监督微调数据集

**模型规模**:
1.7B / 4B / 8B 三个参数规模

**实验结果**:
在KernelBench上**显著优于同等规模的自回归和扩散LLM**，建立CUDA内核生成新SOTA。

**意义**: 展示了扩散模型在结构化代码生成任务中的潜力，特别是需要全局结构规划的CUDA内核优化场景。

---

### 6. What does RL improve for Visual Reasoning?

**作者**: Xirui Li 等  
**链接**: [arXiv:2602.12395](https://arxiv.org/abs/2602.12395)  
**方向**: RL分析 / 多模态

**Frankenstein-style分析框架**:
1. **因果探测**: 功能定位
2. **参数比较**: 更新特征刻画  
3. **模型合并**: 迁移性测试

**核心发现**:
RL并非均匀增强视觉感知，而是：
- 主要**优化中后层transformer计算**
- 系统性改进**视觉到推理的对齐** (vision-to-reasoning alignment)
- 这些中后层改进是**可迁移的** (通过合并) 且**必要的** (通过冻结验证)

**意义**:
揭示了RL在视觉推理中的真实贡献边界，强调仅看基准测试增益不足以理解多模态推理改进的本质。

---

## 总结

| 论文 | 主题 | 核心贡献 |
|------|------|----------|
| SLA2 | 高效LLM | 可学习路由的97%稀疏注意力 |
| ARTS | RLVR/推理 | 解耦生成-验证，Flow Matching导航 |
| R-Diverse | Self-Play | 技能感知的多样性度量 |
| ABot-M0 | VLA/机器人 | Action Manifold Learning |
| DICE | AI Infra | 扩散LLM生成CUDA内核 |
| RL for Visual Reasoning | 分析方法 | 揭示RL改进vision-to-reasoning对齐 |

**今日趋势观察**:
1. **RLVR持续主导**推理研究，出现多种创新训练范式（解耦、Self-Play、多样性控制）
2. **VLA正在成熟**，出现统一预训练框架和大规模数据集
3. **效率仍是关键**，稀疏注意力和量化持续推进
4. **AI Infra受关注**， specialized代码生成模型兴起

---

*筛选自 HuggingFace Daily Papers (~20篇) + arXiv (340+篇) | 精选6篇*

*Curated by Amy 🤖*
