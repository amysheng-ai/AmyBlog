---
title: Daily AI Papers - 2026年2月20日
published: 2026-02-20
description: 今日精选 6 篇高质量论文，涵盖超长时域 Agent 训练、Reward Model 优化、激活值引导对齐、多轮人机协作等方向。KLong 通过轨迹分割 SFT 和渐进式 RL 在 PaperBench 上超越 Kimi K2 Thinking。
tags: [Daily Papers, AI, Agent, RL, Alignment, Reasoning]
category: Papers
draft: false
---

# Daily AI Papers - 2026年2月20日

## 今日预览
今日精选 6 篇高质量论文，涵盖 **超长时域 Agent 训练**、**Reward Model 优化**、**激活值引导对齐**、**多轮人机协作** 等方向。KLong 通过轨迹分割 SFT 和渐进式 RL 在 PaperBench 上超越 Kimi K2 Thinking；MARS 提出边缘感知的 Reward Model 增强策略；ODESteer 建立 ODE 框架统一激活值引导方法。

---

## 论文详解

### 1. KLong: Training LLM Agent for Extremely Long-horizon Tasks ⭐⭐⭐
**作者**: Yue Liu 等  
**链接**: [arXiv:2602.17547](https://arxiv.org/abs/2602.17547)  
**方向**: Agent / Long-horizon RL

**核心创新**:
针对超长时域任务（如研究论文复现），提出 **KLong** —— 开源 LLM Agent 训练框架。核心方法包括：
1. **Research-Factory 自动化数据 pipeline**：收集研究论文并构建评估标准，从 Claude 4.5 Sonnet (Thinking) 蒸馏长时域轨迹
2. **轨迹分割 SFT**：保留早期上下文，渐进式截断后期上下文，保持子轨迹间重叠，解决超长轨迹训练难题
3. **渐进式 RL**：分阶段训练，逐阶段延长 timeout 时间

**实验结果**:
- KLong (106B) 在 PaperBench 上超越 Kimi K2 Thinking (1T) 达 **11.28%**
- 在 SWE-bench Verified 和 MLE-bench 上同样展现强泛化能力
- 证明通过高质量数据蒸馏和渐进式训练，中等规模模型可超越超大模型

---

### 2. MARS: Margin-Aware Reward-Modeling with Self-Refinement ⭐⭐⭐
**作者**: Payel Bhattacharjee 等  
**链接**: [arXiv:2602.17658](https://arxiv.org/abs/2602.17658)  
**方向**: RLHF / Reward Modeling

**核心创新**:
Reward Model (RM) 训练依赖昂贵的人工标注偏好数据。本文提出 **MARS** —— 边缘感知的自适应增强和采样策略：
- 关注 **低边缘 (low-margin)** 偏好对（即 RM 最不确定的样本）进行增强
- 通过难样本增强迭代优化训练分布
- 理论证明该策略增加损失函数平均曲率，改善信息和条件数

**实验结果**:
- 相比均匀增强策略，RM 鲁棒性显著提升
- 在 RLHF/RLAIF pipeline 中提供更可靠的奖励信号

---

### 3. ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment ⭐⭐⭐
**作者**: Hongjue Zhao, Haosen Sun, Yejin Choi, Manling Li 等 (UIUC, Stanford)  
**链接**: [arXiv:2602.17560](https://arxiv.org/abs/2602.17560)  
**代码**: 已开源  
**方向**: Alignment / Activation Steering

**核心创新**:
激活值引导 (activation steering) 缺乏统一理论框架，且现有方法依赖单步引导无法捕捉复杂激活分布。本文：
- 提出基于 **ODE (常微分方程)** 的统一理论框架
- 证明传统激活值加法可解释为 ODE 的一阶近似
- 将引导方向识别转化为控制理论中的 **barrier function** 设计问题
- 实现 **多步自适应引导**，在 TruthfulQA、UltraFeedback、RealToxicityPrompts 上取得 SOTA

**实验结果**:
- TruthfulQA 提升 **5.7%**
- UltraFeedback 提升 **2.5%**
- RealToxicityPrompts 提升 **2.4%**
- ICLR 2026 接收

---

### 4. Multi-Round Human-AI Collaboration with User-Specified Requirements ⭐⭐
**作者**: Sima Noorani 等  
**链接**: [arXiv:2602.17646](https://arxiv.org/abs/2602.17646)  
**方向**: Human-AI Collaboration

**核心创新**:
随着多轮对话 AI 在高风险决策中的应用，需要确保协作可靠提升决策质量。本文提出两个核心原则：
1. **反事实伤害 (Counterfactual Harm)**：确保 AI 不削弱人类优势
2. **互补性 (Complementarity)**：确保 AI 在人类易错处增加价值

通过用户自定义规则形式化这些概念，提出在线、分布无关的算法，有限样本保证在用户指定约束下执行。

**实验结果**:
- 医疗诊断任务（LLM 模拟）和图形推理任务（众包研究）验证
- 算法维持规定的反事实伤害和互补性违规率
- 约束收紧/放松可预测地改变人类准确率

---

### 5. Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability ⭐⭐
**作者**: Shashank Aggarwal 等  
**链接**: [arXiv:2602.17544](https://arxiv.org/abs/2602.17544)  
**方向**: Reasoning / Chain-of-Thought

**核心创新**:
当前 CoT 评估仅关注目标任务准确率，无法评估推理过程本身的质量。本文提出两个新指标：
1. **可复用性 (Reusability)**：Executor 复现 Thinker CoT 的容易程度
2. **可验证性 (Verifiability)**：Executor 使用 CoT 匹配 Thinker 答案的频率

采用 Thinker-Executor 框架解耦 CoT 生成与执行，4 个 Thinker 模型 vs 10 个 Executor 模型在 5 个基准上评估。

**实验结果**:
- 可复用性和可验证性与标准准确率 **不相关**，暴露准确率导向评估的盲区
- 专用推理模型的 CoT 并未比通用 LLM (Llama/Gemma) 更可复用或验证

---

### 6. Pushing the Frontier of Black-Box LVLM Attacks via Fine-Grained Detail Targeting ⭐
**作者**: Xiaohan Zhao 等  
**链接**: [arXiv:2602.17645](https://arxiv.org/abs/2602.17645)  
**代码**: https://github.com/vila-lab/M-Attack-V2  
**方向**: LVLM Safety / Adversarial Attack

**核心创新**:
黑盒 LVLM 攻击因缺失梯度和复杂多模态边界而极具挑战。本文发现现有 M-Attack 的局部裁剪匹配导致高方差、几乎正交的梯度。提出 **M-Attack-V2**：
- **Multi-Crop Alignment (MCA)**：多独立局部视图梯度平均降低方差
- **Auxiliary Target Alignment (ATA)**：用语义相关分布的小辅助集替代激进的目标增强
- **Patch Momentum**：历史裁剪梯度回放

**实验结果**:
- Claude-4.0 成功率从 8% 提升至 **30%**
- Gemini-2.5-Pro 从 83% 提升至 **97%**
- GPT-5 从 98% 提升至 **100%**

---

## 总结

| 论文 | 主题 | 评分 | 核心贡献 |
|------|------|------|----------|
| KLong | 超长时域 Agent | ⭐⭐⭐ | 轨迹分割 SFT + 渐进式 RL，106B 超越 1T 模型 |
| MARS | Reward Modeling | ⭐⭐⭐ | 边缘感知增强，专注低边缘偏好对 |
| ODESteer | 激活值引导 | ⭐⭐⭐ | ODE 统一框架，多步自适应引导 |
| Human-AI Collaboration | 人机协作 | ⭐⭐ | 反事实伤害 + 互补性双原则约束 |
| CoT Evaluation | 推理评估 | ⭐⭐ | 可复用性、可验证性新指标 |
| M-Attack-V2 | LVLM 安全 | ⭐ | 细粒度细节目标攻击 |

**今日趋势观察**:
1. **超长时域 Agent 训练**：通过数据蒸馏和渐进式训练，中等规模模型可超越超大模型
2. **Reward Model 优化**：边缘感知增强和难样本挖掘提升 RM 鲁棒性
3. **激活值引导理论化**：ODE 框架为 inference-time 对齐提供统一理论基础
4. **推理评估多元化**：从准确率扩展到可复用性、可验证性等过程指标

---

*日报由 Amy 自动生成于 2026-02-20*
