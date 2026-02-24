---
title: Daily AI Papers - 2026年02月24日
published: 2026-02-24
description: 今日精选8篇高质量AI论文，涵盖Decoding优化、高效推理、模型压缩、Agentic RL、Neurosymbolic Reasoning、Cross-Embodiment RL等领域。必读推荐包括Best-of-K解码框架、RAT+稀疏推理、OMAD多智能体扩散策略。
tags: [Daily Papers, AI, Decoding, Efficient LLM, Agentic RL, Reasoning, Robotics, Model Compression]
category: Papers
draft: false
---

# Daily AI Papers - 2026年02月24日

## 今日预览

今日从 arXiv 筛选出 **8篇高质量论文**，涵盖 **Decoding优化**、**高效推理**、**模型压缩**、**Agentic RL**、**Neurosymbolic Reasoning**、**Cross-Embodiment RL** 和 **长时程Agent系统**。今日 HuggingFace Daily Papers 因网络问题暂未获取，日报基于 arXiv cs.AI/cs.LG/cs.CL 更新。

**必读推荐**：
- **Decoding as Optimisation on the Probability Simplex**: 统一解码框架，Best-of-K采样器在MATH500上带来+18.6%提升
- **RAT+**: "训练密集、推理稀疏"新范式，16倍稀疏接近密集准确率
- **Diffusing to Coordinate**: 首个在线多智能体扩散策略框架，样本效率提升2.5-5倍
- **Cross-Embodiment Offline RL**: ICLR 2026，16种机器人平台的跨具身离线RL系统研究

---

## 论文详解

### 1. Decoding as Optimisation on the Probability Simplex: From Top-K to Top-P (Nucleus) to Best-of-K Samplers
**作者**: Xiaotong Ji 等  
**链接**: [arXiv:2602.18292](https://arxiv.org/abs/2602.18292)  
**方向**: Decoding / Reasoning

**核心创新**：
该研究将解码重新定义为概率单纯形上的正则化优化问题，统一了贪婪解码、Softmax采样、Top-K、Top-P和Sparsemax等方法。作者提出 **Best-of-K (BoK) 采样器**，通过KL锚定的覆盖率目标，在固定K样本预算内最大化覆盖优质备选答案的概率。

**实验结果**：
- Qwen2.5-Math-7B on MATH500: **+18.6% accuracy** at high temperature
- 统一框架解释了现有解码方法的共同结构

**评价**: ⭐⭐⭐ 必读 — 为解码策略提供了理论统一的视角，BoK采样器对推理任务有显著效果提升。

---

### 2. RAT+: Train Dense, Infer Sparse -- Recurrence Augmented Attention for Dilated Inference
**作者**: Xiuying Wei 等  
**链接**: [arXiv:2602.18196](https://arxiv.org/abs/2602.18196)  
**方向**: Efficient LLM / Attention

**核心创新**：
RAT+ 解决了结构化稀疏注意力在推理时的准确率下降问题。通过在注意力中增强全序列循环机制，模型只需一次密集预训练，即可在推理时灵活切换到不同膨胀率的稀疏注意力模式，仅需 **1B token的短适应** 即可恢复性能。

**实验结果**：
- 1.5B参数，100B tokens训练
- **16倍稀疏**: 接近密集准确率
- **64倍稀疏**: 仅下降2-3个百分点
- 在常识推理和LongBench上超越top-k块注意力

**评价**: ⭐⭐⭐ 必读 — 提出了"训练密集、推理稀疏"的实用范式，对部署大规模模型具有重要价值。

---

### 3. Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies
**作者**: Zhuoran Li 等  
**链接**: [arXiv:2602.18291](https://arxiv.org/abs/2602.18291)  
**方向**: Agentic RL / Multi-Agent RL

**核心创新**：
扩散模型在表达性方面展现巨大潜力，但将其应用于在线多智能体强化学习(MARL)面临核心障碍：扩散模型的似然难以计算，阻碍了基于熵的探索和协调。本文提出 **OMAD** (Online off-policy MARL with Diffusion policies)，这是首个在线MARL扩散策略框架。

关键创新包括：
1. **松弛策略目标**：最大化缩放联合熵，无需依赖可计算似然即可实现有效探索
2. **联合分布价值函数**：在CTDE范式下，利用可计算熵增强目标引导扩散策略同步更新，确保稳定协调

**实验结果**：
在MPE和MAMuJoCo基准上进行广泛评估，在10个多样化任务上建立了新的SOTA。与现有方法相比，样本效率提升 **2.5倍到5倍**。

**评价**: ⭐⭐⭐ 必读 — 多智能体扩散策略的开创性工作，样本效率提升显著。

---

### 4. Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets
**作者**: Haruki Abe 等 (ICLR 2026)  
**链接**: [arXiv:2602.18025](https://arxiv.org/abs/2602.18025)  
**方向**: Offline RL / Robotics / Cross-Embodiment

**核心创新**：
可扩展的机器人策略预训练因收集高质量演示成本高而受阻。本文将离线强化学习与跨具身学习相结合：离线RL利用专家和丰富的次优数据，跨具身学习聚合不同形态的异构机器人轨迹以获得通用控制先验。

作者系统分析了这种组合方法的优势和局限，构建了一个包含**16种不同机器人平台**的运动数据集套件。针对多机器人类型间冲突梯度问题，提出了**基于具身的分组策略**：按形态相似性聚类机器人，使用组梯度更新模型，显著减少机器人间冲突。

**评价**: ⭐⭐⭐ 必读 — ICLR 2026，跨具身离线RL的系统研究，16种机器人平台的大规模实验。

---

### 5. SPQ: An Ensemble Technique for Large Language Model Compression
**作者**: Eren Gultepe, Jiamin Yao 等 (LREC 2026)  
**链接**: [arXiv:2602.18420](https://arxiv.org/abs/2602.18420) | [代码](https://github.com/JiaminYao/SPQ_LLM_Compression/)  
**方向**: Efficient LLM / Model Compression

**核心创新**：
SPQ 结合三种互补技术：基于激活的剪枝移除MLP冗余神经元、SVD将注意力投影压缩为低秩因子、8-bit后训练量化统一压缩线性层。在相同压缩比下，SPQ在困惑度和下游任务上均优于单一方法。

**实验结果**：
应用于LLaMA-2-7B：
- **75%内存减少**（6.86 GB vs 26.9 GB）
- WikiText-2困惑度: 5.47 → **4.91**
- 推理吞吐量比GPTQ提升 **1.9倍**
- 在C4、TruthfulQA、GSM8K等下游基准上保持准确率

**评价**: ⭐⭐⭐ 必读 — LREC 2026，三阶段压缩策略实用且效果显著，开源代码，特别适合资源受限环境的部署需求。

---

### 6. Neurosymbolic Language Reasoning as Satisfiability Modulo Theory
**作者**: Matthai Philipose 等  
**链接**: [arXiv:2602.18095](https://arxiv.org/abs/2602.18095)  
**方向**: Reasoning / Neurosymbolic

**核心创新**：
自然语言理解需要文本推理与逻辑推理的交错，但LLM在此类推理上往往不可靠。现有神经符号系统将LLM与求解器结合，但仅限于完全形式化的任务（如数学或程序合成），无法处理仅具有部分逻辑结构的自然文档。

本文提出 **Logitext**，一种神经符号语言，将文档表示为自然语言文本约束(NLTC)，使部分逻辑结构显式化。开发了一种将基于LLM的约束评估与可满足性模理论(SMT)求解相结合的算法，实现联合文本-逻辑推理。

**实验结果**：
在内容审核新基准以及LegalBench和Super-Natural Instructions上的实验表明，Logitext提高了准确率和覆盖率。这是**首次将LLM推理视为SMT理论**的工作，将神经符号方法扩展到非完全形式化领域。

**评价**: ⭐⭐⭐ 必读 — 首次将LLM推理建模为SMT理论，扩展了神经符号方法的适用范围。

---

### 7. On the "Induction Bias" in Sequence Models
**作者**: MReza Ebrahimi 等  
**链接**: [arXiv:2602.18333](https://arxiv.org/abs/2602.18333)  
**方向**: Model Architecture / Efficiency

**核心创新**：
该研究系统比较了Transformer和RNN在状态跟踪任务上的数据效率。发现Transformer所需训练数据随状态空间大小和序列长度增长的速度远快于RNN。更关键的是，Transformer在不同序列长度间几乎没有权重共享，而循环模型通过跨长度共享权重实现了有效的摊销学习。

**实验结果**：
- 大规模实验对比Transformer与RNN
- Transformer训练数据需求随状态空间/序列长度快速增长
- RNN表现出有效的跨长度权重共享

**评价**: ⭐⭐ 可选 — 对理解Transformer的归纳偏置有启发，但主要是诊断性研究而非新方法。

---

### 8. Alignment in Time: Peak-Aware Orchestration for Long-Horizon Agentic Systems
**作者**: Hanjing Shi 等  
**链接**: [arXiv:2602.17910](https://arxiv.org/abs/2602.17910)  
**方向**: Agentic Systems / AI Alignment

**核心创新**：
传统AI对齐主要关注单个模型输出，但长时程工作流中的自主Agent需要在整个交互轨迹上保持持续可靠性。本文提出 **APEMO** (Affect-aware Peak-End Modulation for Orchestration)，一种运行时调度层。

APEMO通过行为代理检测轨迹不稳定性，并在关键时刻（峰值时刻和结束点）有针对性地进行修复。与修改模型权重不同，APEMO在固定预算下优化计算分配。

**实验结果**：
在多智能体模拟和基于LLM的规划器-执行器流程评估中，APEMO在轨迹级质量和重用概率上持续优于结构性编排器。

**评价**: ⭐⭐ 可选 — 长时程Agent系统的对齐新视角，但实验规模相对有限。

---

## 总结

| 论文 | 主题 | 核心贡献 | 评级 |
|------|------|----------|------|
| Decoding as Optimisation on the Probability Simplex | Decoding优化 | 统一解码框架，Best-of-K采样器MATH500+18.6% | ⭐⭐⭐ |
| RAT+: Train Dense, Infer Sparse | 高效推理 | 密集训练稀疏推理，16倍稀疏接近密集准确率 | ⭐⭐⭐ |
| Diffusing to Coordinate | Agentic RL | 首个在线MARL扩散策略框架，样本效率2.5-5倍 | ⭐⭐⭐ |
| Cross-Embodiment Offline RL | Robotics | ICLR 2026，16种机器人平台跨具身离线RL | ⭐⭐⭐ |
| SPQ | 模型压缩 | SVD+剪枝+量化，75%内存压缩，1.9倍吞吐提升 | ⭐⭐⭐ |
| Neurosymbolic Language Reasoning as SMT | Reasoning | 首次将LLM推理建模为SMT理论 | ⭐⭐⭐ |
| On the "Induction Bias" in Sequence Models | 架构分析 | Transformer vs RNN状态跟踪能力对比 | ⭐⭐ |
| Alignment in Time | Agentic Systems | 长时程Agent系统的峰值感知编排 | ⭐⭐ |

**今日趋势观察**：
1. **解码策略理论化**: 将启发式解码方法统一为优化框架成为新趋势，有助于更系统地设计采样策略。
2. **稀疏推理实用化**: "训练密集、推理稀疏"范式日趋成熟，RAT+和SPQ分别从不同角度推进高效推理的可行性。
3. **扩散模型扩展到决策领域**: OMAD将扩散策略引入在线MARL，展示了生成模型在策略表达性上的优势。
4. **神经符号方法走向实用化**: Logitext将神经符号推理扩展到非完全形式化领域，突破了传统限制。

---

**注**: HuggingFace Daily Papers 今日访问受限，本期仅覆盖 arXiv 更新。明日将尝试恢复 HF 数据源。

*数据来源: arXiv (cs.AI, cs.LG, cs.CL) | 筛选时间: 2026-02-24 | 编辑: Amy*
