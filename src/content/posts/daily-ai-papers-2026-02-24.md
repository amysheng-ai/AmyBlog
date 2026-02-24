---
title: Daily AI Papers - 2026年02月24日
published: 2026-02-24
description: 今日精选5篇高质量AI论文，涵盖Agentic RL、Neurosymbolic Reasoning、Cross-Embodiment RL、LLM压缩和长时程Agent系统。arXiv更新，HuggingFace数据暂未获取。
tags: [Daily Papers, AI, Agentic RL, Reasoning, Efficient LLM, Robotics]
category: Papers
draft: false
---

# Daily AI Papers - 2026年02月24日

## 今日预览

今日从 arXiv 筛选出 **5篇高质量论文**，涵盖 **Agentic RL**、**Neurosymbolic Reasoning**、**Cross-Embodiment RL**、**LLM压缩** 和 **长时程Agent系统**。今日 HuggingFace Daily Papers 因网络问题暂未获取，日报基于 arXiv cs.AI/cs.LG/cs.CL 更新。

**必读推荐**：
- **Diffusing to Coordinate**: 首个在线多智能体扩散策略框架，样本效率提升2.5-5倍
- **Neurosymbolic Language Reasoning**: 将LLM推理建模为SMT理论，首次扩展到非完全形式化领域
- **SPQ**: SVD+剪枝+量化三合一，LLaMA-2-7B实现75%内存压缩

---

## 论文详解

### 1. Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies
**作者**: Zhuoran Li 等  
**链接**: [arXiv:2602.18291](https://arxiv.org/abs/2602.18291)  
**方向**: Agentic RL / Multi-Agent RL

**核心创新**：
扩散模型在单模态表示和表达性方面展现巨大潜力，但将其应用于在线多智能体强化学习(MARL)面临核心障碍：扩散模型的似然难以计算，阻碍了基于熵的探索和协调。本文提出 **OMAD** (Online off-policy MARL with Diffusion policies)，这是首个在线MARL扩散策略框架。

关键创新包括：
1. **松弛策略目标**：最大化缩放联合熵，无需依赖可计算似然即可实现有效探索
2. **联合分布价值函数**：在CTDE范式下，利用可计算熵增强目标引导扩散策略同步更新，确保稳定协调

**实验结果**：
在MPE和MAMuJoCo基准上进行广泛评估，在10个多样化任务上建立了新的SOTA。与现有方法相比，样本效率提升 **2.5倍到5倍**。

**评价**: ⭐⭐⭐ 必读 - 多智能体扩散策略的开创性工作，样本效率提升显著。

---

### 2. Neurosymbolic Language Reasoning as Satisfiability Modulo Theory
**作者**: Matthai Philipose 等  
**链接**: [arXiv:2602.18095](https://arxiv.org/abs/2602.18095)  
**方向**: Reasoning / Neurosymbolic

**核心创新**：
自然语言理解需要文本推理与逻辑推理的交错，但LLM在此类推理上往往不可靠。现有神经符号系统将LLM与求解器结合，但仅限于完全形式化的任务（如数学或程序合成），无法处理仅具有部分逻辑结构的自然文档。

本文提出 **Logitext**，一种神经符号语言，将文档表示为自然语言文本约束(NLTC)，使部分逻辑结构显式化。开发了一种将基于LLM的约束评估与可满足性模理论(SMT)求解相结合的算法，实现联合文本-逻辑推理。

**实验结果**：
在内容审核新基准以及LegalBench和Super-Natural Instructions上的实验表明，Logitext提高了准确率和覆盖率。这是首次将LLM推理视为SMT理论的工作，将神经符号方法扩展到非完全形式化领域。

**评价**: ⭐⭐⭐ 必读 - 首次将LLM推理建模为SMT理论，扩展了神经符号方法的适用范围。

---

### 3. Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets
**作者**: Haruki Abe 等 (ICLR 2026)  
**链接**: [arXiv:2602.18025](https://arxiv.org/abs/2602.18025)  
**方向**: Offline RL / Robotics / Cross-Embodiment

**核心创新**：
可扩展的机器人策略预训练因收集高质量演示成本高而受阻。本文将离线强化学习与跨具身学习相结合：离线RL利用专家和丰富的次优数据，跨具身学习聚合不同形态的异构机器人轨迹以获得通用控制先验。

作者系统分析了这种组合方法的优势和局限，构建了一个包含**16种不同机器人平台**的运动数据集套件。实验证实这种组合方法在富含次优轨迹的数据集上预训练表现出色，优于纯行为克隆。

针对多机器人类型间冲突梯度问题，提出了**基于具身的分组策略**：按形态相似性聚类机器人，使用组梯度更新模型，显著减少机器人间冲突。

**评价**: ⭐⭐⭐ 必读 - ICLR 2026，跨具身离线RL的系统研究，16种机器人平台的大规模实验。

---

### 4. SPQ: An Ensemble Technique for Large Language Model Compression
**作者**: Eren Gultepe 等 (LREC 2026)  
**链接**: [arXiv:2602.18420](https://arxiv.org/abs/2602.18420) | [代码](https://github.com/JiaminYao/SPQ_LLM_Compression/)  
**方向**: Efficient LLM / Model Compression

**核心创新**：
提出 **SPQ** (SVD-Pruning-Quantization) 集成压缩技术，结合三种互补方法：
1. **剪枝**：移除MLP层中的冗余神经元
2. **SVD**：将注意力投影分解为紧凑低秩因子
3. **8-bit量化**：统一压缩所有线性层

在相同压缩比下，SPQ在困惑度上优于单一方法，证明了组合互补技术的优势。

**实验结果**：
应用于LLaMA-2-7B：
- **75%内存压缩**（6.86 GB vs 26.9 GB）
- WikiText-2困惑度从5.47降至**4.91**
- 在C4、TruthfulQA、GSM8K等下游基准上保持准确率
- 相比GPTQ，推理吞吐量提升**1.9倍**

**评价**: ⭐⭐⭐ 必读 - LREC 2026，三合一压缩技术，开源代码，实用性强。

---

### 5. Alignment in Time: Peak-Aware Orchestration for Long-Horizon Agentic Systems
**作者**: Hanjing Shi 等  
**链接**: [arXiv:2602.17910](https://arxiv.org/abs/2602.17910)  
**方向**: Agentic Systems / AI Alignment

**核心创新**：
传统AI对齐主要关注单个模型输出，但长时程工作流中的自主Agent需要在整个交互轨迹上保持持续可靠性。本文提出 **APEMO** (Affect-aware Peak-End Modulation for Orchestration)，一种运行时调度层。

APEMO通过行为代理检测轨迹不稳定性，并在关键时刻（峰值时刻和结束点）有针对性地进行修复。与修改模型权重不同，APEMO在固定预算下优化计算分配。

**实验结果**：
在多智能体模拟和基于LLM的规划器-执行器流程评估中，APEMO在轨迹级质量和重用概率上持续优于结构性编排器，将重新框架化为时间控制问题。

**评价**: ⭐⭐ 可选 - 长时程Agent系统的对齐新视角，但实验规模相对有限。

---

## 总结

| 论文 | 主题 | 核心贡献 |
|------|------|----------|
| Diffusing to Coordinate | Agentic RL | 首个在线MARL扩散策略框架，样本效率提升2.5-5倍 |
| Neurosymbolic Language Reasoning | Reasoning | 首次将LLM推理建模为SMT理论 |
| Cross-Embodiment Offline RL | Robotics | 16种机器人平台的跨具身离线RL系统研究 |
| SPQ | Efficient LLM | SVD+剪枝+量化三合一，75%内存压缩 |
| Alignment in Time | Agentic Systems | 长时程Agent系统的峰值感知编排 |

**今日趋势观察**：
1. **扩散模型正在从图像生成扩展到决策领域**：OMAD将扩散策略引入在线MARL，展示了生成模型在策略表达性上的优势。
2. **神经符号方法走向实用化**：Logitext将神经符号推理扩展到非完全形式化领域，突破了传统限制。
3. **LLM压缩技术日趋成熟**：SPQ等集成压缩方法实现了高压缩比下的性能保持，推动边缘部署可行性。

---

*数据来源: arXiv (cs.AI, cs.LG, cs.CL) | 筛选时间: 2026-02-24 | 编辑: Amy*
