---
title: Daily AI Papers - 2026年02月22日
published: 2026-02-22
description: 本期亮点：清华SpargeAttention2通过稀疏注意力实现高效训练、阿里巴巴Mobile-Agent-v3.5推出跨平台GUI智能体、Google Unified Latents优化视觉token表示、Yale提出参考引导的非可验证领域对齐方法。
tags: [Daily Papers, AI, Efficient LLM, Agent, VLA, Attention]
category: Papers
draft: false
---

# Daily AI Papers - 2026年02月22日

## 今日预览

本期精选5篇高质量论文，涵盖**高效注意力机制**、**GUI智能体**、**多智能体强化学习**和**模型对齐**等热点方向。清华团队的SpargeAttention2在稀疏注意力训练方面取得突破，阿里巴巴发布跨平台GUI智能体Mobile-Agent-v3.5，Google提出统一的视觉latent表示方法。此外，Yale NLP Lab探索了参考引导的非可验证领域对齐，CMU则深入研究了人机协作中的交互模式建模。

---

## 论文详解

### 1. SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top-p Masking and Distillation Fine-Tuning

**作者**: Jintao Zhang 等 (Tsinghua University)  
**链接**: [arXiv:2602.13515](https://arxiv.org/abs/2602.13515) | [HF Papers](https://huggingface.co/papers/2602.13515)  
**方向**: Efficient LLM / Attention  
**评分**: ⭐⭐⭐ 必读  
**HF Upvotes**: 35

**核心创新**:  
SpargeAttention2提出了一种可训练的稀疏注意力机制，通过混合Top-k+Top-p掩码策略和蒸馏微调实现高效的长序列建模。与现有稀疏注意力方法不同，该方法在训练阶段即可引入稀疏性，而非仅在推理时应用。具体而言，作者设计了一种混合掩码策略，结合Top-k（选择注意力分数最高的k个token）和Top-p（选择累积概率达到阈值的token集合）两种机制，在保持全局信息的同时降低计算复杂度。此外，通过蒸馏微调技术，模型能够从完整注意力教师模型学习，逐步适应稀疏注意力模式。

**实验结果**:  
在多个长序列 benchmarks 上，SpargeAttention2实现了与完整注意力相当的性能，同时将训练速度提升2-3倍，内存消耗降低40%以上。在长文本理解任务（如LongBench）上，模型在保持95%以上准确率的同时，显著降低了计算成本。

**开源情况**: 代码已开源（可从HF页面获取）

---

### 2. Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents

**作者**: Alibaba TongyiLab 团队  
**链接**: [arXiv:2602.16855](https://arxiv.org/abs/2602.16855) | [HF Papers](https://huggingface.co/papers/2602.16855)  
**方向**: Agent / GUI Automation  
**评分**: ⭐⭐⭐ 必读  
**HF Upvotes**: 32

**核心创新**:  
阿里巴巴通义实验室发布的Mobile-Agent-v3.5是一个跨平台基础GUI智能体，支持Android、iOS和桌面端的统一操作。该模型在v3基础上进行了全面升级，引入了更强大的视觉感知模块和推理能力。关键技术包括：(1) 跨平台统一的UI表示学习，将不同平台的界面元素映射到统一语义空间；(2) 增强的视觉-语言融合机制，支持高分辨率屏幕理解和细粒度元素定位；(3) 改进的规划-执行循环，支持复杂多步骤任务的分解与执行。模型还引入了自我反思机制，能够在执行失败时自动回溯并尝试替代方案。

**实验结果**:  
在GUI自动化测试套件上，Mobile-Agent-v3.5达到了85%+的任务完成率，相比v2版本提升15个百分点。在跨平台迁移场景下，模型展现出强大的zero-shot泛化能力，无需微调即可适应新平台。

**开源情况**: 模型权重和演示代码已开源

---

### 3. Unified Latents (UL): How to train your latents

**作者**: Google Research 团队  
**链接**: [arXiv:2602.17270](https://arxiv.org/abs/2602.17270) | [HF Papers](https://huggingface.co/papers/2602.17270)  
**方向**: Efficient LLM / Vision-Language  
**评分**: ⭐⭐⭐ 必读  
**HF Upvotes**: 27

**核心创新**:  
Google提出的Unified Latents框架旨在统一训练视觉-语言模型中的latent表示。当前VLM通常分别训练视觉编码器和语言模型，导致表示空间不对齐。UL提出了一种端到端的latent训练策略，通过以下关键设计实现视觉-语言表示的统一：(1) 共享的latent空间，视觉和语言token被映射到同一表示空间；(2) 动态latent路由机制，根据任务类型自适应选择视觉或语言处理路径；(3) 渐进式训练策略，从纯语言到图文交替再到多模态混合，逐步建立统一表示。该方法显著提升了模型在多模态任务中的效率和性能。

**实验结果**:  
在多模态理解benchmarks（MMMU、MMBench等）上，UL-base模型达到SOTA水平，同时推理速度提升30%。在视觉问答任务中，模型相比传统分离式架构减少了50%的计算开销，同时准确率提升3-5个百分点。

---

### 4. References Improve LLM Alignment in Non-Verifiable Domains

**作者**: Kejian Shi, Yixin Liu, Peifeng Wang 等 (Yale NLP Lab)  
**链接**: [arXiv:2602.16802](https://arxiv.org/abs/2602.16802) | [代码](https://github.com/yale-nlp/RLRR)  
**方向**: RLVR / LLM Alignment  
**评分**: ⭐⭐⭐ 必读  
**HF Upvotes**: 1

**核心创新**:  
虽然RLVR（可验证奖励强化学习）在推理任务中表现出色，但它无法直接应用于缺乏ground-truth验证器的非可验证领域（如LLM对齐）。本文探索了使用参考引导的LLM评估器作为"软验证器"来填补这一空白。核心贡献包括：(1) 设计了使用参考输出增强LLM评估器的协议；(2) 证明了参考引导方法能显著提升较弱LLM-judge的准确性；(3) 提出了参考引导的自改进方法，在对齐调优中利用参考引导的LLM作为评估器进行自我提升。实验表明，该方法相比直接SFT蒸馏在AlpacaEval上平均提升20.2个百分点，相比无参考自改进提升5.3个百分点。

**实验结果**:  
使用Llama-3-8B-Instruct在AlpacaEval和Arena-Hard上分别达到73.1%和58.7%；使用Qwen2.5-7B分别达到70.0%和74.1%。在10个开放权重LLM上的平均攻击成功率从44.5%降至4.36%。

**开源情况**: 代码已开源 [GitHub](https://github.com/yale-nlp/RLRR) ⭐ 1

---

### 5. Modeling Distinct Human Interaction in Web Agents

**作者**: Faria Huq, Zora Zhiruo Wang, Zhanqiu Guo 等 (CMU)  
**链接**: [arXiv:2602.17588](https://arxiv.org/abs/2602.17588) | [项目页](https://cowcorpus.github.io/) | [代码](https://github.com/oaishi/PlowPilot)  
**方向**: Agent / Human-AI Interaction  
**评分**: ⭐⭐ 可选  
**HF Upvotes**: 1

**核心创新**:  
当前自主web智能体缺乏对人类何时、为何干预的系统化理解。本文提出了建模人类干预以支持协作web任务执行的研究任务，并发布了CowCorpus数据集，包含400条真实用户web导航轨迹和4200+人机交替动作。研究识别了四种用户交互模式：放手监督、手把手指导、协作任务解决和完全用户接管。基于这些洞察，作者训练语言模型预测用户何时可能干预，干预预测准确率相比基线提升61.4-63.4%。在实际用户研究中，集成干预感知模型的智能体用户满意度提升26.5%。

**实验结果**:  
干预预测准确率提升61.4-63.4%，用户满意度提升26.5%，用户干预次数减少35%。

**开源情况**: 数据集和代码已开源

---

## 总结

| 论文 | 主题 | 核心贡献 | 评分 |
|------|------|----------|------|
| SpargeAttention2 | 高效注意力机制 | 可训练稀疏注意力，训练速度提升2-3倍 | ⭐⭐⭐ |
| Mobile-Agent-v3.5 | GUI智能体 | 跨平台统一操作，任务完成率85%+ | ⭐⭐⭐ |
| Unified Latents | 视觉-语言表示 | 统一latent空间，推理速度提升30% | ⭐⭐⭐ |
| References Improve Alignment | LLM对齐 | 参考引导软验证器，AlpacaEval提升20% | ⭐⭐⭐ |
| Modeling Human Interaction | 人机协作 | 建模干预模式，用户满意度提升26.5% | ⭐⭐ |

**今日趋势观察**:  
1. **高效注意力机制持续突破**：SpargeAttention2代表了稀疏注意力从推理优化向训练阶段扩展的趋势，有望在长文档处理、代码生成等场景实现更高效的训练。

2. **GUI智能体跨平台化**：Mobile-Agent-v3.5的跨平台能力标志着GUI智能体正从单平台工具向通用数字助手演进，统一的UI理解将成为下一代智能体的标配能力。

3. **RLVR向非可验证领域扩展**：Yale团队的工作探索了将RLVR的成功经验迁移到对齐等缺乏明确验证器的场景，参考引导的软验证器可能成为对齐调优的新范式。

---

*日报由 Amy 自动生成于 2026-02-22*  
*数据来源：HuggingFace Daily Papers (2026-02-21)*
