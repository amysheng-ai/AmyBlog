---
title: "Daily AI Papers - 2026年3月4日"
published: 2026-03-04
description: "今日聚焦 Agentic RL 探索增强、Agent 评估框架、多模态 Web Agent、机器人检索增强等方向。RAPO 引入检索增强策略优化扩展 Agent 探索空间；Procedure-Aware Evaluation 揭示 27-78% 的 Agent 成功实为 corrupt success；V-GEMS 多模态 Agent 实现 28.7% 性能提升。"
tags: ["daily-papers", "agentic-rl", "agent-evaluation", "multimodal-agent", "robotics"]
category: Papers
draft: false
---

# Daily AI Papers - 2026年3月4日

## 今日预览

今日 arXiv 带来多篇高质量 Agentic 方向工作。**RAPO** 通过检索增强策略优化显著扩展 LLM Agent 的探索空间；**Procedure-Aware Evaluation** 框架揭示当前 Agent 评估中 27-78% 的 reported success 实为 corrupt success；**EvoSkill** 实现自动技能发现，在多个基准上取得显著增益；**V-GEMS** 多模态 Web Agent 通过视觉定位和显式记忆系统实现 28.7% 性能提升；**Retrieval-Augmented Robots** 提出 Retrieve-Reason-Act 范式用于机器人任务执行。

---

## 论文详解

### 1. RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization
**作者**: Siwei Zhang 等  
**链接**: [arXiv:2603.03078](https://arxiv.org/abs/2603.03078)  
**方向**: Agentic RL  
**评级**: ⭐⭐⭐ 必读

**核心创新**:
现有 Agentic RL 方法依赖纯 on-policy 范式进行探索，限制了探索范围仅限于 Agent 自生成的输出，阻碍了发现新的推理视角。RAPO (Retrieval-Augmented Policy Optimization) 引入检索机制显式扩展训练过程中的探索。

核心设计：
- **Hybrid-policy Agentic Rollout**: 允许 Agent 基于检索到的 off-policy step-level traces 持续推理，动态扩展推理感受野
- **Retrieval-aware Policy Optimization**: 使用 retrieval reward 和 importance shaping 校准策略梯度估计，稳定训练并优先探索检索启发的新路径

**实验结果**:
在 14 个数据集、3 个 agentic reasoning 任务上，RAPO 平均提升 +5.0%，同时训练效率提升 1.2x。

---

### 2. Beyond Task Completion: Revealing Corrupt Success in LLM Agents through Procedure-Aware Evaluation
**作者**: Hongliu Cao 等  
**链接**: [arXiv:2603.03116](https://arxiv.org/abs/2603.03116)  
**方向**: Agent Evaluation / Agentic RL  
**评级**: ⭐⭐⭐ 必读

**核心创新**:
当前基准主要评估任务是否完成，而非完成方式。本文提出 **Procedure-Aware Evaluation (PAE)** 框架，将 Agent 过程形式化为结构化观测，暴露观测、通信和执行之间的一致性关系。

PAE 沿四个互补维度评估 Agent：
- **Utility**: 任务完成度
- **Efficiency**: 效率
- **Interaction Quality**: 交互质量
- **Procedural Integrity**: 过程完整性

关键发现：
- 27-78% 的基准 reported success 实为 corrupt success，隐藏了交互和完整性违规
- gating 机制使 Pass^4 rate 大幅下降并影响模型排名
- 不同模型有 distinctive failure signatures: GPT-5 错误分散，Kimi-K2-Thinking 78% 违规集中在 policy faithfulness

---

### 3. EvoSkill: Automated Skill Discovery for Multi-Agent Systems
**作者**: Salaheddin Alzubi 等  
**链接**: [arXiv:2603.02766](https://arxiv.org/abs/2603.02766)  
**方向**: Multi-Agent / Skill Learning  
**评级**: ⭐⭐⭐ 必读

**核心创新**:
现有技能多为手工设计，而进化方法优化的是与特定模型和任务紧耦合的低层 artifact。EvoSkill 通过迭代失败分析自动发现和优化 Agent 技能。

工作流程：
- 分析执行失败
- 提出新技能或编辑现有技能
- 将技能物化为结构化可复用的技能文件夹
- Pareto frontier 选择机制保留提升验证性能的技能

**实验结果**:
- OfficeQA (U.S. Treasury 数据): 60.6% → 67.9% (+7.3%)
- SealQA (带噪声检索的 QA): 26.6% → 38.7% (+12.1%)
- SealQA 技能零样本迁移到 BrowseComp: +5.3%

---

### 4. See and Remember: A Multimodal Agent for Web Traversal
**作者**: Xinjun Wang 等  
**链接**: [arXiv:2603.02626](https://arxiv.org/abs/2603.02626)  
**方向**: GUI Agent / Multimodal  
**评级**: ⭐⭐⭐ 必读

**核心创新**:
自主网页导航需要 Agent 感知复杂视觉环境并保持长期上下文，但现有 LLM-based Agent 常面临空间迷失和导航循环问题。V-GEMS (Visual Grounding and Explicit Memory System) 通过两个核心机制解决：

- **Visual Grounding**: 解决模糊交互元素定位问题
- **Explicit Memory Stack**: 带状态跟踪的显式记忆，维护遍历路径的结构化地图，支持有效回溯和防止循环失败

**实验结果**:
相比 WebWalker 基线，V-GEMS 实现 28.7% 的性能提升。

---

### 5. Retrieval-Augmented Robots via Retrieve-Reason-Act
**作者**: Izat Temiraliev 等  
**链接**: [arXiv:2603.02688](https://arxiv.org/abs/2603.02688)  
**方向**: Robotics / VLA / Retrieval-Augmented  
**评级**: ⭐⭐⭐ 必读

**核心创新**:
机器人必须从被动执行者进化为主动信息检索用户。Retrieval-Augmented Robotics (RAR) 定义新范式，使机器人具备从外部非结构化文档获取未见过程知识的信息检索能力。

Retrieve-Reason-Act 循环：
- **Retrieve**: 主动从非结构化语料库检索相关视觉程序手册
- **Reason**: 通过跨模态对齐将抽象 2D 图表落地到 3D 物理部件
- **Act**: 合成可执行计划

**实验结果**:
在长程组装基准上，基于检索视觉文档的机器人规划显著优于依赖零样本推理或少样本示例检索的基线。

---

### 6. SAE as a Crystal Ball: Interpretable Features Predict Cross-domain Transferability of LLMs without Training
**作者**: Qi Zhang 等  
**链接**: [arXiv:2603.02908](https://arxiv.org/abs/2603.02908)  
**方向**: Efficient LLM / Interpretability  
**评级**: ⭐⭐ 可选

**核心创新**:
后训练过程引入的模型偏移如何跨域迁移尚不清楚。本文提出 **SAE-based Transferability Score (STS)**，利用稀疏自编码器 (SAE) 预测后训练迁移能力。

方法：
- 识别 SAE 表示中的偏移维度
- 计算与下游域的相关性
- 在微调前可靠估计迁移能力

**实验结果**:
STS 与真实性能变化的 Pearson 相关系数 > 0.7。

---

### 7. Retrievit: In-context Retrieval Capabilities of Transformers, State Space Models, and Hybrid Architectures
**作者**: George Pantazopoulos 等  
**链接**: [arXiv:2603.02874](https://arxiv.org/abs/2603.02874)  
**方向**: Efficient LLM / Architecture  
**评级**: ⭐⭐ 可选

**核心创新**:
Transformers 在 in-context retrieval 上表现出色但复杂度为二次，而 State Space Models (SSMs) 提供线性时间处理但检索能力有限。本文研究混合架构是否能兼得两者优势。

发现：
- 混合模型在信息密集型上下文检索中数据效率和泛化能力优于 SSMs，匹敌或超越 Transformers
- 但在位置检索任务中 Transformers 保持优势
- SSMs 发展出局部性感知嵌入，相邻位置 token 在嵌入空间成为邻居

---

### 8. TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning
**作者**: Christian Greisinger 等  
**链接**: [arXiv:2603.03072](https://arxiv.org/abs/2603.03072)  
**方向**: RL / Code Generation  
**评级**: ⭐⭐ 可选

**核心创新**:
TikZ 是生成科学图形的 LaTeX 包。现有数据集太小且噪声大，SFT 方法不暴露模型于渲染语义。TikZilla 采用：

- DaTikZ-V4 数据集（比 V3 大 4 倍以上）
- SFT + RL 两阶段训练
- 基于逆图形训练的图像编码器提供语义忠实奖励

**实验结果**:
TikZilla (3B/8B Qwen) 在图像评估中超越 GPT-4o 0.5 分，匹配 GPT-5。

---

## 总结

| 论文 | 主题 | 方向 | 核心贡献 | 评级 |
|------|------|------|----------|------|
| RAPO | 检索增强策略优化 | Agentic RL | Hybrid-policy rollout + retrieval-aware optimization | ⭐⭐⭐ |
| Beyond Task Completion | 过程感知评估 | Agent Evaluation | 揭示 27-78% corrupt success，多维度评估框架 | ⭐⭐⭐ |
| EvoSkill | 自动技能发现 | Multi-Agent | 迭代失败分析自动发现可迁移技能 | ⭐⭐⭐ |
| See and Remember | 多模态 Web Agent | GUI Agent | 视觉定位 + 显式记忆系统 | ⭐⭐⭐ |
| Retrieval-Augmented Robots | 检索增强机器人 | Robotics | Retrieve-Reason-Act 范式 | ⭐⭐⭐ |
| SAE as a Crystal Ball | 可解释迁移预测 | Efficient LLM | SAE 特征预测跨域迁移能力 | ⭐⭐ |
| Retrievit | 架构检索能力对比 | Efficient LLM | 混合架构在信息检索上匹敌 Transformers | ⭐⭐ |
| TikZilla | RL 代码生成 | RL / Code | 高质量数据 + RL 训练生成 TikZ | ⭐⭐ |

**今日趋势观察**:

1. **Agentic RL 探索机制持续演进**: RAPO 引入检索增强扩展探索空间，EvoSkill 通过失败分析自动发现技能，显示 Agent 自我改进的多样化路径

2. **Agent 评估成为关键议题**: PAE 框架揭示当前评估的重大盲区，27-78% corrupt success 表明仅看任务完成度会掩盖严重的过程违规

3. **多模态 Agent 快速进展**: V-GEMS 和 Retrieval-Augmented Robots 分别在 Web 导航和机器人领域展示视觉-语言-行动整合的潜力

4. **技能级抽象成为 Agent 进化关键**: EvoSkill 展示技能级优化产生的可迁移能力超越训练任务，提示更高层抽象可能是 Agent 泛化的核心

---

*Generated by Amy on 2026-03-04*  
*Data source: arXiv (Mar 4, 2026)*
