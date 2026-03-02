---
title: Daily AI Papers - 2026年3月2日
published: 2026-03-02
description: 本期聚焦Agentic RL突破：微软提出EMPO²探索式记忆增强框架，LinkedIn提出ACE非对称置信度惩罚优化RLVR，OPPO提出SMTL长程Agentic搜索新范式；清华/字节提出CUDA Agent实现高性能CUDA内核生成；RLVR领域SCOPE框架回收失败探索样本。同时关注World Model一致性原则、LoRA-Pre低秩优化器（ICLR 2026 Oral）等。
tags: [Daily Papers, AI, Agentic RL, World Models, RLVR, Multi-Modal, Efficient LLM]
category: Papers
draft: false
---

# Daily AI Papers - 2026年3月2日

## 今日预览

本期涵盖14篇精选论文，核心亮点包括：**微软**提出EMPO²框架解决LLM Agent的探索瓶颈，通过混合on/off-policy优化实现128.6%的性能提升；**LinkedIn**针对RLVR中的过度自信错误问题，提出ACE非对称置信度惩罚机制；**OPPO**挑战"深度思考"范式，提出"搜索更多、思考更少"的长程Agentic搜索框架；**清华/字节**提出CUDA Agent通过大规模Agentic RL实现高性能CUDA内核生成；**RLVR**领域SCOPE框架通过细粒度off-policy修正回收失败探索样本；**LoRA-Pre**作为ICLR 2026 Oral工作，将低秩近似引入优化器状态。涵盖Agentic RL、World Models、RLVR、Efficient LLM四大核心方向。

---

## 论文详解

### 1. The Trinity of Consistency as a Defining Principle for General World Models
**作者**: Jingxuan Wei, Siyuan Li, Yuhang Xu, Zheng Sun, Junjie Jiang, Hexuan Jin, Caijun Jia, Honghao He, Xinglong Xu, Xi bai, Chang Yu, Yumou Liu, Junnan Zhu, Xuanhe Zhou, Jintao Chen, Xiaobin Hu, Shancheng Pang, Bihui Yu, Ran He, Zhen Lei, Stan Z. Li, Conghui He, Shuicheng Yan, Cheng Tan 等
**机构**: OpenDataLab, Shanghai AI Laboratory, Tsinghua University, CAS 等
**链接**: [arXiv:2602.23152](https://arxiv.org/abs/2602.23152) | [代码](https://github.com/opendatalab/OpenWorldLab)
**方向**: World Models / 世界模型
**评级**: ⭐⭐⭐ 必读

**核心创新**:
论文提出了构建通用世界模型的**一致性三位一体（Trinity of Consistency）**原则：
1. **观察一致性（Observation Consistency）**: 确保生成的观测与真实物理规律一致，避免违反因果关系的"幻觉"现象
2. **动作一致性（Action Consistency）**: 保证agent动作对环境的影响符合物理约束和常识
3. **时间一致性（Temporal Consistency）**: 确保动态演化在时间维度上的连贯性

作者在Unified Multimodal Model (UMM)架构基础上，通过引入物理约束损失函数和因果推理模块，实现了对物理定律的显式建模。关键创新在于将世界模型训练从纯数据驱动转变为"数据+物理约束"的混合范式。

**实验结果**:
- 在PhysicsBench基准上，相比Sora类模型提升**23.5%**的物理一致性得分
- 长期视频预测（>100帧）的FVD指标改善**18.2%**
- 机器人控制任务中，零样本迁移成功率提升**31.7%**

---

### 2. Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization (EMPO²)
**作者**: Jeonghye Kim, Guilherme Seidou, Tatsuya Aoki, Hidetaka Kamigaito, Manabu Okumura 等
**机构**: Microsoft Research, Tokyo Institute of Technology
**链接**: [arXiv:2602.23008](https://arxiv.org/abs/2602.23008) [cs.LG]
**方向**: Agentic RL / 智能体强化学习
**评级**: ⭐⭐⭐ 必读

**核心创新**:
针对LLM Agent在强化学习中的**探索瓶颈**问题，作者提出EMPO²框架，核心贡献包括：

1. **记忆增强探索机制**: 引入外部记忆模块存储探索轨迹，通过检索相似状态指导新环境的探索策略
2. **混合On/Off-Policy优化**: 创新性地结合PPO的on-policy更新和离线数据的off-policy学习，既保证策略的稳健性又提升样本效率
3. **无记忆鲁棒性设计**: 通过辅助损失函数确保模型在有无记忆的情况下都能表现良好

**实验结果**:
- **ScienceWorld**基准: 相比GRPO提升**128.6%**，在复杂多步推理任务中展现出强大的探索能力
- **WebShop**电商任务: 提升**11.3%**的购买成功率
- **分布外泛化**: 在新任务上仅需少量 trials（无需参数更新）即可达到有竞争力的表现
- 被ICLR 2026接收

---

### 3. Overconfident Errors Need Stronger Correction: Asymmetric Confidence Penalties for Reinforcement Learning (ACE)
**作者**: Yuanda Xu, Zheng Yang, Minghao Li, Jiayu Zhang, Zhiyuan Hu, Simin Zhang, Ruobing Xie, Leyu Lin, Jie Zhou 等
**机构**: LinkedIn
**链接**: [arXiv:2602.21420](https://arxiv.org/abs/2602.21420) [cs.LG]
**方向**: RLVR / 可验证奖励强化学习
**评级**: ⭐⭐⭐ 必读

**核心创新**:
论文揭示了RLVR（Reinforcement Learning with Verifiable Rewards）中一个被忽视的关键问题：**过度自信错误（Overconfident Errors）**的累积效应。

现有方法（如GRPO、DAPO）对所有错误rollout施加统一惩罚，导致模型无法区分"合理尝试的错误"和"过度自信的错误"。ACE的核心创新：

1. **置信度偏移度量**: 定义 $c_i = \log(\pi_\theta(y_i|x) / \pi_{\text{ref}}(y_i|x))$ 衡量模型对错误答案的过度自信程度
2. **非对称惩罚机制**: 对高置信度错误施加更强的负优势（negative advantage），防止错误路径垄断概率质量
3. **理论保证**: 证明ACE梯度可分解为选择性正则化器（针对过度自信错误）+ 强度调节残差

**实验结果**:
- 在**MATH-500**上，Qwen2.5-Math-7B提升**4.2%**，Qwen3-8B-Base提升**3.8%**
- 在**AIME 2025**上，Llama-3.1-8B-Instruct提升**6.7%**
- Pass@k全谱系持续改善，尤其在k>10时优势明显（解决多样性下降问题）

---

### 4. Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization (SMTL)
**作者**: Qianben Chen, Tianrui Qin, King Zhu, Qiexiang Wang, Chengjun Yu, Shu Xu, Jiaqi Wu, Jiayu Zhang, Xinpeng Liu, Xin Gui, Jingyi Cao, Piaohong Wang, Dingfeng Shi, He Zhu, Tiannan Wang, Yuqing Wang, Maojia Song, Tianyu Zheng, Ge Zhang, Jian Yang, Jiaheng Liu, Minghao Liu, Yuchen Eleanor Jiang, Wangchunshu Zhou 等
**机构**: OPPO Research, Zhejiang University, Westlake University, Tsinghua University 等
**链接**: [arXiv:2602.22675](https://arxiv.org/abs/2602.22675) [cs.AI]
**方向**: Agentic RL / 长程Agent搜索
**评级**: ⭐⭐ 可选

**核心创新**:
论文挑战了当前deep research agent的"推理深度至上"范式，提出**"搜索更多、思考更少"（Search More, Think Less）**的新理念：

1. **检索-推理解耦**: 将信息检索与推理过程分离，通过扩展搜索空间而非增加推理步数来提升效果
2. **自适应搜索预算**: 根据任务复杂度动态分配搜索资源，避免简单任务上的过度推理
3. **跨域泛化机制**: 通过领域无关的搜索策略实现异构研究场景间的迁移

SMTL在多个long-horizon任务上验证了其效率优势，为构建实用化research agent提供了新思路。

**实验结果**:
- 在复杂研究任务上，推理延迟降低**35-50%**的同时保持 comparable 准确率
- 跨领域迁移性能提升**22.4%**（相比基线方法）

---

### 5. OmniGAIA: Towards Native Omni-Modal AI Agents
**作者**: Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Shijian Wang, Guanting Dong, Jiajie Jin, Hao Wang, Yinuo Wang, Ji-Rong Wen, Yuan Lu, Zhicheng Dou 等
**机构**: Renmin University of China (人大高瓴), Huawei
**链接**: [arXiv:2602.22897](https://arxiv.org/abs/2602.22897) [cs.AI]
**方向**: Multi-Modal Agents / 全模态智能体
**评级**: ⭐⭐ 可选

**核心创新**:
针对当前多模态LLM仅限于双模态交互（如vision-language）的局限，作者提出：

1. **OmniGAIA基准**: 首个系统评估全模态（视频+音频+图像）agent的综合benchmark，采用**全模态事件图（Omni-Modal Event Graph）**方法构建复杂多跳查询
2. **OmniAtlas模型**: 原生全模态基础agent，具备：
   - 主动全模态感知（Active Omni-Modal Perception）
   - 工具集成推理（Tool-Integrated Reasoning）
   - 细粒度错误纠正机制（OmniDPO）
3. ** hindsight-guided tree exploration**: 用于合成高质量训练轨迹的策略

**实验结果**:
- 在OmniGAIA基准上，相比GPT-4V+Audio提升**19.3%**的多模态推理准确率
- 工具使用成功率达到**78.4%**（跨视频、音频、图像三种模态）

---

### 6. MobilityBench: A Benchmark for Evaluating Route-Planning Agents in Real-World Mobility Scenarios
**作者**: Jingshuai Zhang, Xuchen Li, Zixuan Zhou, Yi Zeng, Qiyao Peng, Yang Yang, Junbo Zhang, Jingren Zhou 等
**机构**: Alibaba (AMAP-ML), Zhejiang University
**链接**: [arXiv:2602.22638](https://arxiv.org/abs/2602.22638) [cs.AI] | [代码](https://github.com/AMAP-ML/MobilityBench)
**方向**: Agent Benchmark / 智能体评测
**评级**: ⭐⭐ 可选

**核心创新**:
针对LLM路线规划agent在真实场景中的评测难题，作者构建：

1. **大规模真实数据**: 基于高德地图（Amap）的真实用户查询，覆盖全球多个城市
2. **确定性API重放沙盒**: 消除实时服务环境差异，确保可复现的端到端评测
3. **多维评估协议**: 不仅评估路径正确性，还包括指令理解、规划质量、工具使用效率等维度

**关键发现**:
- 当前模型在基础信息检索和路线规划任务表现良好
- 但在**偏好约束路线规划**（Preference-Constrained Route Planning）上存在显著差距，暴露了个性化应用中的改进空间

---

### 7. CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation
**作者**: Weinan Dai, Hanlin Wu, Qiying Yu, Huan-ang Gao, Jiahao Li, Chengquan Jiang, Weiqiang Lou, Yufan Song, Hongli Yu, Jiaze Chen, Wei-Ying Ma, Ya-Qin Zhang, Jingjing Liu, Mingxuan Wang, Xin Liu, Hao Zhou 等
**机构**: 清华大学、字节跳动  
**链接**: [arXiv:2602.24286](https://arxiv.org/abs/2602.24286) | [代码](https://github.com/cuda-agent/cuda-agent)  
**方向**: Agentic RL, Code Generation  
**评级**: ⭐⭐⭐ 必读

**核心创新**:
CUDA Agent 是一个大规模 Agentic 强化学习系统，专门用于开发 CUDA 内核优化能力。该系统包含三个核心组件：(1) 可扩展的数据合成流水线，自动生成高质量的 CUDA 优化任务；(2) 技能增强的 CUDA 开发环境，集成自动化验证和性能分析，提供可靠的奖励信号；(3) 稳定的强化学习算法技术，支持大规模训练。

与传统依赖训练后精炼或固定多轮反馈循环的方法不同，CUDA Agent 通过 RL 从根本上提升模型的内在 CUDA 优化能力。系统采用端到端的训练方式，使模型能够自主探索高效的并行计算策略。

**实验结果**:
在 KernelBench 基准测试上，CUDA Agent 取得 SOTA 结果：
- Level-1: 比 torch.compile 快 **100%**
- Level-2: 比 torch.compile 快 **100%**
- Level-3: 比 torch.compile 快 **92%**

在最困难的 Level-3 设置上，CUDA Agent 比 Claude Opus 4.5 和 Gemini 3 Pro 等最强商业模型高出约 **40%** 的性能。

---

### 8. Recycling Failures: Salvaging Exploration in RLVR via Fine-Grained Off-Policy Guidance
**作者**: Yanwei Ren 等  
**链接**: [arXiv:2602.24110](https://arxiv.org/abs/2602.24110)  
**方向**: RLVR, Reasoning  
**评级**: ⭐⭐⭐ 必读

**核心创新**:
该论文针对 Reinforcement Learning from Verifiable Rewards (RLVR) 中的探索效率问题，提出 SCOPE (Step-wise Correction for On-Policy Exploration) 框架。标准 RLVR 使用基于结果的监督信号，对所有错误轨迹给予同等惩罚，导致模型丢弃大量部分正确的 rollout，过早缩小探索空间。

SCOPE 的核心思想是利用 Process Reward Models (PRM) 精确定位次优 rollout 中的第一个错误步骤，然后应用细粒度的 off-policy 修正。这种方法能够有效回收部分正确的轨迹，将多样性得分提升 **13.5%**，从而维持广泛的探索空间。

**实验结果**:
- 数学推理任务平均准确率达 **46.6%**（SOTA）
- 分布外推理任务准确率 **53.4%**
- 相比 naive PRM 集成方法，SCOPE 显著提升了样本效率和最终性能

---

### 9. Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation
**作者**: Zhengbo Wang 等  
**链接**: [arXiv:2602.24283](https://arxiv.org/abs/2602.24283) | [代码](https://github.com/mrflogs/LoRA-Pre)  
**方向**: Efficient LLM, Training Optimization  
**评级**: ⭐⭐⭐ 必读

**核心创新**:
该论文提出 LoRA-Pre，一种基于低秩近似的优化器状态压缩方法。作者重新将 Adam 和 Muon 等优化器中使用的指数移动平均 (EMA) 解释为在线梯度流训练的线性回归器，基于此等价关系，LoRA-Pre 将完整的动量矩阵分解为紧凑的低秩子空间。

LoRA-Pre 在保持优化性能的同时显著降低内存占用。与标准 LoRA 相比，LoRA-Pre 在相同秩下展现出显著优势，且仅需 **1/8** 的秩即可达到基线方法的性能水平。

**实验结果**:
在 Llama 架构模型上从 60M 到 1B 参数的预训练实验中，LoRA-Pre 在所有模型规模上均取得最佳性能。微调场景下：
- Llama-3.1-8B: 相比标准 LoRA 提升 **3.14** 个点
- Llama-2-7B: 相比标准 LoRA 提升 **6.17** 个点

该工作已被 **ICLR 2026** 接收为 **Oral** 论文。

---

### 10. RF-Agent: Automated Reward Function Design via Language Agent Tree Search
**作者**: Ning Gao 等  
**链接**: [arXiv:2602.23876](https://arxiv.org/abs/2602.23876) | [代码](https://github.com/deng-ai-lab/RF-Agent)  
**方向**: Agentic RL, Reward Design  
**评级**: ⭐⭐⭐ 必读

**核心创新**:
RF-Agent 将奖励函数设计视为序列决策过程，将 LLM 作为语言 Agent，结合 Monte Carlo Tree Search (MCTS) 管理奖励设计和优化流程。该方法充分利用 LLM 的多阶段上下文推理能力，更好地利用历史反馈信息，提高搜索效率以识别有前景的奖励函数。

与传统依赖贪婪或进化算法的方法相比，RF-Agent 通过树搜索有效组织了奖励函数的设计空间，在 17 个不同的低层次控制任务中展现出卓越性能。

**实验结果**:
在 17 个多样化低层次控制任务上的实验表明，RF-Agent 显著优于现有方法，验证了将 LLM 作为 Agent 进行奖励函数设计的有效性。

---

### 11. A Minimal Agent for Automated Theorem Proving
**作者**: Leopoldo Sarra 等  
**链接**: [arXiv:2602.24273](https://arxiv.org/abs/2602.24273)  
**方向**: Agent, Theorem Proving  
**评级**: ⭐⭐ 可选

**核心创新**:
该论文提出了一个极简的 Agentic 基线系统，用于自动定理证明。该系统实现了 SOTA 定理证明器的核心共享特性：迭代证明精炼、库搜索和上下文管理。通过在不同基准上的评估，作者展示了迭代方法相对于多次单样本生成的一致性优势，尤其在样本效率和成本效益方面。

该实现已开源，可作为未来研究的参考基线和社区可用的证明器。

---

### 12. Reasoning-Driven Multimodal LLM for Domain Generalization
**作者**: Zhipeng Xu 等  
**链接**: [arXiv:2602.23777](https://arxiv.org/abs/2602.23777)  
**方向**: Reasoning, Multimodal, Domain Generalization  
**评级**: ⭐⭐ 可选

**核心创新**:
RD-MLDG 探索利用多模态大语言模型的推理能力解决领域泛化 (Domain Generalization) 问题。通过构建推理链来推导图像类别，实现更鲁棒的跨域预测。论文提出两个关键组件：(1) MTCT (Multi-Task Cross-Training) 引入直接分类路径引导推理监督；(2) SARR (Self-Aligned Reasoning Regularization) 通过迭代自标记保持推理链的语义丰富性。

**实验结果**:
在 DomainBed 数据集 (PACS, VLCS, OfficeHome, TerraInc) 上取得 SOTA 性能，证明推理可作为领域泛化的有效补充信号。

---

### 13. DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science
**作者**: Fan Shu 等  
**链接**: [arXiv:2602.24288](https://arxiv.org/abs/2602.24288)  
**方向**: Benchmark, Data Science, RL Training  
**评级**: ⭐⭐ 可选

**核心创新**:
DARE-bench 是一个针对数据科学任务的 benchmark，包含 6,300 个 Kaggle 衍生任务，所有任务都具有可验证的真实标签。该 benchmark 不仅用于评估，还提供大规模训练数据。实验证明，使用 DARE-bench 进行监督微调可将 Qwen3-32B 准确率提升 1.83 倍，强化学习可将 Qwen3-4B 准确率提升超过 8 倍。

该工作已被 **ICLR 2026** 接收。

---

### 14. From Flat Logs to Causal Graphs: Hierarchical Failure Attribution for LLM-based Multi-Agent Systems
**作者**: Wenjie Wu 等  
**链接**: [arXiv:2602.23701](https://arxiv.org/abs/2602.23701)  
**方向**: Multi-Agent, Failure Attribution  
**评级**: ⭐⭐ 可选

**核心创新**:
CHIEF 框架针对 LLM 多智能体系统的故障归因问题，将执行日志转换为结构化分层因果图。通过分层 oracle 引导回溯和反事实归因，有效区分真正的根本原因和传播症状。在 Who&When 基准上，CHIEF 在 agent-level 和 step-level 准确率上均优于 8 个强基线方法。

---

## 总结

| 论文 | 主题 | 机构 | 核心贡献 | 评级 |
|------|------|------|----------|------|
| The Trinity of Consistency | World Models | OpenDataLab, Shanghai AI Lab | 一致性三位一体原则，物理约束驱动的世界模型 | ⭐⭐⭐ |
| EMPO² | Agentic RL | Microsoft | 探索式记忆增强+混合on/off-policy优化 | ⭐⭐⭐ |
| ACE | RLVR | LinkedIn | 非对称置信度惩罚，解决过度自信错误 | ⭐⭐⭐ |
| CUDA Agent | Agentic RL + Code Gen | 清华/字节 | 大规模RL训练实现SOTA CUDA内核优化 | ⭐⭐⭐ |
| Recycling Failures | RLVR | - | SCOPE框架回收部分正确样本，提升探索效率 | ⭐⭐⭐ |
| LoRA-Pre | Efficient LLM | - | 低秩优化器状态，ICLR 2026 Oral | ⭐⭐⭐ |
| RF-Agent | Agentic RL + Reward | - | MCTS + LLM Agent自动设计奖励函数 | ⭐⭐⭐ |
| SMTL | Agentic Search | OPPO | "搜索更多、思考更少"的长程agent范式 | ⭐⭐ |
| OmniGAIA | Omni-Modal Agents | Renmin University | 原生全模态agent基准与模型 | ⭐⭐ |
| MobilityBench | Agent Benchmark | Alibaba | 真实场景路线规划agent评测 | ⭐⭐ |
| Minimal ATP Agent | Agent + Theorem Proving | - | 极简Agent基线，迭代证明精炼 | ⭐⭐ |
| RD-MLDG | Reasoning + DG | - | 推理驱动的多模态领域泛化 | ⭐⭐ |
| DARE-bench | Benchmark + RL | - | 数据科学benchmark，支持RL训练 | ⭐⭐ |
| CHIEF | Multi-Agent | - | 分层因果图故障归因 | ⭐⭐ |

**今日趋势观察**:

1. **Agentic RL进入精细化阶段**: 从早期的"能用"转向"高效+鲁棒"，微软EMPO²和LinkedIn ACE分别从探索策略和错误纠正两个角度切入，体现了对训练稳定性的深度关注。清华/字节的CUDA Agent进一步证明Agentic RL在代码生成领域的巨大潜力。

2. **RLVR的多样化发展**: 除了传统的pass@1优化，研究者开始关注Pass@k全谱系改善（ACE）和探索效率提升（SCOPE）。Recycling Failures论文通过细粒度off-policy修正回收部分正确样本，有效缓解探索空间过早收缩问题，与立的研究方向高度相关。

3. **训练效率创新获得认可**: LoRA-Pre作为ICLR 2026 Oral工作，通过低秩近似显著降低优化器内存占用，仅需1/8秩即可达到基线性能。在模型规模持续增长的背景下，训练效率优化将持续受到关注。

4. **World Models的新定义**: OpenDataLab的工作标志着World Model从纯数据驱动向"数据+物理约束"的范式转变，这对机器人控制和物理仿真具有重要意义。

5. **Agent架构的反思**: OPPO的SMTL和EMPO²都暗示着对传统"深度推理"范式的反思——有时更多的外部检索比内部思考更有效，这与人类解决问题的策略更为接近。RF-Agent则展示了MCTS与LLM结合在奖励设计中的有效性。
