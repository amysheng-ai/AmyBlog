---
title: Daily AI Papers - 2026年3月2日
published: 2026-03-02
description: 本期聚焦Agentic RL突破：微软提出EMPO²探索式记忆增强框架，LinkedIn提出ACE非对称置信度惩罚优化RLVR，OPPO提出SMTL长程Agentic搜索新范式；同时关注World Model一致性原则与Omni-Modal原生Agent发展。
tags: [Daily Papers, AI, Agentic RL, World Models, RLVR, Multi-Modal]
category: Papers
draft: false
---

# Daily AI Papers - 2026年3月2日

## 今日预览

本期涵盖6篇精选论文，核心亮点包括：**微软**提出EMPO²框架解决LLM Agent的探索瓶颈，通过混合on/off-policy优化实现128.6%的性能提升；**LinkedIn**针对RLVR中的过度自信错误问题，提出ACE非对称置信度惩罚机制；**OPPO**挑战"深度思考"范式，提出"搜索更多、思考更少"的长程Agentic搜索框架；**OpenDataLab**从一致性角度重新定义World Model构建原则；**人大高瓴**推出OmniGAIA原生全模态Agent基准与模型。涵盖Agentic RL、World Models、Reasoning三大核心方向。

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

## 总结

| 论文 | 主题 | 机构 | 核心贡献 | 评级 |
|------|------|------|----------|------|
| The Trinity of Consistency | World Models | OpenDataLab, Shanghai AI Lab | 一致性三位一体原则，物理约束驱动的世界模型 | ⭐⭐⭐ |
| EMPO² | Agentic RL | Microsoft | 探索式记忆增强+混合on/off-policy优化 | ⭐⭐⭐ |
| ACE | RLVR | LinkedIn | 非对称置信度惩罚，解决过度自信错误 | ⭐⭐⭐ |
| SMTL | Agentic Search | OPPO | "搜索更多、思考更少"的长程agent范式 | ⭐⭐ |
| OmniGAIA | Omni-Modal Agents | Renmin University | 原生全模态agent基准与模型 | ⭐⭐ |
| MobilityBench | Agent Benchmark | Alibaba | 真实场景路线规划agent评测 | ⭐⭐ |

**今日趋势观察**:

1. **Agentic RL进入精细化阶段**: 从早期的"能用"转向"高效+鲁棒"，微软EMPO²和LinkedIn ACE分别从探索策略和错误纠正两个角度切入，体现了对训练稳定性的深度关注。

2. **RLVR的多样化发展**: 除了传统的pass@1优化，研究者开始关注Pass@k全谱系改善（ACE）和长期一致性（Consistency），表明社区已认识到推理多样性的重要性。

3. **World Models的新定义**: OpenDataLab的工作标志着World Model从纯数据驱动向"数据+物理约束"的范式转变，这对机器人控制和物理仿真具有重要意义。

4. **Agent架构的反思**: OPPO的SMTL和EMPO²都暗示着对传统"深度推理"范式的反思——有时更多的外部检索比内部思考更有效，这与人类解决问题的策略更为接近。
