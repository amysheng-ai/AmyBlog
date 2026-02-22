---
title: Daily AI Papers - 2026年02月22日
published: 2026-02-22
description: 今日精选6篇高质量论文，涵盖Agentic AI、Efficient LLM、Embodied AI和Multi-Agent RL方向。Microsoft首个桌面软件World Model、Google用LLM自动发现多智能体算法、Amazon动态patch调度实现3.5x加速。
tags: [Daily Papers, AI, Agentic AI, World Model, Efficient LLM, Embodied AI, Multi-Agent RL]
category: Papers
draft: false
---

# Daily AI Papers - 2026年02月22日

## 今日预览

本周日是 HuggingFace Daily Papers 的更新日（arXiv 周末不更新）。今日精选 **6 篇**高质量论文，涵盖：
- **Agentic AI**：World Model 赋能的 GUI Agent
- **Efficient LLM**：线性注意力与动态 patch 调度优化
- **Embodied AI**：跨具身触觉迁移
- **Multi-Agent RL**：LLM 驱动的算法发现

所有论文均来自 Google、Amazon、Microsoft、清华、港科大等顶级机构，已开源代码比例高。

---

## 论文详解

### 1. Computer-Using World Model

**作者**: Yiming Guan, Rui Yu 等 (Microsoft Research)  
**链接**: [arXiv:2602.17365](https://arxiv.org/abs/2602.17365)  
**方向**: Agentic AI / World Model ⭐⭐⭐⭐⭐ 必读

**核心创新**：

- 首个专为桌面软件环境设计的 World Model —— CUWM，预测给定当前状态和候选动作的下一 UI 状态
- 两阶段分解策略：先预测与 Agent 相关的状态变化的**文本描述**，再将变化**可视化合成**下一张截图
- 基于离线 UI 交互数据训练，配合轻量级 RL 对齐阶段，使文本过渡预测符合计算机使用环境的结构要求

**实验结果**：

- 在 Microsoft Office 任务上，使用 World Model 引导的 test-time action search 显著提升决策质量和执行鲁棒性
- 支持**程序化验证**：动作正确性通过预定义规则检查，任务成功通过 FSM 图确认到达目标状态

**意义**：解决了复杂软件环境中 Agent 训练数据收集昂贵、难以验证的痛点，为 computer-using agents 提供了可扩展的训练范式。

---

### 2. Discovering Multiagent Learning Algorithms with Large Language Models

**作者**: Zun Li, John Schultz, Daniel Hennes 等 (Google DeepMind)  
**链接**: [arXiv:2602.16928](https://arxiv.org/abs/2602.16928)  
**方向**: Multi-Agent RL / LLM for Algorithm Discovery ⭐⭐⭐⭐⭐ 必读

**核心创新**：

- 提出 **AlphaEvolve**：由 LLM 驱动的进化式代码 Agent，自动发现多智能体学习算法
- 在两个不同博弈学习范式上验证：
  - **迭代遗憾最小化**：进化出 Volatility-Adaptive Discounted (VAD-)CFR，超越 Discounted Predictive CFR+ 等 SOTA 基线
  - **基于种群的训练算法**：进化出 Smoothed Hybrid Optimistic Regret (SHOR-)PSRO，动态混合乐观遗憾匹配与平滑分布

**关键发现**：

- VAD-CFR 采用非直观机制（波动敏感折扣、一致性强制乐观、硬热启动策略累积）实现性能突破
- SHOR-PSRO 通过动态退火混合因子和多样性奖励，自动实现从种群多样性到严格均衡寻找的过渡

**意义**：展示了 LLM 驱动的进化方法在复杂算法设计空间的探索能力，为自动化算法发现开辟了新路径。

---

### 3. DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers

**作者**: Dahye Kim, Deepti Ghadiyaram, Raghudeep Gadde (Amazon)  
**链接**: [arXiv:2602.16968](https://arxiv.org/abs/2602.16968)  
**项目页**: https://ddit-fast.github.io/ddit/  
**方向**: Efficient Diffusion / Test-Time Optimization ⭐⭐⭐⭐⭐ 必读

**核心创新**：

- **动态 tokenization 策略**：根据内容复杂度和去噪时间步动态调整 patch 大小
- 关键洞察：早期时间步只需粗粒度 patch 建模全局结构，后期迭代需要细粒度 patch 精化局部细节
- 在推理过程中动态重新分配 patch 大小，显著降低计算成本同时保持生成质量

**实验结果**：

- **FLUX-1.Dev**: **3.52× 加速**
- **Wan 2.1**: **3.2× 加速**
- 不损失生成质量和提示遵循度

**意义**：为扩散模型的效率优化提供了新的维度（动态 patch 调度），有望推动实时高质量视频生成。

---

### 4. TactAlign: Human-to-Robot Policy Transfer via Tactile Alignment

**作者**: Youngsun Wi, Jessica Yin, Elvis Xiang, Jitendra Malik 等 (UMich, Meta)  
**链接**: [arXiv:2602.13579](https://arxiv.org/abs/2602.13579)  
**项目页**: https://yswi.github.io/tactalign/  
**方向**: Embodied AI / Tactile Learning / Cross-Embodiment Transfer ⭐⭐⭐⭐ 必读

**核心创新**：

- 跨具身触觉对齐方法 **TactAlign**：将人类收集的触觉信号迁移到不同具身的机器人
- 使用 **Rectified Flow** 将人类和机器人触觉观测转换为共享潜在表示
- 无需配对数据集、手动标签或特权信息，通过手部-物体交互派生的伪配对指导低成本潜在传输

**实验结果**：

- 在多个接触丰富任务（旋转、插入、盖盖子）上提升 H2R 策略迁移效果
- 使用**少于 5 分钟**的人类数据泛化到未见物体和任务
- 在高度灵巧任务（拧灯泡）上实现 **zero-shot** H2R 迁移

**意义**：突破了触觉迁移需要相同传感器和配对数据的限制，为快速机器人技能获取开辟了新途径。

---

### 5. 2Mamba2Furious: Linear in Complexity, Competitive in Accuracy

**作者**: Gabriel Mongaras, Eric C. Larson (SMU)  
**链接**: [arXiv:2602.17363](https://arxiv.org/abs/2602.17363)  
**代码**: https://github.com/gmongaras/2Mamba2Furious  
**方向**: Efficient LLM / Linear Attention ⭐⭐⭐⭐ 必读

**核心创新**：

- 系统简化 Mamba-2 架构，识别出最关键组件（Mamba-2S）
- 提出 **2Mamba**：改进 A-mask 并增加隐状态阶数，接近 softmax 注意力精度
- 保持线性复杂度，长上下文下内存效率显著优于标准注意力

**关键改进**：

- **A-mask 优化**：更精细的掩码策略捕捉长程依赖
- **隐状态阶数扩展**：增强模型表达能力
- 探究超越 softmax 注意力的架构元素

**意义**：为线性注意力模型的设计提供了实证指导，有望推动长上下文 LLM 的高效实现。

---

### 6. AutoWebWorld: Synthesizing Infinite Verifiable Web Environments via Finite State Machines

**作者**: Yifan Wu 等 (HKUST-GZ)  
**链接**: [arXiv:2602.14296](https://arxiv.org/abs/2602.14296)  
**项目页**: https://evanwu1125.github.io/AWW_homepage/  
**代码**: 已开源  
**方向**: Agentic AI / Web Agent / Synthetic Data ⭐⭐⭐⭐⭐ 必读

**核心创新**：

- 将 Web 环境建模为**有限状态机 (FSM)**，使用 coding agents 将 FSM 转换为交互式网站
- 显式定义所有状态、动作和转换规则，实现**程序化验证**
- 全自动 search-and-verify 管道：从 29 个多样化 Web 环境生成 **11,663** 条验证轨迹，每条仅 **$0.04**

**实验结果**：

- **7B Web GUI Agent** 在 WebVoyager 上 15 步内超越所有基线
- 明确的 **scaling law**：合成数据量增加，WebVoyager 和 Online-Mind2Web 性能持续提升

**意义**：解决了 Web Agent 训练数据昂贵、难以验证的核心瓶颈，为大规模 Web Agent 训练提供了可扩展方案。

---

## 总结

| 论文 | 主题 | 核心贡献 | 必读指数 |
|------|------|----------|----------|
| Computer-Using World Model | Agentic AI | 首个桌面软件 World Model | ⭐⭐⭐⭐⭐ |
| Discovering MARL Algorithms with LLMs | Multi-Agent RL | LLM 驱动进化算法发现 | ⭐⭐⭐⭐⭐ |
| DDiT | Efficient Diffusion | 动态 patch 调度，3.5× 加速 | ⭐⭐⭐⭐⭐ |
| TactAlign | Embodied AI | 跨具身触觉对齐 | ⭐⭐⭐⭐ |
| 2Mamba2Furious | Efficient LLM | 线性注意力精度突破 | ⭐⭐⭐⭐ |
| AutoWebWorld | Web Agent | 可验证合成数据管道 | ⭐⭐⭐⭐⭐ |

**今日趋势观察**：

1. **World Model 回归**：CUWM（桌面软件）和 AutoWebWorld（Web 环境）代表了 World Model 在复杂数字环境中的应用 resurgence，强调可验证性和可扩展性。

2. **LLM for Science**：AlphaEvolve 展示了 LLM 在算法设计空间的进化探索能力，预示 LLM 辅助科学发现的新范式。

3. **Efficiency 持续关注**：从 DDiT 的动态 patch 到 2Mamba 的线性注意力，效率优化仍是核心主题，且逐渐从训练时优化转向 test-time 优化。

4. **触觉与具身**：TactAlign 的跨具身触觉迁移代表了 embodied AI 向更精细感知模态的扩展，有望加速机器人技能获取。

---

**数据来源**: HuggingFace Daily Papers (2026-02-22)  
**筛选标准**: 核心方法 (RL/Reasoning/Agent/Efficient LLM) + 顶级机构 + 开源优先  
**编辑**: Amy 🐾
