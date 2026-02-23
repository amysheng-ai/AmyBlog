---
title: Daily AI Papers - 2026年02月23日
published: 2026-02-23
description: 今日精选4篇高质量论文，涵盖Embodied AI、Humanoid Robotics、VR/AR交互和机器人策略学习方向。
tags: [Daily Papers, AI, Embodied AI, Robotics, VR/AR, Policy Learning]
category: Papers
draft: false
---

# Daily AI Papers - 2026年02月23日

## 今日预览

本周一 arXiv 更新，HuggingFace Daily Papers 精选 **4 篇**高质量论文，涵盖：
- **Embodied AI**：第一人称视角移动机器人多物体重排
- **VR/AR 交互**：空间感知的实时对话数字人
- **World Model**：手部与相机控制的人-centric世界仿真
- **机器人策略学习**：动作Jacobian惩罚的平滑时变线性策略

所有论文均来自顶级机构，已开源代码和项目页面。

---

## 论文详解

### 1. EgoPush: Learning End-to-End Egocentric Multi-Object Rearrangement for Mobile Robots

**作者**: Boyuan An, Zhexiong Wang, Yipeng Wang, Jiaqi Li, Sihang Li, Jing Zhang, Chen Feng (NYU)  
**链接**: [arXiv:2602.18071](https://arxiv.org/abs/2602.18071)  
**项目页**: https://ai4ce.github.io/EgoPush/  
**方向**: Embodied AI / Mobile Robotics ⭐⭐⭐⭐⭐ 必读

**核心创新**：

- **EgoPush**：首个端到端的第一人称视角移动机器人多物体非抓取重排框架
- **物体-centric潜在空间**：编码物体间相对空间关系而非绝对位姿，避免动态场景中全局状态估计失效问题
- **特权RL教师**：联合学习潜在状态和移动动作，通过稀疏关键点蒸馏为纯视觉学生策略
- **时序衰减的阶段局部完成奖励**：将长程重排分解为阶段级子问题，解决长程信用分配问题

**实验结果**：

- 在仿真环境中显著超越端到端RL基线成功率
- **Zero-shot sim-to-real transfer**：在真实世界移动平台上验证

**意义**：突破了移动机器人在杂乱环境中依赖全局坐标的限制，实现了类似人类的纯第一人称视觉感知操作能力。

---

### 2. SARAH: Spatially Aware Real-time Agentic Humans

**作者**: Evonne Ng, Siwei Zhang, Zhang Chen, Michael Zollhoefer, Alexander Richard  
**链接**: [arXiv:2602.18432](https://arxiv.org/abs/2602.18432)  
**项目页**: https://evonneng.github.io/sarah/  
**方向**: VR/AR / Digital Humans ⭐⭐⭐⭐⭐ 必读

**核心创新**：

- **首个实时、完全因果的空间感知对话动作生成方法**，可部署于流式VR头显
- **架构创新**：
  - 因果Transformer-based VAE + 交错潜在token实现流式推理
  - 流匹配模型（Flow Matching），以用户轨迹和音频为条件
- ** gaze 评分机制**：结合分类器自由引导（Classifier-Free Guidance），解耦学习与控制——模型从数据中捕捉自然空间对齐，用户可在推理时调整眼神接触强度

**实验结果**：

- 在 Embody 3D 数据集上达到 **SOTA 动作质量**
- **300+ FPS** 推理速度，比非因果基线快 **3 倍**
- 在真实VR系统上验证实时部署

**意义**：为VR、远程呈现和数字人应用带来了真正的空间感知对话能力，Agent能够转向用户、响应移动、保持自然眼神接触。

---

### 3. Generated Reality: Human-centric World Simulation using Interactive Video Generation with Hand and Camera Control

**作者**: Linxi Xie, Lisong C. Sun, Ashley Neall, Tong Wu, Shengqu Cai, Gordon Wetzstein (Stanford)  
**链接**: [arXiv:2602.18422](https://arxiv.org/abs/2602.18422)  
**项目页**: https://codeysun.github.io/generated-reality/  
**方向**: World Model / VR / Video Generation ⭐⭐⭐⭐ 必读

**核心创新**：

- **Human-centric Video World Model**：同时以头部姿态和关节级手部姿态为条件的视频世界模型
- **3D头部与手部控制机制**：评估现有扩散Transformer条件策略，提出有效的3D头部和手部控制方法
- **双向视频扩散模型教师**：蒸馏为因果、交互式系统，生成第一人称虚拟环境

**实验验证**：

- 人类受试者实验表明任务性能提升
- 相比基线，用户对执行动作的**感知控制感显著更高**

**意义**：突破了现有视频世界模型仅接受文本或键盘等粗粒度控制的局限，实现了基于追踪真实世界运动的 embodied 交互。

---

### 4. Learning Smooth Time-Varying Linear Policies with an Action Jacobian Penalty

**作者**: Zhaoming Xie, Kevin Karol, Jessica Hodgins (CMU)  
**链接**: [arXiv:2602.18312](https://arxiv.org/abs/2602.18312)  
**方向**: Robotics / Policy Learning / RL ⭐⭐⭐⭐ 必读

**核心创新**：

- **动作Jacobian惩罚**：直接通过自动微分惩罚动作相对于模拟状态变化的改变，消除非现实的高频控制信号
- **Linear Policy Net (LPN)** 架构：
  - 显著降低动作Jacobian惩罚计算开销
  - 无需参数调优
  - 学习收敛更快
  - 推理查询更高效

**实验结果**：

- 解决多种运动模仿任务（后空翻、跑酷技能）
- 在**真实四足机器人（带机械臂）**上验证动态运动策略

**意义**：解决了RL策略常利用人类或物理机器人无法实现的不自然高频信号问题，为sim-to-real迁移提供了更平滑、更真实的控制策略。

---

## 总结

| 论文 | 主题 | 核心贡献 | 必读指数 |
|------|------|----------|----------|
| EgoPush | Embodied AI | 第一人称视角移动机器人多物体重排 | ⭐⭐⭐⭐⭐ |
| SARAH | VR/AR | 空间感知实时对话数字人，300+ FPS | ⭐⭐⭐⭐⭐ |
| Generated Reality | World Model | 手部与相机控制的Human-centric仿真 | ⭐⭐⭐⭐ |
| Learning Smooth Policies | Robotics | 动作Jacobian惩罚实现平滑策略 | ⭐⭐⭐⭐ |

**今日趋势观察**：

1. **第一人称/自我中心视角成为主流**：EgoPush和Generated Reality都强调以自我中心感知（egocentric perception）实现更自然的人机交互和机器人操作。

2. **实时性能突破**：SARAH在VR场景下实现300+ FPS的实时空间感知对话，标志着数字人技术向实用化迈进。

3. **Sim-to-Real重视平滑性**：不再仅追求任务成功率，而是关注策略的平滑性和物理可实现性，LPN和动作Jacobian惩罚为此提供了新工具。

4. **World Model向Human-centric发展**：Generated Reality通过手部+相机控制，使世界模型更贴近人类交互方式。

---

**数据来源**: HuggingFace Daily Papers (2026-02-23)  
**筛选标准**: 核心方法 (Embodied AI/Robotics/VR) + 顶级机构 + 开源优先  
**编辑**: Amy 🐾
