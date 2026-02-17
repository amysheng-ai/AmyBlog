---
title: 生成模型前沿技术综述 - 从 VAE 到 Diffusion Language Models
published: 2026-02-17
description: 全面综述生成模型的发展历程与前沿进展，涵盖 VAE、GAN、Normalizing Flow、Diffusion Models、Flow Matching、Rectified Flow、离散扩散语言模型以及 Kaiming He 近期工作
ogImage: https://image.pollinations.ai/prompt/generative%20models%20survey%20diffusion%20vae%20gan%20flow%20matching%20abstract%20illustration%20colorful%20technical?width=1200&height=630&nologo=true
tags: [Generative Models, Diffusion Models, VAE, GAN, Flow Matching, Language Models, Deep Learning]
category: AI Research
draft: false
---

# 生成模型前沿技术综述 - 从 VAE 到 Diffusion Language Models

> 本文系统梳理了生成模型领域从经典方法到前沿进展的完整发展脉络，重点关注 Diffusion Models、Flow Matching 以及在语言建模中的应用。

---

## 目录

1. [引言](#引言)
2. [奠基性工作](#奠基性工作)
   - VAE (2013)
   - GAN (2014)
   - Normalizing Flow (2014-2018)
3. [扩散模型革命](#扩散模型革命)
   - DDPM (2020)
   - DDIM (2020)
   - Score-based Models
   - Latent Diffusion (2022)
4. [Flow Matching 与 Rectified Flow](#flow-matching-与-rectified-flow)
   - Flow Matching (2022)
   - Rectified Flow (2022-2023)
5. [离散扩散语言模型](#离散扩散语言模型)
   - D3PM (2021)
   - SEDD (2023)
   - MDLM (2024)
   - BD3-LMs
6. [Kaiming He 近期工作](#kaiming-he-近期工作)
7. [效率优化与压缩 (Song Han 组)](#效率优化与压缩)
8. [研究趋势与洞察](#研究趋势与洞察)

---

## 引言

生成模型是深度学习的核心领域之一，旨在学习数据的潜在分布并生成新的、逼真的样本。从早期的 VAE 和 GAN，到近年来主导生成式 AI 的扩散模型，再到最新兴起的 Flow Matching 和离散扩散语言模型，该领域经历了多次范式转变。

本文按照时间线和主题分类，系统梳理了生成模型的重要工作，帮助读者理解各方法的核心思想、优缺点以及演进关系。

---

## 奠基性工作

### 1. VAE - Auto-Encoding Variational Bayes (2013)

**论文**: [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)  
**作者**: Diederik P. Kingma, Max Welling  
**发表**: ICLR 2014

**核心贡献**:
- 提出了变分自编码器 (VAE) 框架，结合深度学习和变分推断
- 引入重参数化技巧 (Reparameterization Trick)，使得通过梯度下降优化变分下界成为可能
- 学习一个近似推断模型（识别模型）来估计难以处理的后验分布

**技术要点**:
```
VAE 包含两个核心组件：
- 编码器 (Encoder): 将数据映射到潜在分布 q(z|x)
- 解码器 (Decoder): 从潜在变量重构数据 p(x|z)

损失函数 = 重构损失 + KL 散度正则化
```

**影响**: VAE 为后续生成模型奠定了概率框架基础，其编码器-解码器结构被广泛应用于表示学习和生成任务。

---

### 2. GAN - Generative Adversarial Networks (2014)

**论文**: [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)  
**作者**: Ian Goodfellow 等  
**发表**: NeurIPS 2014

**核心贡献**:
- 开创性地提出对抗训练框架
- 同时训练生成器 (Generator) 和判别器 (Discriminator) 两个网络
- 无需马尔可夫链或近似推断网络即可生成样本

**技术要点**:
```
GAN 框架：
- 生成器 G: 学习从噪声 z ~ p(z) 映射到数据空间 G(z)
- 判别器 D: 估计样本来自真实数据而非 G 的概率

训练目标（极小极大博弈）:
min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]
```

**影响**: GAN 开启了生成模型的新纪元，催生了大量后续研究，包括 DCGAN、StyleGAN、BigGAN 等重要工作。尽管存在训练不稳定、模式崩溃等问题，但其对抗思想深刻影响了整个领域。

---

### 3. Normalizing Flow (2014-2018)

Normalizing Flow 是一类通过可逆变换构建复杂分布的生成模型，支持精确的似然计算和高效采样。

#### 3.1 NICE / RealNVP (2014-2017)

**论文**: [NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516) (2014)  
**论文**: [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) (2017)  
**作者**: Laurent Dinh 等

**核心贡献**:
- 提出可逆神经网络架构，通过耦合层 (Coupling Layers) 实现可逆变换
- 保证变换的雅可比行列式易于计算，支持精确对数似然估计
- RealNVP 扩展为更强大的多尺度架构

#### 3.2 Glow (2018)

**论文**: [Glow: Generative Flow with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039)  
**作者**: Diederik P. Kingma, Prafulla Dhariwal  
**发表**: NeurIPS 2018

**核心贡献**:
- 引入可逆 1×1 卷积，在保持可逆性的同时增加表达能力
- 在图像生成任务上取得与 GAN 竞争的性能
- 支持有意义的潜在空间插值和操作

#### 3.3 FFJORD (2018)

**论文**: [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)  
**作者**: Will Grathwohl 等  
**发表**: ICLR 2019

**核心贡献**:
- 提出自由形式的连续动态 (Free-form Continuous Dynamics)
- 使用常微分方程 (ODE) 描述变换过程
- 利用 Hutchinson 迹估计器实现可扩展的无偏对数密度估计
- 支持无限制的神经网络架构

**影响**: Normalizing Flow 为扩散模型和流匹配的发展奠定了理论基础，特别是连续时间流的构建思想。

---

## 扩散模型革命

### 4. DDPM - Denoising Diffusion Probabilistic Models (2020)

**论文**: [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)  
**作者**: Jonathan Ho, Ajay Jain, Pieter Abbeel  
**发表**: NeurIPS 2020

**核心贡献**:
- 复兴了扩散概率模型，并在图像生成任务上取得突破
- 揭示扩散模型与去噪分数匹配 (Denoising Score Matching) 之间的联系
- 在 CIFAR-10 上取得 FID 3.17 的先进结果

**技术要点**:
```
DDPM 包含两个过程：

前向扩散过程（固定）:
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_tI)

反向去噪过程（学习）:
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))

训练目标: 简化后的均方误差损失
L = E[||ε - ε_θ(x_t, t)||²]
```

**影响**: DDPM 开启了扩散模型的新时代，直接催生了 Stable Diffusion、DALL-E 2、Imagen 等影响力巨大的生成系统。

---

### 5. DDIM - Denoising Diffusion Implicit Models (2020)

**论文**: [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)  
**作者**: Jiaming Song 等  
**发表**: ICLR 2021

**核心贡献**:
- 提出非马尔可夫扩散过程，加速采样
- 可将采样步数从 1000 步减少到 50 步，速度提升 10-50 倍
- 支持确定性和随机性采样之间的平滑插值
- 实现潜在空间中的语义有意义的图像插值

**技术要点**:
```
DDIM 的核心思想：
- 构造一类非马尔可夫扩散过程
- 保持与 DDPM 相同的训练目标
- 反向过程可以更快采样

采样速度: 10× 到 50× 快于 DDPM
FID 质量: 相当
```

**影响**: DDIM 使得扩散模型在实践中的部署成为可能，是后续蒸馏和加速方法的重要基础。

---

### 6. Improved DDPM (2021)

**论文**: [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)  
**作者**: Prafulla Dhariwal, Alex Nichol  
**发表**: NeurIPS 2021

**核心贡献**:
- 通过简单修改使 DDPM 达到竞争性的对数似然
- 学习反向扩散过程的方差，进一步减少采样步数
- 使用精确率和召回率 (Precision & Recall) 比较 DDPM 和 GAN 对目标分布的覆盖

---

### 7. Score-based Generative Models

Score-based 模型是与扩散模型密切相关但独立发展的方向，核心思想是估计数据分布的分数（对数密度的梯度）。

#### 7.1 NCSN - Noise Conditional Score Networks (2019)

**论文**: [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)  
**作者**: Yang Song, Stefano Ermon  
**发表**: NeurIPS 2019

**核心贡献**:
- 提出通过估计分数函数来学习数据分布
- 使用多尺度噪声条件分数网络处理不同尺度下的分布
- 使用 Langevin 动力学进行采样

#### 7.2 NCSN++ / Score SDE (2021)

**论文**: [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)  
**作者**: Yang Song 等  
**发表**: ICLR 2021

**核心贡献**:
- 将扩散过程统一为随机微分方程 (SDE) 框架
- 扰动核可以表示为 SDE 的转移核
- 统一了 SMLD (Score Matching with Langevin Dynamics) 和 DDPM 框架

**影响**: Score-based 模型与扩散模型的理论统一，为后续 Flow Matching 的发展提供了数学基础。

---

### 8. Latent Diffusion Models / Stable Diffusion (2022)

**论文**: [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)  
**作者**: Robin Rombach 等  
**发表**: CVPR 2022

**核心贡献**:
- 将扩散模型应用于预训练自编码器的潜在空间
- 首次在复杂度降低和细节保留之间达到近最优点
- 引入交叉注意力层，支持文本、边界框等条件输入
- 在图像修复、语义场景合成、超分辨率等任务上取得 SOTA

**技术要点**:
```
LDM 架构：
1. 预训练 VQ-VAE 或 VAE 将图像编码到潜在空间
2. 在潜在空间训练扩散模型（U-Net）
3. 通过交叉注意力注入条件信息
4. 解码器将潜在表示还原为图像

优势：
- 计算需求显著降低
- 支持高分辨率生成
- 灵活的条件控制
```

**影响**: LDM（即 Stable Diffusion）是当今最广泛使用的开源图像生成模型，推动了 AI 生成内容的民主化。

---

## Flow Matching 与 Rectified Flow

### 9. Flow Matching (2022)

**论文**: [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)  
**作者**: Yaron Lipman 等  
**发表**: ICLR 2023

**核心贡献**:
- 提出 Flow Matching (FM) 新范式，用于训练连续归一化流 (CNF)
- 基于回归固定条件概率路径的向量场，实现无需模拟的训练
- 与一般的扩散路径兼容，同时支持非扩散概率路径
- 使用最优传输 (OT) 位移插值定义条件概率路径，比扩散路径更高效

**技术要点**:
```
Flow Matching 核心：
- 目标：学习从噪声分布 p_0 到数据分布 p_1 的流
- 方法：回归向量场 u_t，对应条件概率路径 p_t
- 关键：使用条件流匹配目标，避免 costly 的模拟

概率路径选择：
1. 扩散路径（DDPM 的特例）
2. 最优传输路径（推荐）：更快的训练和采样
```

**实验结果**:
- 在 ImageNet 上，Flow Matching 在似然和样本质量上均优于扩散方法
- 使用现成 ODE 求解器实现快速可靠的采样

**影响**: Flow Matching 为生成模型提供了更简洁、更高效的数学框架，是后续 Rectified Flow 和 Stable Diffusion 3 的基础。

---

### 10. Rectified Flow (2022-2023)

**论文**: [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)  
**作者**: Xingchao Liu 等  
**发表**: ICLR 2023

**核心贡献**:
- 提出 Rectified Flow，学习几乎直线的概率流
- 实现一步或少步生成，同时保持生成质量
- 可以迭代"整流"，进一步减少采样步数

**技术要点**:
```
Rectified Flow 原理：
- 标准流：曲线轨迹，需要多步 ODE 求解
- Rectified Flow：近似直线路径，支持大步长甚至单步采样

整流过程：
1. 训练初始流模型
2. 使用学习到的流生成配对数据 (z_0, z_1)
3. 在新数据上重新训练流
4. 迭代使流越来越直
```

**影响**: Rectified Flow 是最近生成模型加速的重要方向，InstaFlow、SDXL Turbo 等快速采样方法都受其启发。

---

## 离散扩散语言模型

将扩散模型应用于离散数据（如文本）面临独特挑战。近年来，离散扩散语言模型取得了显著进展。

### 11. D3PM - Diffusion Models for Discrete Data (2021)

**论文**: [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)  
**作者**: Jacob Austin 等  
**发表**: NeurIPS 2021

**核心贡献**:
- 将扩散模型扩展到离散状态空间
- 定义离散数据的"去噪"过程
- 提出多种转移矩阵设计（吸收态、均匀、离散余弦等）

---

### 12. SEDD - Score Entropy Discrete Diffusion (2023)

**论文**: [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)  
**作者**: Aaron Lou, Chenlin Meng, Stefano Ermon  
**发表**: ICML 2024 (Oral)

**核心贡献**:
- 提出 Score Entropy 损失，将分数匹配自然扩展到离散空间
- 在语言建模任务上显著优于现有扩散范式
- 与自回归模型竞争，在某些方面超过 GPT-2

**技术要点**:
```
SEDD 创新：
- 问题：标准扩散依赖分数匹配理论，离散空间推广未取得同等经验收益
- 解决方案：Score Entropy 损失，自然扩展分数匹配到离散空间

实验结果（相比现有扩散方法）：
- 困惑度降低 25-75%
- 与 GPT-2 规模相当的模型性能竞争
- 无需温度缩放等技术即可生成忠实文本
- 支持计算与质量的权衡（少 32 倍网络评估获得相似质量）
- 支持可控填充（匹配核采样质量，支持非左到右提示）
```

**影响**: SEDD 证明了扩散模型在语言建模上的潜力，是离散扩散领域的重要里程碑。

---

### 13. MDLM - Masked Diffusion Language Models (2024)

**论文**: [Masked Diffusion Language Models](https://arxiv.org/abs/2406.07516)  
**作者**: Subham Sekhar Sahoo 等

**核心贡献**:
- 将掩码语言建模与连续扩散模型统一
- 提出简化的训练目标，与标准语言模型预训练兼容
- 在零样本生成任务上取得竞争性能

---

### 14. BD3-LMs - Bayesian Discrete Diffusion for Language Models

**相关论文**: [Discrete Diffusion Language Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) 等

**核心思想**:
- 将离散扩散建模为贝叶斯推断问题
- 通过估计数据分布的比率来学习去噪
- 结合大规模预训练语言模型的先验知识

---

## Kaiming He 近期工作 (2024-2026)

Kaiming He（何恺明）作为深度学习领域的领军人物，其近期工作聚焦于生成模型的根本性问题和新范式探索。

### 15. MAR - Autoregressive Image Generation without Vector Quantization (2024)

**论文**: [arXiv:2406.11838](https://arxiv.org/abs/2406.11838)  
**作者**: Tianhong Li 等 (Kaiming He 组)  
**发表**: NeurIPS 2024 (Spotlight)  
**代码**: [github.com/LTH14/mar](https://github.com/LTH14/mar)

**核心贡献**:
- 挑战传统认知：自回归图像生成必须配合向量量化
- 观察发现：离散值空间有助于表示分类分布，但不是自回归建模的必要条件
- 提出使用扩散过程建模每 token 的概率分布，允许在连续值空间应用自回归模型
- 定义 Diffusion Loss 替代分类交叉熵损失

**技术要点**:
```
MAR 核心创新：
- 移除向量量化 (VQ)，在连续空间进行自回归建模
- 每个 token 的概率分布通过扩散过程建模
- 支持标准自回归模型和广义掩码自回归 (MAR) 变体

优势：
- 无需离散值 tokenizer
- 序列建模的速度优势
- 与量化方法相当或更好的生成质量
```

**实验结果**:
- 在 ImageNet 基准上取得强性能
- 生成质量与量化自回归模型相当
- 享受序列建模的速度优势

**影响**: MAR 为图像生成提供了新的自回归范式，可能对语言和视觉的统一建模产生深远影响。

---

### 16. 其他相关探索

**生成模型与表示学习的统一**:  
Kaiming He 组持续关注生成模型与自监督表示学习的结合，探索 MAE (Masked Autoencoder) 与生成模型的融合方向。

---

## 效率优化与压缩 (Song Han 组)

### DC-AE - Deep Compression Autoencoder (2024)

**论文**: [arXiv:2410.10733](https://arxiv.org/abs/2410.10733)  
**作者**: Han Cai 等 (MIT, Song Han 组)  
**发表**: ICLR 2025

**核心贡献**:
- 提出深度压缩自编码器 (DC-AE) 新家族
- 解决高空间压缩比（如 64x）下重建精度不足的问题
- 实现最高 128x 空间压缩比，同时保持重建质量

**技术要点**:
```
DC-AE 关键技术：
1. 残差自编码 (Residual Autoencoding):
   - 基于空间到通道变换的特征学习残差
   - 缓解高空间压缩自编码器的优化困难

2. 解耦高分辨率适应 (Decoupled High-Resolution Adaptation):
   - 三阶段解耦训练策略
   - 减轻高空间压缩自编码器的泛化惩罚
```

**实验结果**:
- ImageNet 512×512:
  - 19.1× 推理速度提升
  - 17.9× 训练速度提升
  - 相比 SD-VAE-f8 取得更好的 FID

**影响**: DC-AE 为高效图像生成提供了关键基础设施，是实用化生成模型系统的重要组成部分。

---

## 研究趋势与洞察

### 趋势 1: 从扩散到流 - 生成模型的数学统一

**演进路径**:  
DDPM (2020) → Score SDE (2021) → Flow Matching (2022) → Rectified Flow (2023)

**核心洞察**:
- 扩散模型可以统一在 SDE/ODE 框架下
- Flow Matching 提供了更简洁的数学表述
- Rectified Flow 解决了采样效率问题

**未来方向**:
- 一步生成模型的实用化
- 流模型的理论深入（最优传输、黎曼几何）

---

### 趋势 2: 离散扩散 - 语言模型的下一个范式?

**演进路径**:  
D3PM (2021) → SEDD (2023) → MDLM (2024) → BD3-LMs (2024+)

**核心洞察**:
- 自回归模型面临并行化和全局依赖的挑战
- 扩散模型天然支持并行采样和可控生成
- 离散空间需要新的数学工具（Score Entropy）

**挑战与机遇**:
- **挑战**: 离散扩散的采样效率仍低于自回归
- **机遇**: 可控生成、填充、编辑任务上的独特优势
- **展望**: 可能与自回归方法形成互补，而非替代

---

### 趋势 3: 自回归 vs 扩散 - 图像生成的范式竞争

**代表性工作**:
- 扩散派: Stable Diffusion, DALL-E 3, Imagen
- 自回归派: Parti, Muse, MAR

**对比**:

| 维度 | 扩散模型 | 自回归模型 |
|------|----------|------------|
| 采样并行度 | 逐步去噪 | 逐步生成 |
| 全局一致性 | 迭代优化 | 顺序依赖 |
| 可控性 | 强 | 中等 |
| 与语言模型统一 | 较难 | 较易 |

**MAR 的意义**:  
MAR 尝试结合两者的优势——自回归的结构 + 连续空间的扩散建模。这可能是图像-语言统一建模的重要一步。

---

### 趋势 4: 压缩与效率 - 实用化的关键

**核心方向**:
- **蒸馏**: 将多步扩散模型压缩为少步或单步模型
- **量化**: 降低模型精度以加速推理
- **压缩**: 如 DC-AE 的高倍率空间压缩
- **架构优化**: DiT (Diffusion Transformer)、线性注意力等

**代表性技术**:
- Progressive Distillation
- Consistency Models
- LCM (Latent Consistency Models)
- DC-AE

---

### 趋势 5: 多模态统一生成模型

**愿景**: 单一模型统一处理文本、图像、音频、视频等多种模态

**技术路线**:
- 将各种模态 token 化（VQ-VAE, VQ-GAN）
- 统一的自回归或扩散框架
- 跨模态的联合建模

**代表性工作**:
- GPT-4V (视觉-语言)
- Gemini (多模态)
- Sora (视频生成)
- MAR (图像自回归)

---

## 关键洞察总结

1. **范式演进**: 生成模型经历了 GAN → VAE/Flow → Diffusion → Flow Matching 的演进，每一代都在解决前一代的核心问题，同时引入新的挑战。

2. **数学统一**: Flow Matching 和 Score-based 框架提供了理解各种生成模型的统一视角，有望推动理论进一步发展。

3. **效率革命**: 采样效率是扩散模型实用化的关键瓶颈，Rectified Flow、蒸馏、一致性模型等方向正在取得突破。

4. **语言模型的新可能**: 离散扩散为语言生成提供了不同于自回归的范式，在可控性和编辑任务上有独特优势，但效率仍是挑战。

5. **统一建模趋势**: 图像生成的自回归化（MAR）和语言生成的扩散化（SEDD）暗示了两条路径可能在未来收敛。

---

## 参考资源

### 论文链接汇总

**奠基性工作**:
- VAE: [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- GAN: [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- RealNVP: [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
- Glow: [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)
- FFJORD: [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)

**扩散模型**:
- DDPM: [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- DDIM: [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Improved DDPM: [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- Score SDE: [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- LDM/Stable Diffusion: [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

**Flow Matching**:
- Flow Matching: [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Rectified Flow: [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)

**离散扩散**:
- D3PM: [arXiv:2107.03006](https://arxiv.org/abs/2107.03006)
- SEDD: [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)
- MDLM: [arXiv:2406.07516](https://arxiv.org/abs/2406.07516)

**Kaiming He 近期工作**:
- MAR: [arXiv:2406.11838](https://arxiv.org/abs/2406.11838) | [代码](https://github.com/LTH14/mar)

**Song Han 组 (效率优化)**:
- DC-AE: [arXiv:2410.10733](https://arxiv.org/abs/2410.10733)

---

*调研完成于 2026-02-17 by [Amy](https://github.com/amysheng-ai)*
