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
3. [扩散模型革命](#扩散模型革命)
4. [Flow Matching 与 Rectified Flow](#flow-matching-与-rectified-flow)
5. [离散扩散语言模型](#离散扩散语言模型)
6. [Kaiming He 近期工作](#kaiming-he-近期工作)
7. [效率优化与压缩](#效率优化与压缩)
8. [研究趋势与洞察](#研究趋势与洞察)

---

## 引言

生成模型是深度学习的核心领域之一，旨在学习数据的潜在分布并生成新的、逼真的样本。从早期的 VAE 和 GAN，到近年来主导生成式 AI 的扩散模型，再到最新兴起的 Flow Matching 和离散扩散语言模型，该领域经历了多次范式转变。

本文按照时间线和主题分类，系统梳理了生成模型的重要工作，帮助读者理解各方法的核心思想、优缺点以及演进关系。

---

## 奠基性工作

### 1. VAE - Auto-Encoding Variational Bayes (2013)

#### Meta
- **Title**: Auto-Encoding Variational Bayes
- **Link**: [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- **Venue**: ICLR 2014
- **Date**: 2013-12
- **Tags**: [VAE, Variational Inference, Generative Models, Latent Variable Models]
- **Authors**: Diederik P. Kingma, Max Welling
- **TL;DR**: 提出变分自编码器框架，通过重参数化技巧实现端到端的变分推断，学习深度潜在变量模型。

#### Problem & Contribution
- **解决的问题**: 如何在深度神经网络框架下进行可扩展的变分推断，处理难以计算的后验分布 $p(z|x)$
- **核心想法/方法一句话**: 使用识别网络 $q_\phi(z|x)$ 近似后验，通过重参数化技巧实现梯度传播
- **主要贡献**:
  1. 提出变分自编码器 (VAE) 统一框架，结合深度学习和变分推断
  2. 引入重参数化技巧 (Reparameterization Trick): $z = \mu(x) + \sigma(x) \cdot \epsilon$, 其中 $\epsilon \sim \mathcal{N}(0, I)$
  3. 使端到端训练复杂生成模型成为可能

#### Method
- **方法结构/流程**: VAE 包含编码器网络 $q_\phi(z|x)$ 和解码器网络 $p_\theta(x|z)$，联合优化变分下界 (ELBO)
- **关键设计**: 
  - 编码器将数据映射到潜在分布参数
  - 解码器从采样潜在变量重构数据
  - 重参数化实现随机节点的梯度传播
- **数学公式**:
  - ELBO: $\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$
  - 联合目标: $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left[ \mathbb{E}_{q_\phi(z|x^{(i)})}[\log p_\theta(x^{(i)}|z)] - D_{KL}(q_\phi(z|x^{(i)}) \| p(z)) \right]$$
  - 后验近似: $q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma^2_\phi(x))$

#### Evidence
- **Benchmark / setting**: MNIST, Frey Face 等图像数据集
- **对比对象**: 传统变分推断方法 (Mean-Field VI)
- **关键结果**: 首次实现基于神经网络的生成模型端到端训练，学习有意义的潜在表示

#### Takeaways
- **核心洞察**: 重参数化技巧是连接深度学习和概率推断的关键桥梁
- **影响与意义**: 奠定了深度生成模型的基础，启发后续大量工作 (CVAE, $\beta$-VAE, VQ-VAE 等)
- **局限性**: 生成样本质量相对较模糊，后验坍缩 (Posterior Collapse) 问题

---

### 2. GAN - Generative Adversarial Networks (2014)

#### Meta
- **Title**: Generative Adversarial Networks
- **Link**: [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- **Venue**: NeurIPS 2014
- **Date**: 2014-06
- **Tags**: [GAN, Adversarial Training, Generative Models, Minimax Game]
- **Authors**: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- **TL;DR**: 提出对抗训练框架，通过生成器和判别器的博弈学习数据分布。

#### Problem & Contribution
- **解决的问题**: 传统生成模型依赖马尔可夫链或近似推断网络，采样效率低
- **核心想法/方法一句话**: 通过生成器 $G$ 和判别器 $D$ 的对抗博弈，直接学习从噪声到数据的映射
- **主要贡献**:
  1. 开创性地提出对抗训练框架
  2. 无需马尔可夫链即可高效生成样本
  3. 引入极小极大博弈作为训练目标

#### Method
- **方法结构/流程**: 同时训练两个网络：生成器 $G$ 学习从先验噪声 $p_z(z)$ 生成样本，判别器 $D$ 区分真实样本和生成样本
- **关键设计**: 
  - 生成器: $G(z; \theta_g)$ 将噪声 $z$ 映射到数据空间
  - 判别器: $D(x; \theta_d)$ 输出样本为真的概率
  - 交替优化两个网络的参数
- **数学公式**:
  - 价值函数: $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$
  - 生成器目标: $\mathcal{L}_G = -\mathbb{E}_{z \sim p_z(z)}[\log D(G(z))]$ (非饱和版本)
  - 判别器目标: $\mathcal{L}_D = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$

#### Evidence
- **Benchmark / setting**: MNIST, CIFAR-10, Toronto Face Database
- **对比对象**: DBN (Deep Belief Networks), Deep NADE, VAE
- **关键结果**: 在 MNIST 和 TFD 上取得更低的测试集对数似然 (更优)，CIFAR-10 上生成高质量样本

#### Takeaways
- **核心洞察**: 对抗博弈可以隐式学习目标分布，避免显式密度估计
- **影响与意义**: 开启生成模型新纪元，催生 DCGAN、StyleGAN、BigGAN 等重要工作
- **局限性**: 训练不稳定、模式崩溃 (Mode Collapse)、难以评估

---

### 3. NICE / RealNVP - Normalizing Flow (2014-2017)

#### Meta
- **Title**: Density estimation using Real NVP
- **Link**: [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
- **Venue**: ICLR 2017
- **Date**: 2016-05
- **Tags**: [Normalizing Flow, Invertible Networks, Exact Likelihood, Coupling Layers]
- **Authors**: Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
- **TL;DR**: 提出可逆神经网络架构，通过耦合层实现可逆变换，支持精确对数似然估计。

#### Problem & Contribution
- **解决的问题**: 如何在保持可逆性的同时构建表达能力强的生成模型，支持精确似然计算
- **核心想法/方法一句话**: 使用耦合层 (Coupling Layers) 构建三角雅可比矩阵，使行列式计算可在 $O(D)$ 时间完成
- **主要贡献**:
  1. 提出 Real-valued Non-Volume Preserving (RealNVP) 变换
  2. 设计耦合层实现高效可逆变换
  3. 支持精确对数似然和高效采样

#### Method
- **方法结构/流程**: 将输入维度分为两部分，一部分直接通过，另一部分通过可逆变换依赖于第一部分
- **关键设计**: 
  - 耦合层: $y_{1:d} = x_{1:d}$, $y_{d+1:D} = x_{d+1:D} \odot \exp(s(x_{1:d})) + t(x_{1:d})$
  - 缩放函数 $s$ 和平移函数 $t$ 可以是任意神经网络
  - 多尺度架构逐步处理高维数据
- **数学公式**:
  - 变量变换公式: $$p_X(x) = p_Z(f(x)) \left| \det \frac{\partial f(x)}{\partial x} \right|$$
  - 对数似然: $$\log p_X(x) = \log p_Z(f(x)) + \sum_{i=1}^{d} s_i(x_{1:d})$$
  - 雅可比行列式: $\det \frac{\partial y}{\partial x} = \exp\left(\sum_{j} s(x_{1:d})_j\right)$
  - 逆变换: $x_{d+1:D} = (y_{d+1:D} - t(y_{1:d})) \odot \exp(-s(y_{1:d}))$

#### Evidence
- **Benchmark / setting**: CIFAR-10, ImageNet, CelebA, LSUN
- **对比对象**: VAE, GAN, PixelCNN
- **关键结果**: CIFAR-10 测试集对数似然 3.49 bits/dim，图像生成质量接近 GAN

#### Takeaways
- **核心洞察**: 三角雅可比结构可以在可逆性和表达能力之间取得平衡
- **影响与意义**: 为后续 Glow、FFJORD 和流匹配奠定基础
- **局限性**: 网络架构受限，表达能力不如无约束网络

---

### 4. Glow - Generative Flow with Invertible 1×1 Convolutions (2018)

#### Meta
- **Title**: Glow: Generative Flow with Invertible 1×1 Convolutions
- **Link**: [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)
- **Venue**: NeurIPS 2018
- **Date**: 2018-07
- **Tags**: [Normalizing Flow, Invertible Convolutions, Image Generation, Exact Likelihood]
- **Authors**: Diederik P. Kingma, Prafulla Dhariwal
- **TL;DR**: 引入可逆 1×1 卷积扩展 Normalizing Flow，在保持可逆性的同时显著增强表达能力。

#### Problem & Contribution
- **解决的问题**: RealNVP 中维度划分固定导致表达能力受限
- **核心想法/方法一句话**: 使用可逆 1×1 卷积替代固定划分，实现维度间的信息混合
- **主要贡献**:
  1. 提出可逆 1×1 卷积层
  2. 支持有意义的潜在空间插值和操作
  3. 在图像生成上取得与 GAN 竞争的性能

#### Method
- **方法结构/流程**: 结合 ActNorm (可逆激活归一化)、可逆 1×1 卷积和耦合层构建流
- **关键设计**: 
  - ActNorm: 数据依赖初始化，类似批归一化但可逆
  - 可逆 1×1 卷积: $y = Wx$, 其中 $W$ 是可逆方阵
  - 保持 RealNVP 的耦合层结构
- **数学公式**:
  - 1×1 卷积对数行列式: $\log |\det(W)| = h \cdot w \cdot \log |\det(W)|$
  - LU 分解加速: $W = PL(U + \text{diag}(s))$, 其中 $P$ 是置换矩阵，$L$ 是下三角，$U$ 是上三角
  - 变量变换: $$\log p_X(x) = \log p_Z(z) + \sum_{l=1}^{L} \log |\det(J_l)|$$

#### Evidence
- **Benchmark / setting**: ImageNet 32×32, ImageNet 64×64, 5-bit CelebA-HQ 256×256
- **对比对象**: RealNVP, PixelCNN++, GAN
- **关键结果**: 
  - ImageNet 64×64: 3.81 bits/dim
  - 高质量人脸生成，支持属性操作 (如改变年龄、微笑)

#### Takeaways
- **核心洞察**: 可逆卷积可以在保持可逆性的同时实现通道间信息混合
- **影响与意义**: 展示 Normalizing Flow 可以生成高质量图像
- **局限性**: 计算成本较高，难以扩展到非常高维数据

---

### 5. FFJORD - Free-form Jacobian of Reversible Dynamics (2018)

#### Meta
- **Title**: FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models
- **Link**: [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)
- **Venue**: ICLR 2019
- **Date**: 2018-10
- **Tags**: [Continuous Normalizing Flow, ODE, Neural ODE, Hutchinson Trace Estimator]
- **Authors**: Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, David Duvenaud
- **TL;DR**: 使用神经网络定义的常微分方程 (ODE) 描述连续时间流，通过 Hutchinson 迹估计器实现可扩展训练。

#### Problem & Contribution
- **解决的问题**: 离散 Normalizing Flow 受限于特定架构约束，表达能力受限
- **核心想法/方法一句话**: 用连续时间动力学替代离散变换，使用 ODE 求解器和迹估计实现无限制架构
- **主要贡献**:
  1. 提出自由形式连续动态，无架构约束
  2. 使用 Hutchinson 迹估计器计算对数密度变化
  3. 支持可扩展的无偏对数密度估计

#### Method
- **方法结构/流程**: 定义隐藏状态的 ODE: $\frac{dz(t)}{dt} = f(z(t), t; \theta)$，使用神经网络定义动力学函数
- **关键设计**: 
  - 连续时间变换通过 ODE 求解器实现
  - Hutchinson 迹估计: $\text{Tr}\left(\frac{\partial f}{\partial z}\right) = \mathbb{E}_{p(\epsilon)}[\epsilon^T \frac{\partial f}{\partial z} \epsilon]$
  - 支持任意神经网络架构
- **数学公式**:
  - 连续流定义: $$\frac{\partial z(t)}{\partial t} = f(z(t), t; \theta)$$
  - 对数密度变化 (Instantaneous Change of Variables): $$\frac{\partial \log p(z(t))}{\partial t} = -\text{Tr}\left(\frac{\partial f}{\partial z(t)}\right)$$
  - 迹估计: $$\text{Tr}\left(\frac{\partial f}{\partial z}\right) \approx \frac{1}{L} \sum_{l=1}^{L} \epsilon_l^T \frac{\partial f}{\partial z} \epsilon_l$$
  - 似然计算: $$\log p(x) = \log p(z) - \int_{t_0}^{t_1} \text{Tr}\left(\frac{\partial f}{\partial z(t)}\right) dt$$

#### Evidence
- **Benchmark / setting**: MNIST, CIFAR-10, ImageNet 64×64
- **对比对象**: RealNVP, Glow, Residual Flow
- **关键结果**: 与 Glow 相当的性能，但使用更灵活的架构

#### Takeaways
- **核心洞察**: 连续时间视角解放了架构约束，是扩散模型理论的先驱
- **影响与意义**: 为 Score-based 模型和 Flow Matching 奠定理论基础
- **局限性**: ODE 求解器计算开销大，训练较慢

---

## 扩散模型革命

### 6. DDPM - Denoising Diffusion Probabilistic Models (2020)

#### Meta
- **Title**: Denoising Diffusion Probabilistic Models
- **Link**: [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- **Venue**: NeurIPS 2020
- **Date**: 2020-06
- **Tags**: [Diffusion Models, DDPM, Denoising, Score Matching, Image Generation]
- **Authors**: Jonathan Ho, Ajay Jain, Pieter Abbeel
- **TL;DR**: 复兴扩散概率模型，揭示其与去噪分数匹配的联系，在图像生成上取得突破。

#### Problem & Contribution
- **解决的问题**: 早期扩散模型训练困难，样本质量不佳
- **核心想法/方法一句话**: 使用重参数化和简化目标函数训练去噪模型，直接预测噪声而非均值
- **主要贡献**:
  1. 揭示扩散模型与去噪分数匹配的等价性
  2. 提出简化的均方误差训练目标
  3. 在 CIFAR-10 和 LSUN 上取得 SOTA 生成质量

#### Method
- **方法结构/流程**: 定义前向加噪过程和反向去噪过程，训练神经网络预测噪声
- **关键设计**: 
  - 前向过程: 固定马尔可夫链逐渐加噪
  - 反向过程: 学习去噪分布
  - 简化的训练目标直接预测噪声
- **数学公式**:
  - 前向过程: $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$
  - 重参数化: $x_t(x_0, \epsilon) = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$, 其中 $\epsilon \sim \mathcal{N}(0, I)$
  - 简化目标: $$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$$
  - 反向过程均值: $$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

#### Evidence
- **Benchmark / setting**: CIFAR-10, CelebA-HQ, LSUN Church/Bedroom
- **对比对象**: GAN, VAE, Flow-based models, PixelCNN
- **关键结果**: 
  - CIFAR-10: FID 3.17 (无条件)
  - CelebA-HQ 256×256: 高质量人脸生成
  - 首次展示扩散模型可以生成与 GAN 竞争的高质量图像

#### Takeaways
- **核心洞察**: 扩散模型的成功关键在于简化的训练目标和与分数匹配的理论联系
- **影响与意义**: 开启扩散模型新时代，直接催生 Stable Diffusion、DALL-E 2、Imagen 等
- **局限性**: 采样需要 1000 步，速度缓慢

---

### 7. DDIM - Denoising Diffusion Implicit Models (2020)

#### Meta
- **Title**: Denoising Diffusion Implicit Models
- **Link**: [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- **Venue**: ICLR 2021
- **Date**: 2020-10
- **Tags**: [Diffusion Models, Fast Sampling, Non-Markovian, Implicit Models]
- **Authors**: Jiaming Song, Chenlin Meng, Stefano Ermon
- **TL;DR**: 提出非马尔可夫扩散过程，实现 10-50 倍采样加速，同时保持生成质量。

#### Problem & Contribution
- **解决的问题**: DDPM 采样需要 1000 步，速度极慢，难以实际应用
- **核心想法/方法一句话**: 构造一类非马尔可夫扩散过程，保持相同训练目标但允许更快采样
- **主要贡献**:
  1. 提出非马尔可夫前向过程
  2. 将采样步数从 1000 减少到 50 步，速度提升 10-50 倍
  3. 支持确定性和随机性采样之间的平滑插值

#### Method
- **方法结构/流程**: 修改反向过程为一组隐式概率模型，使用跳跃式采样
- **关键设计**: 
  - 定义非马尔可夫前向过程: $q_\sigma(x_{t-1} | x_t, x_0)$
  - 反向过程使用调整后的方差参数 $\sigma$
  - 当 $\sigma = 0$ 时退化为确定性采样
- **数学公式**:
  - 反向过程分布: $$p_\theta^{(t)}(x_{t-1} | x_t) = \begin{cases} \mathcal{N}(f_\theta^{(1)}(x_1), \sigma_1^2 I) & \text{if } t = 1 \\ q_\sigma(x_{t-1} | x_t, f_\theta^{(t)}(x_t)) & \text{otherwise} \end{cases}$$
  - 生成过程: $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta^{(t)}(x_t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的 } x_0} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta^{(t)}(x_t) + \sigma_t \epsilon_t$$
  - 确定性采样 ($\sigma_t = 0$): $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} f_\theta^{(t)}(x_t) + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta^{(t)}(x_t)$$

#### Evidence
- **Benchmark / setting**: CIFAR-10, CelebA, ImageNet 64×64
- **对比对象**: DDPM, NCSN, PixelCNN
- **关键结果**: 
  - 50 步采样: FID 4.67 (vs DDPM 1000 步 FID 3.17)
  - 10 步采样仍保持可接受的样本质量
  - 确定性采样支持语义有意义的潜在空间插值

#### Takeaways
- **核心洞察**: 扩散模型的前向过程可以有多种选择，非马尔可夫构造实现加速
- **影响与意义**: 使扩散模型实用化，是后续蒸馏和加速方法的基础
- **局限性**: 少于 50 步时质量明显下降

---

### 8. Improved DDPM (2021)

#### Meta
- **Title**: Diffusion Models Beat GANs on Image Synthesis
- **Link**: [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- **Venue**: NeurIPS 2021
- **Date**: 2021-02
- **Tags**: [Diffusion Models, Image Synthesis, Learned Variance, Noise Schedule]
- **Authors**: Prafulla Dhariwal, Alex Nichol
- **TL;DR**: 通过简单修改 (学习方差、改进架构、余弦噪声 schedule) 使 DDPM 超越 GAN 的图像生成质量。

#### Problem & Contribution
- **解决的问题**: DDPM 样本质量落后于 GAN，尤其是多样性方面
- **核心想法/方法一句话**: 联合学习均值和方差，改进噪声 schedule 和模型架构
- **主要贡献**:
  1. 学习反向扩散过程的方差，进一步减少采样步数
  2. 设计余弦噪声 schedule 改善训练稳定性
  3. 在 ImageNet 上首次超越 StyleGAN2 的 FID

#### Method
- **方法结构/流程**: 扩展 DDPM 目标同时学习均值和方差，改进模型架构
- **关键设计**: 
  - 混合目标: $\mathcal{L}_{hybrid} = \mathcal{L}_{simple} + \lambda \mathcal{L}_{vlb}$
  - 余弦噪声 schedule: $\bar{\alpha}_t = \frac{f(t)}{f(0)}$, $f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$
  - 架构改进: 增加深度、宽度，使用 BigGAN 的残差块
- **数学公式**:
  - 混合目标: $$\mathcal{L}_{hybrid} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2 + \lambda \cdot D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))\right]$$
  - 学习方差: $$\Sigma_\theta(x_t, t) = \exp(v \log \beta_t + (1 - v) \log \tilde{\beta}_t)$$, $v$ 由网络预测
  - 余弦 schedule 平滑了信号和噪声的比例变化

#### Evidence
- **Benchmark / setting**: ImageNet 128×128, ImageNet 256×256, ImageNet 512×512
- **对比对象**: StyleGAN2, BigGAN, DDPM, VQ-VAE-2
- **关键结果**: 
  - ImageNet 128×128: FID 2.97 (超越 StyleGAN2 的 3.80)
  - ImageNet 256×256: FID 4.59 (超越 StyleGAN2 的 5.91)
  - ImageNet 512×512: FID 7.72
  - 在 Precision 和 Recall 指标上都优于 GAN

#### Takeaways
- **核心洞察**: 简单的工程改进 (方差学习、噪声 schedule) 可以显著提升扩散模型性能
- **影响与意义**: 证明扩散模型可以全面超越 GAN，确立其主流地位
- **局限性**: 训练计算成本仍然很高

---

### 9. NCSN - Noise Conditional Score Networks (2019)

#### Meta
- **Title**: Generative Modeling by Estimating Gradients of the Data Distribution
- **Link**: [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
- **Venue**: NeurIPS 2019
- **Date**: 2019-07
- **Tags**: [Score Matching, Langevin Dynamics, Score Networks, Generative Models]
- **Authors**: Yang Song, Stefano Ermon
- **TL;DR**: 提出通过估计数据分布的分数（对数密度梯度）来学习生成模型，使用噪声条件分数网络处理多尺度分布。

#### Problem & Contribution
- **解决的问题**: 传统密度估计在高维数据上困难，MCMC 采样缓慢
- **核心想法/方法一句话**: 估计分数函数 $\nabla_x \log p(x)$ 而非密度本身，使用 Langevin 动力学采样
- **主要贡献**:
  1. 提出噪声条件分数网络 (NCSN) 估计多尺度分数
  2. 使用退火 Langevin 动力学进行采样
  3. 避免直接密度估计的困难

#### Method
- **方法结构/流程**: 训练神经网络 $s_\theta(x, \sigma)$ 估计扰动数据分布的分数，使用 Langevin 动力学采样
- **关键设计**: 
  - 分数匹配目标: 学习分数函数而非密度
  - 多尺度噪声: 处理不同尺度下的分布
  - 退火采样: 从粗到细逐步采样
- **数学公式**:
  - 分数函数: $$s_\theta(x) \approx \nabla_x \log p(x)$$
  - 退火 Langevin 动力学: $$x_{i+1} = x_i + \frac{\alpha_i}{2} s_\theta(x_i, \sigma_i) + \sqrt{\alpha_i} z_i$$
  - 去噪分数匹配目标: $$\mathcal{L}_{DSM} = \mathbb{E}_{x \sim p_{data}, \tilde{x} \sim q_\sigma(\tilde{x}|x)}\left[\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x)\|^2\right]$$
  - 分数匹配等价于去噪: $$\nabla_x \log q_\sigma(x) = \frac{\mathbb{E}_{x_0 \sim p_{data}}[q_\sigma(x|x_0) \nabla_x \log q_\sigma(x|x_0)]}{q_\sigma(x)}$$

#### Evidence
- **Benchmark / setting**: CIFAR-10, CelebA, MNIST
- **对比对象**: PixelCNN++, MMDGAN, Spectral GAN
- **关键结果**: 
  - CIFH10: Inception Score 8.91
  - 证明分数匹配可以生成高质量图像

#### Takeaways
- **核心洞察**: 分数函数比密度函数更容易估计，且足以支持生成建模
- **影响与意义**: 与 DDPM 并列为扩散模型的理论基础
- **局限性**: 训练不稳定，需要仔细选择噪声水平

---

### 10. NCSN++ / Score SDE (2021)

#### Meta
- **Title**: Score-Based Generative Modeling through Stochastic Differential Equations
- **Link**: [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- **Venue**: ICLR 2021
- **Date**: 2020-11
- **Tags**: [Score-based Models, SDE, ODE, Diffusion Models, Continuous-time]
- **Authors**: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole
- **TL;DR**: 将扩散过程和分数匹配统一在随机微分方程 (SDE) 框架下，统一了 SMLD 和 DDPM。

#### Problem & Contribution
- **解决的问题**: SMLD 和 DDPM 看似不同，缺乏统一理论框架
- **核心想法/方法一句话**: 用 SDE 描述扩散过程，将两种方法统一在连续时间框架下
- **主要贡献**:
  1. 将扩散过程统一为 SDE 框架
  2. 证明 SDE 有对应的概率流 ODE，支持精确似然计算
  3. 统一了 SMLD 和 DDPM 框架

#### Method
- **方法结构/流程**: 定义前向 SDE 逐渐扰动数据，通过反向 SDE 或概率流 ODE 生成样本
- **关键设计**: 
  - 前向 SDE: $dx = f(x, t)dt + g(t)dw$
  - 反向 SDE: $dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$
  - 概率流 ODE (确定性): $dx = [f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)]dt$
- **数学公式**:
  - 前向 SDE: $$dx = f(x, t)dt + g(t)dw$$
  - 反向 SDE: $$dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)d\bar{w}$$
  - 概率流 ODE: $$dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt$$
  - Fokker-Planck 方程连接 SDE 和概率演化: $$\frac{\partial p_t(x)}{\partial t} = -\sum_{i=1}^{d} \frac{\partial}{\partial x_i}[f_i(x, t)p_t(x)] + \frac{1}{2}\sum_{i=1}^{d}\sum_{j=1}^{d}\frac{\partial^2}{\partial x_i \partial x_j}[g(t)^2p_t(x)]$$

#### Evidence
- **Benchmark / setting**: CIFAR-10, CelebA-HQ, ImageNet 32×32
- **对比对象**: DDPM, DDIM, NCSN
- **关键结果**: 
  - CIFAR-10: FID 2.20 (新 SOTA)
  - 首次实现扩散模型的精确似然计算
  - 展示可控的生成过程

#### Takeaways
- **核心洞察**: SDE/ODE 框架提供了理解扩散模型的统一视角
- **影响与意义**: 为 Flow Matching 和后续理论发展奠定基础
- **局限性**: SDE 求解计算开销大

---

### 11. Latent Diffusion Models / Stable Diffusion (2022)

#### Meta
- **Title**: High-Resolution Image Synthesis with Latent Diffusion Models
- **Link**: [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- **Venue**: CVPR 2022
- **Date**: 2021-12
- **Tags**: [Latent Diffusion, Stable Diffusion, Text-to-Image, VAE, Cross-Attention]
- **Authors**: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
- **TL;DR**: 在预训练自编码器的潜在空间训练扩散模型，实现计算效率和生成质量的平衡，支持文本到图像生成。

#### Problem & Contribution
- **解决的问题**: 像素空间扩散模型计算成本极高，难以扩展到高分辨率
- **核心想法/方法一句话**: 将图像编码到低维潜在空间，在潜在空间训练扩散模型
- **主要贡献**:
  1. 首次在复杂度降低和细节保留之间达到近最优点
  2. 引入交叉注意力层，支持文本、边界框等条件输入
  3. 在图像修复、语义场景合成等任务上取得 SOTA

#### Method
- **方法结构/流程**: 预训练 VQ-VAE 或 VAE 编码器，在潜在空间训练 U-Net 扩散模型
- **关键设计**: 
  - 感知压缩模型: 将图像 $x$ 编码为潜在表示 $z = \mathcal{E}(x)$
  - 在潜在空间训练扩散模型
  - 交叉注意力注入条件信息
- **数学公式**:
  - 感知损失: $$\mathcal{L}_{auto} = \mathcal{L}_{rec} + \lambda_{perc}\mathcal{L}_{perc} + \lambda_{reg}\mathcal{L}_{reg}$$
  - 潜在扩散目标: $$\mathcal{L}_{LDM} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|^2\right]$$
  - 交叉注意力条件: $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$
  - 其中 $Q = W_Q z_t$, $K = W_K \tau_\theta(y)$, $V = W_V \tau_\theta(y)$

#### Evidence
- **Benchmark / setting**: ImageNet, COCO, CelebA-HQ, LAION-400M
- **对比对象**: DALL-E, VQGAN, ADM
- **关键结果**: 
  - ImageNet 256×256: FID 3.60
  - COCO 文本到图像: FID 12.63
  - 相比像素空间扩散，计算需求显著降低

#### Takeaways
- **核心洞察**: 在合适的潜在空间进行建模可以大幅降低计算成本而不损失质量
- **影响与意义**: Stable Diffusion 是当今最广泛使用的开源图像生成模型
- **局限性**: 依赖预训练编码器的质量，条件对齐仍有提升空间

---

## Flow Matching 与 Rectified Flow

### 12. Flow Matching (2022)

#### Meta
- **Title**: Flow Matching for Generative Modeling
- **Link**: [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **Venue**: ICLR 2023
- **Date**: 2022-10
- **Tags**: [Flow Matching, Continuous Normalizing Flow, Optimal Transport, Vector Fields]
- **Authors**: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le
- **TL;DR**: 提出 Flow Matching 新范式，通过回归固定条件概率路径的向量场来训练 CNF，无需模拟。

#### Problem & Contribution
- **解决的问题**: CNF 训练需要模拟 ODE，计算成本高昂
- **核心想法/方法一句话**: 直接回归向量场，而非通过模拟优化，使用条件流匹配目标
- **主要贡献**:
  1. 提出 Flow Matching (FM) 训练范式
  2. 支持一般的概率路径，包括非扩散路径
  3. 使用最优传输路径比扩散路径更高效

#### Method
- **方法结构/流程**: 学习向量场 $v_t$ 生成从噪声到数据的概率流，通过回归固定条件路径实现
- **关键设计**: 
  - 条件流匹配: 使用固定条件概率路径
  - 概率路径选择: 扩散路径或最优传输路径
  - 避免 costly 的模拟训练
- **数学公式**:
  - 流定义: $$\frac{d\phi_t(x)}{dt} = v_t(\phi_t(x))$$
  - 流匹配目标: $$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, q(x_1)}\left[\|v_t(\phi_t(x_0)) - u_t(\phi_t(x_0)|x_1)\|^2\right]$$
  - 条件流匹配目标 (等价): $$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)}\left[\|v_t(x) - u_t(x|x_1)\|^2\right]$$
  - 最优传输路径: $$p_t(x|x_1) = \mathcal{N}(x; t x_1 + (1-t)x_0, \sigma^2)$$, $u_t(x|x_1) = x_1 - x_0$

#### Evidence
- **Benchmark / setting**: CIFAR-10, ImageNet 32×32, ImageNet 64×64
- **对比对象**: DDPM, Score SDE, FFJORD
- **关键结果**: 
  - ImageNet 64×64: FID 优于扩散方法
  - 使用 ODE 求解器实现快速可靠采样
  - 训练速度比模拟方法快几个数量级

#### Takeaways
- **核心洞察**: 直接回归向量场比通过模拟优化更高效，最优传输路径优于扩散路径
- **影响与意义**: 为生成模型提供更简洁的数学框架，是 Stable Diffusion 3 的基础
- **局限性**: 理论分析仍在发展中

---

### 13. Rectified Flow (2022-2023)

#### Meta
- **Title**: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
- **Link**: [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- **Venue**: ICLR 2023
- **Date**: 2022-09
- **Tags**: [Rectified Flow, Straight Flow, Fast Sampling, One-step Generation, ODE]
- **Authors**: Xingchao Liu, Chengyue Gong, Qiang Liu
- **TL;DR**: 提出 Rectified Flow 学习几乎直线的概率流，实现一步或少步生成，同时保持质量。

#### Problem & Contribution
- **解决的问题**: 扩散模型采样步数多，难以实现实时生成
- **核心想法/方法一句话**: 学习直线（或接近直线）的流路径，支持大步长甚至单步采样
- **主要贡献**:
  1. 提出 Rectified Flow 学习直线概率流
  2. 实现一步或少步高质量生成
  3. 可以迭代"整流"进一步减少采样步数

#### Method
- **方法结构/流程**: 训练初始流模型，使用学习到的流生成配对数据，迭代重新训练
- **关键设计**: 
  - 直线流: 轨迹近似直线
  - 整流过程: 迭代使流越来越直
  - 大步长采样: 单步或少数几步即可
- **数学公式**:
  - 直线流性质: $$\phi_t(x_0) \approx (1-t)x_0 + t x_1$$
  - 整流目标: $$\min_\theta \mathbb{E}_{x_0, x_1}\left[\int_0^1 \|\phi_t(x_0) - ((1-t)x_0 + t x_1)\|^2 dt\right]$$
  - 一步采样: $$x_1 = \phi_1(x_0) \approx x_0 + v_0(x_0)$$
  - 迭代整流: 使用 $x_0^{(k)}, x_1^{(k)} = \phi(x_0^{(k-1)})$ 作为新训练数据

#### Evidence
- **Benchmark / setting**: CIFAR-10, CelebA, LSUN
- **对比对象**: DDPM, DDIM, Flow Matching
- **关键结果**: 
  - 单步生成质量接近多步扩散
  - 迭代整流持续改善路径直线度
  - 显著加速采样过程

#### Takeaways
- **核心洞察**: 流路径的直线度与采样效率直接相关
- **影响与意义**: InstaFlow、SDXL Turbo 等快速采样方法的灵感来源
- **局限性**: 单步质量仍有提升空间

---

## 离散扩散语言模型

### 14. D3PM - Diffusion Models for Discrete Data (2021)

#### Meta
- **Title**: Structured Denoising Diffusion Models in Discrete State-Spaces
- **Link**: [arXiv:2107.03006](https://arxiv.org/abs/2107.03006)
- **Venue**: NeurIPS 2021
- **Date**: 2021-07
- **Tags**: [Discrete Diffusion, D3PM, Text Generation, Categorical Data]
- **Authors**: Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg
- **TL;DR**: 将扩散模型扩展到离散状态空间，定义离散数据的"去噪"过程和多种转移矩阵设计。

#### Problem & Contribution
- **解决的问题**: 标准扩散模型针对连续数据设计，无法直接应用于离散数据（如文本）
- **核心想法/方法一句话**: 定义离散状态空间上的扩散过程，使用分类分布替代高斯分布
- **主要贡献**:
  1. 将扩散模型扩展到离散状态空间
  2. 提出多种转移矩阵设计（吸收态、均匀、离散余弦等）
  3. 在文本生成和图像标注上验证有效性

#### Method
- **方法结构/流程**: 定义离散状态空间上的前向转移和后向去噪过程
- **关键设计**: 
  - 分类扩散: 每一步是类别之间的转移
  - 转移矩阵: 定义 $[\bar{Q}_t]_{ij} = q(x_t = j | x_0 = i)$
  - 后验: $q(x_{t-1} | x_t, x_0) \propto q(x_t | x_{t-1}) q(x_{t-1} | x_0)$
- **数学公式**:
  - 前向转移: $$q(x_t | x_{t-1}) = \text{Categorical}(x_t; p = x_{t-1} Q_t)$$
  - 累积转移: $$q(x_t | x_0) = \text{Categorical}(x_t; p = x_0 \bar{Q}_t)$$
  - 后验采样: $$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) q(x_{t-1} | x_0)}{q(x_t | x_0)}$$
  - 训练目标: $$\mathcal{L} = \mathbb{E}_{t, x_0, x_t}\left[ D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t)) \right]$$

#### Evidence
- **Benchmark / setting**: Text8, COCO Captions, CIFAR-10 (tokenized)
- **对比对象**: GPT-2, BERT
- **关键结果**: 
  - 首次展示离散扩散可以生成有意义的文本
  - 在可控生成任务上有潜力

#### Takeaways
- **核心洞察**: 扩散框架可以推广到离散空间，但需要新的数学工具
- **影响与意义**: 离散扩散领域的开创性工作
- **局限性**: 性能不如自回归模型，采样效率低

---

### 15. SEDD - Score Entropy Discrete Diffusion (2023)

#### Meta
- **Title**: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
- **Link**: [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)
- **Venue**: ICML 2024 (Oral)
- **Date**: 2023-10
- **Tags**: [Discrete Diffusion, Score Entropy, Language Models, Score Matching]
- **Authors**: Aaron Lou, Chenlin Meng, Stefano Ermon
- **TL;DR**: 提出 Score Entropy 损失，将分数匹配自然扩展到离散空间，在语言建模上与 GPT-2 竞争。

#### Problem & Contribution
- **解决的问题**: 标准扩散依赖分数匹配理论，离散空间推广未取得同等经验收益
- **核心想法/方法一句话**: 提出 Score Entropy 损失，将连续空间的分数匹配自然扩展到离散空间
- **主要贡献**:
  1. 提出 Score Entropy 损失
  2. 在语言建模上显著优于现有扩散范式
  3. 与 GPT-2 规模相当的模型性能竞争

#### Method
- **方法结构/流程**: 定义离散空间的"分数"为对数概率比，使用 Score Entropy 损失训练
- **关键设计**: 
  - 离散分数: $s_\theta(x, t)_i = \log \frac{p_t(x)}{p_t(x^{\backslash i})}$
  - Score Entropy 损失
  - 支持灵活的条件生成
- **数学公式**:
  - 离散分数定义: $$s_\theta(x, t) = \log p_\theta(x) - \log p_\theta(x^{\backslash i})$$
  - Score Entropy 损失: $$\mathcal{L}_{SE} = \mathbb{E}_{t, x_0, x_t}\left[\sum_{i=1}^{d} \text{SE}(s_\theta(x_t, t)_i, s^*(x_t, t)_i)\right]$$
  - 其中 $\text{SE}(s, s^*) = \exp(s^*) - s \cdot \exp(s^*) + s$
  - 训练目标可简化为: $$\mathcal{L} = \mathbb{E}_{t, x_0, x_t}\left[\sum_{i} \text{score\_entropy}(p_\theta(x_{t-1}^{(i)}|x_t), p_{target}(x_{t-1}^{(i)}|x_t, x_0))\right]$$

#### Evidence
- **Benchmark / setting**: OpenWebText, WikiText-103
- **对比对象**: GPT-2, ARDM, D3PM, Diffusion-LM
- **关键结果**: 
  - 困惑度相比现有扩散方法降低 25-75%
  - 与 GPT-2 规模相当的模型性能竞争
  - 无需温度缩放即可生成忠实文本
  - 少 32 倍网络评估获得相似质量
  - 支持可控填充，匹配核采样质量

#### Takeaways
- **核心洞察**: Score Entropy 是连接连续和离散扩散的桥梁
- **影响与意义**: 证明扩散模型在语言建模上的潜力，离散扩散领域的重要里程碑
- **局限性**: 采样效率仍低于自回归模型

---

### 16. MDLM - Masked Diffusion Language Models (2024)

#### Meta
- **Title**: Masked Diffusion Language Models
- **Link**: [arXiv:2406.07516](https://arxiv.org/abs/2406.07516)
- **Venue**: arXiv 2024
- **Date**: 2024-06
- **Tags**: [Masked Diffusion, Language Models, Discrete Diffusion, BERT]
- **Authors**: Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, Volodymyr Kuleshov
- **TL;DR**: 将掩码语言建模与连续扩散模型统一，提出简化训练目标，在零样本生成任务上取得竞争性能。

#### Problem & Contribution
- **解决的问题**: 离散扩散训练复杂，难以与现有语言模型预训练流程结合
- **核心想法/方法一句话**: 将掩码语言建模 (MLM) 视为一种特殊形式的离散扩散，统一训练框架
- **主要贡献**:
  1. 将掩码语言建模与连续扩散模型统一
  2. 提出简化的训练目标
  3. 与标准语言模型预训练兼容

#### Method
- **方法结构/流程**: 将 MLM 解释为离散扩散的一种形式，使用连续扩散的技术改进
- **关键设计**: 
  - 统一视角: BERT 可以看作特殊的扩散模型
  - 简化目标: 去除复杂的转移矩阵计算
  - 连续时间视角
- **数学公式**:
  - 掩码作为扩散: $$q(x_t | x_0) = \begin{cases} x_0 & \text{with prob } \alpha_t \\ \text{[MASK]} & \text{with prob } 1 - \alpha_t \end{cases}$$
  - 连续时间目标: $$\mathcal{L}_{MDLM} = \mathbb{E}_{t, x_0, x_t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$
  - 其中 $x_t$ 包含原始 token 和 [MASK] token 的混合
  - 训练时采样时间 $t \sim \text{Uniform}[0, 1]$

#### Evidence
- **Benchmark / setting**: 零样本生成任务，文本填空
- **对比对象**: GPT-2, BERT, D3PM, SEDD
- **关键结果**: 
  - 零样本生成任务上竞争性能
  - 简化训练目标提高训练效率
  - 与现有预训练流程兼容

#### Takeaways
- **核心洞察**: MLM 和扩散模型可以统一在同一框架下
- **影响与意义**: 为结合 BERT 和扩散模型的优势提供路径
- **局限性**: 大规模语言任务上仍有待验证

---

## Kaiming He 近期工作

### 17. MAR - Autoregressive Image Generation without Vector Quantization (2024)

#### Meta
- **Title**: Autoregressive Image Generation without Vector Quantization
- **Link**: [arXiv:2406.11838](https://arxiv.org/abs/2406.11838)
- **Venue**: NeurIPS 2024 (Spotlight)
- **Date**: 2024-06
- **Tags**: [Autoregressive, Image Generation, Diffusion Loss, Continuous Tokens, VQ-free]
- **Authors**: Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, Kaiming He
- **TL;DR**: 挑战自回归图像生成必须配合向量量化的传统认知，在连续值空间使用扩散损失进行自回归建模。

#### Problem & Contribution
- **解决的问题**: 自回归图像生成被认为必须依赖向量量化 (VQ)，限制表示能力
- **核心想法/方法一句话**: 在连续值空间进行自回归建模，使用扩散损失建模每个 token 的分布
- **主要贡献**:
  1. 证明离散值空间有助于表示分类分布，但不是自回归建模的必要条件
  2. 提出使用扩散过程建模每 token 的概率分布
  3. 定义 Diffusion Loss 替代分类交叉熵损失

#### Method
- **方法结构/流程**: 使用自回归模型预测下一个 token 的连续值表示，通过扩散损失训练
- **关键设计**: 
  - 移除向量量化 (VQ)
  - 每个 token 的概率分布通过扩散过程建模
  - 支持标准自回归和掩码自回归 (MAR) 变体
- **数学公式**:
  - 标准自回归: $$p(x) = \prod_{i=1}^{n} p(x_i | x_{<i})$$
  - 连续值建模: $$p(x_i | x_{<i}) = \int p(x_i | z_i) p(z_i | x_{<i}) dz_i$$
  - Diffusion Loss: $$\mathcal{L}_{diff} = \mathbb{E}_{t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_i, t, x_{<i})\|^2\right]$$
  - 目标: $$\min_\theta \sum_{i=1}^{n} \mathcal{L}_{diff}(x_i, x_{<i}; \theta)$$
  - 掩码自回归变体: $$p(x_{masked} | x_{observed}) = \prod_{j \in masked} p(x_j | x_{observed}, x_{<j \text{ in masked}})$$

#### Evidence
- **Benchmark / setting**: ImageNet 256×256, ImageNet 512×512
- **对比对象**: VQ-GAN, DALL-E, Parti, Muse
- **关键结果**: 
  - ImageNet 256×256: 与量化自回归模型相当或更好的生成质量
  - 无需离散值 tokenizer
  - 享受序列建模的速度优势

#### Takeaways
- **核心洞察**: 向量量化是自回归图像生成的便利工具而非必要条件
- **影响与意义**: 为图像生成提供新的自回归范式，可能促进语言和视觉统一建模
- **局限性**: 需要进一步验证在大规模应用中的效果

---

## 效率优化与压缩

### 18. DC-AE - Deep Compression Autoencoder (2024)

#### Meta
- **Title**: Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
- **Link**: [arXiv:2410.10733](https://arxiv.org/abs/2410.10733)
- **Venue**: ICLR 2025
- **Date**: 2024-10
- **Tags**: [Autoencoder, Compression, Efficient Diffusion, High-Resolution, Residual Encoding]
- **Authors**: Han Cai, Muyang Li, Ju Hu, Yawen Cui, Chuang Gan, Song Han
- **TL;DR**: 提出深度压缩自编码器 (DC-AE) 新家族，实现最高 128x 空间压缩比，解决高空间压缩比下重建精度不足的问题。

#### Problem & Contribution
- **解决的问题**: 高空间压缩比（如 64x）下重建精度不足，限制扩散模型效率
- **核心想法/方法一句话**: 通过残差自编码和解耦高分辨率适应实现极高压缩比同时保持质量
- **主要贡献**:
  1. 提出 DC-AE 新家族，实现 128x 空间压缩比
  2. 残差自编码缓解高压缩优化困难
  3. 解耦高分辨率适应减轻泛化惩罚

#### Method
- **方法结构/流程**: 基于空间到通道变换的特征学习残差，三阶段解耦训练
- **关键设计**: 
  - 残差自编码: 学习残差而非完整表示
  - 解耦训练: 分阶段适应高分辨率
  - 空间到通道变换
- **数学公式**:
  - 残差编码: $$z = \mathcal{E}_{base}(x) + \mathcal{E}_{res}(\text{STC}(x - \mathcal{D}_{base}(\mathcal{E}_{base}(x))))$$
  - 空间到通道变换: $$\text{STC}(h)_{c \cdot k^2 + i} = h_{c, \lfloor i/k \rfloor, i \bmod k}$$
  - 三阶段训练:
    - 阶段 1: 低分辨率预训练 $\min_{\mathcal{E}, \mathcal{D}} \|x - \mathcal{D}(\mathcal{E}(x))\|^2 + \lambda \mathcal{L}_{perceptual}$
    - 阶段 2: 高分辨率适应
    - 阶段 3: 蒸馏和微调
  - 压缩表示: $$z \in \mathbb{R}^{C \times H/r \times W/r}$$, $r$ 为压缩比

#### Evidence
- **Benchmark / setting**: ImageNet 256×256, ImageNet 512×512, SD 生成任务
- **对比对象**: SD-VAE-f8, SDXL-VAE, KL-f8
- **关键结果**: 
  - ImageNet 512×512: 19.1× 推理速度提升，17.9× 训练速度提升
  - 相比 SD-VAE-f8 取得更好的 FID
  - 支持 64x、128x 压缩比同时保持质量

#### Takeaways
- **核心洞察**: 残差学习和解耦训练是超高压缩比的关键
- **影响与意义**: 为高效图像生成提供关键基础设施
- **局限性**: 压缩比过高时细节损失仍不可避免

---

## 研究趋势与洞察

### 趋势 1: 从扩散到流 - 生成模型的数学统一

**演进路径**:  
DDPM (2020) → Score SDE (2021) → Flow Matching (2022) → Rectified Flow (2023)

**核心洞察**:
- 扩散模型可以统一在 SDE/ODE 框架下: $dx = f(x,t)dt + g(t)dw$
- Flow Matching 提供了更简洁的数学表述: $$\mathcal{L}_{CFM} = \mathbb{E}_{t, x_1, x}\left[\|v_t(x) - u_t(x|x_1)\|^2\right]$$
- Rectified Flow 解决了采样效率问题: $\phi_t(x_0) \approx (1-t)x_0 + t x_1$

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
- 离散空间需要新的数学工具（Score Entropy）: $$\mathcal{L}_{SE} = \mathbb{E}\left[\sum_{i} \text{score\_entropy}(s_\theta, s^*)\right]$$

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
MAR 尝试结合两者的优势——自回归的结构 + 连续空间的扩散建模: $$\mathcal{L}_{diff} = \mathbb{E}_{t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_i, t, x_{<i})\|^2\right]$$
这可能是图像-语言统一建模的重要一步。

---

### 趋势 4: 压缩与效率 - 实用化的关键

**核心方向**:
- **蒸馏**: 将多步扩散模型压缩为少步或单步模型
- **量化**: 降低模型精度以加速推理
- **压缩**: 如 DC-AE 的高倍率空间压缩: $z = \mathcal{E}(x)$ with 64x-128x ratio
- **架构优化**: DiT (Diffusion Transformer)、线性注意力等

**代表性技术**:
- Progressive Distillation
- Consistency Models
- LCM (Latent Consistency Models)
- DC-AE: 19.1× 推理加速

---

### 趋势 5: 多模态统一生成模型

**愿景**: 单一模型统一处理文本、图像、音频、视频等多种模态

**技术路线**:
- 将各种模态 token 化（VQ-VAE, VQ-GAN）
- 统一的自回归或扩散框架
- 跨模态的联合建模: $$p(\text{text}, \text{image}) = p(\text{text}) \cdot p(\text{image} | \text{text})$$

**代表性工作**:
- GPT-4V (视觉-语言)
- Gemini (多模态)
- Sora (视频生成)
- MAR (图像自回归)

---

## 关键洞察总结

1. **范式演进**: 生成模型经历了 GAN → VAE/Flow → Diffusion → Flow Matching 的演进，每一代都在解决前一代的核心问题，同时引入新的挑战。

2. **数学统一**: Flow Matching 和 Score-based 框架提供了理解各种生成模型的统一视角。核心方程:
   - SDE: $dx = f(x,t)dt + g(t)dw$
   - 概率流 ODE: $dx = [f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)]dt$
   - Flow Matching: $\mathcal{L}_{CFM} = \mathbb{E}\left[\|v_t(x) - u_t(x|x_1)\|^2\right]$

3. **效率革命**: 采样效率是扩散模型实用化的关键瓶颈，Rectified Flow、蒸馏、一致性模型等方向正在取得突破。

4. **语言模型的新可能**: 离散扩散为语言生成提供了不同于自回归的范式，在可控性和编辑任务上有独特优势，但效率仍是挑战。关键方程:
   - Score Entropy: $\mathcal{L}_{SE} = \mathbb{E}\left[\sum_i \text{score\_entropy}(s_\theta, s^*)\right]$

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
- LDM/Stable Diffusion: [arXXiv:2112.10752](https://arxiv.org/abs/2112.10752)

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
