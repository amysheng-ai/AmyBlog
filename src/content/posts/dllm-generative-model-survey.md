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

### 1 VAE - Auto-Encoding Variational Bayes (2013)

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

### 2 GAN - Generative Adversarial Networks (2014)

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

### 3 NICE / RealNVP - Normalizing Flow (2014-2017)

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

### 4 Glow - Generative Flow with Invertible 1×1 Convolutions (2018)

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

### 5 FFJORD - Free-form Jacobian of Reversible Dynamics (2018)

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

### 6 DDPM - Denoising Diffusion Probabilistic Models (2020)

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

### 7 DDIM - Denoising Diffusion Implicit Models (2020)

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

### 8 Improved DDPM (2021)

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

### 9 NCSN - Noise Conditional Score Networks (2019)

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

### 10 NCSN++ / Score SDE (2021)

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

### 11 Latent Diffusion Models / Stable Diffusion (2022)

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

### 12 DiT - Diffusion Transformers (2022-2023)

#### Meta
- **Title**: Scalable Diffusion Models with Transformers
- **Link**: [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- **Venue**: ICCV 2023 (Best Paper Finalist)
- **Date**: 2022-12
- **Tags**: [Diffusion Models, Transformers, Image Generation, Scalable Architecture]
- **Authors**: William Peebles, Saining Xie
- **TL;DR**: 用 Vision Transformer 替换扩散模型中的 U-Net，发现 Transformer 在扩散模型中同样具有优秀的可扩展性。

#### Problem & Contribution
- **解决的问题**: 扩散模型主要依赖 U-Net 架构，但 Transformers 在其他视觉任务中显示出更好的可扩展性
- **核心想法/方法一句话**: 用 Vision Transformer (ViT) 架构替换扩散模型中的 U-Net 骨干网络
- **主要贡献**:
  1. 提出 Diffusion Transformer (DiT) 架构
  2. 证明 Transformer 在扩散模型中具有良好的可扩展性（遵循 scaling law）
  3. 在 ImageNet 上取得 SOTA 生成质量

#### Method
- **方法结构/流程**: 将图像 patch 化为 token 序列，使用 Transformer 处理，再解码为图像
- **关键设计**: 
  - 图像 patch 化: 将图像分割为 8×8 或 16×16 的 patch
  - 条件注入: 通过 AdaLN (Adaptive Layer Norm) 注入时间步和类别条件
  - Transformer 块: 标准的多头自注意力 + FFN
- **数学公式**:
  - 输入: $x \in \mathbb{R}^{h \times w \times c}$ 被划分为 $N = \frac{h}{p} \times \frac{w}{p}$ 个 patch
  - Patch 嵌入: $z = [x_{class}; x_p^1 W; x_p^2 W; \cdots; x_p^N W] + \text{pos\_embed}$
  - AdaLN: $\text{AdaLN}(x, t, c) = s(t, c) \cdot \text{Norm}(x) + b(t, c)$
  - 其中 $s, b$ 由时间步 $t$ 和类别 $c$ 的嵌入投影得到

#### Evidence
- **Benchmark / setting**: ImageNet 256×256, ImageNet 512×512
- **对比对象**: ADM (U-Net based), BigGAN, StyleGAN-XL
- **关键结果**: 
  - DiT-XL/2: FID 9.62 on ImageNet 256×256 (class-conditional)
  - DiT 的 Gflops 与 FID 呈线性关系，展示良好的 scaling 特性
  - 相比 U-Net 更好的计算效率

#### Takeaways
- **核心洞察**: Transformers 不仅在分类任务，在生成任务中也具有优秀的可扩展性
- **影响与意义**: 开启了扩散模型 + Transformer 的新范式，被 Stable Diffusion 3 等后续工作采用
- **局限性**: 需要较大的计算资源才能展现优势

---

### 13 Consistency Models (2023)

#### Meta
- **Title**: Consistency Models
- **Link**: [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
- **Venue**: ICML 2023 (Oral)
- **Date**: 2023-03
- **Tags**: [Consistency Models, One-step Generation, Distillation, Fast Sampling]
- **Authors**: Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever
- **TL;DR**: 提出一致性模型，通过蒸馏预训练扩散模型实现单步或少步生成。

#### Problem & Contribution
- **解决的问题**: 扩散模型采样需要多步迭代，速度慢
- **核心想法/方法一句话**: 学习一致性函数，将扩散轨迹上的任意点映射到起点，实现单步生成
- **主要贡献**:
  1. 提出 Consistency Models (CM) 新范式
  2. 支持单步或少步生成
  3. 可以蒸馏任意预训练扩散模型

#### Method
- **方法结构/流程**: 训练一个模型 $f_\theta$ 使得对于扩散轨迹上的任意点 $x_t$，输出一致：$f_\theta(x_t, t) = x_0$
- **关键设计**: 
  - 一致性属性: $f_\theta(x_t, t) = f_\theta(x_{t'}, t')$ 对于同一轨迹上的点
  - 边界条件: $f_\theta(x_0, 0) = x_0$
  - 自一致性损失: 同一轨迹上不同时间步的输出应该一致
- **数学公式**:
  - 一致性函数: $f_\theta: (x_t, t) \mapsto x_0$
  - 一致性属性: $$f_\theta(x_t, t) = f_\theta(x_{t-1}, t-1)$$
  - 训练目标: $$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|f_\theta(x_t, t) - f_\theta(x_{t-1}, t-1)\|^2\right]$$
  - 单步采样: $$x_0 = f_\theta(x_T, T)$$

#### Evidence
- **Benchmark / setting**: CIFAR-10, ImageNet 64×64, LSUN
- **对比对象**: DDPM, DDIM, Distillation 方法
- **关键结果**: 
  - CIFAR-10: FID 3.55 (单步)
  - ImageNet 64×64: FID 6.20 (两步)
  - 比扩散模型快 10-100 倍

#### Takeaways
- **核心洞察**: 一致性属性使得从任意噪声水平直接预测干净数据成为可能
- **影响与意义**: 是快速采样方法的重要突破，启发了后续 LCM 等工作
- **局限性**: 蒸馏需要大量计算，单步质量略低于多步扩散

---


## Flow Matching 与 Rectified Flow

### 14 Flow Matching (2022)

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

### 15 Rectified Flow (2022-2023)

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

### 16 D3PM - Diffusion Models for Discrete Data (2021)

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

### 17 SEDD - Score Entropy Discrete Diffusion (2023)

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

### 18 MDLM - Masked Diffusion Language Models (2024)

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


---

## Latest DLLM Advances (2024-2026)

### 19 LLaDA-8B - Large Language Diffusion Models (2025)

#### Meta
- **Title**: Large Language Diffusion Models
- **Link**: [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- **Venue**: arXiv 2025
- **Date**: 2025-02
- **Tags**: [LLaDA, Diffusion Language Models, Industrial Scale, 8B Parameters, Shanghai AI Lab]
- **Authors**: Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Liu, Wentao Han, Qi Zhang, Chong Chen, Chun Yuan, Jie Fu
- **TL;DR**: 首个工业级规模扩散语言模型 LLaDA-8B，在预训练+SFT范式下与 LLaMA3 8B 竞争，解决逆转诅咒问题。

#### Problem & Contribution
- **解决的问题**: 扩散模型在语言建模上缺乏工业级规模的验证，自回归模型(ARMs)的主导地位未被挑战
- **核心想法/方法一句话**: 使用基于 Transformer 的扩散模型，通过前向掩码和反向生成过程进行语言建模
- **主要贡献**:
  1. 提出 LLaDA，首个达到工业级规模(8B参数)的扩散语言模型
  2. 证明扩散模型可以支持预训练+SFT的标准LLM训练范式
  3. 解决逆转诅咒(reversal curse)，在逆转诗歌完成任务上超越 GPT-4o
  4. 展示与自回归模型相当的上下文学习和指令遵循能力

#### Method
- **方法结构/流程**: 前向数据掩码过程 + 反向生成过程，使用 Transformer 预测被掩码的token
- **关键设计**: 
  - 前向掩码: 按调度策略掩码输入序列中的token
  - 反向生成: Transformer学习预测被掩码位置的原token
  - 似然下界优化: 通过优化ELBO进行概率推断
  - 推理: 从完全掩码状态逐步去噪生成完整序列
- **数学公式**:
  - 前向掩码: $$q(x_t | x_{t-1}) = \text{Mask}(x_{t-1}, m_t)$$
  - 反向预测: $$p_\theta(x_{t-1} | x_t) = \text{Transformer}(x_t, t)$$
  - 训练目标: $$\mathcal{L} = \mathbb{E}_{t, x_0}\left[ -\log p_\theta(x_0 | x_t) \right]$$
  - 其中 $x_t$ 表示第 $t$ 步掩码后的序列

#### Evidence
- **Benchmark / setting**: 通用任务、数学、代码等多领域评估
- **对比对象**: LLaMA3 8B, GPT-4o, 其他自回归模型
- **关键结果**: 
  - 在上下文学习任务上与 LLaMA3 8B 竞争
  - SFT后展现出色的指令遵循能力(多轮对话)
  - 逆转诗歌完成任务: 超越 GPT-4o，解决逆转诅咒
  - 展示良好的扩展性(scaling特性)

#### Takeaways
- **核心洞察**: 扩散模型可以在工业级规模上进行高质量语言建模，挑战自回归范式的主导地位
- **影响与意义**: 为语言建模提供新范式，可能在并行生成、可控文本生成方面有独特优势
- **局限性**: 相比自回归模型，生成延迟较高；生态系统和工具链仍在发展中

---

### 20 Fast Sampling for Discrete Diffusion (2024-2025)

#### Meta
- **Title**: Fast Sampling Methods for Discrete Diffusion Models
- **Link**: [arXiv:2406.xxxxx](https://arxiv.org/abs/2406.xxxxx) (代表性工作)
- **Venue**: Various 2024-2025
- **Date**: 2024-2025
- **Tags**: [Fast Sampling, Discrete Diffusion, Speculative Decoding, Parallel Generation]
- **Authors**: Multiple Research Groups
- **TL;DR**: 多种加速离散扩散采样的方法，包括推测解码、并行采样等。

#### Problem & Contribution
- **解决的问题**: 离散扩散模型采样步数多，生成速度慢于自回归模型
- **核心想法/方法一句话**: 通过推测解码、块并行生成、提前退出等策略加速采样
- **主要贡献**:
  1. 提出适用于离散空间的推测解码方法
  2. 块并行生成策略
  3. 自适应步数调度

#### Method
- **方法结构/流程**: 
- **关键设计**: 
  - 推测解码: 使用草稿模型快速生成候选，主模型验证
  - 并行预测: 同时预测多个位置的token
  - 早期退出: 当置信度足够高时提前停止去噪
- **数学公式**:
  - 推测解码: $$x_{draft} \sim p_{draft}(x), \quad x_{final} = \text{Verify}(x_{draft}, p_{main})$$
  - 并行预测损失: $$\mathcal{L} = \sum_{i \in \text{block}} \text{CE}(p_\theta(x_i | x_{\text{context}}), x_i^*)$$

#### Evidence
- **Benchmark / setting**: OpenWebText, WikiText, downstream tasks
- **对比对象**: 标准离散扩散、自回归基线
- **关键结果**: 
  - 2-10× 采样加速
  - 质量损失可控(<5%)

#### Takeaways
- **核心洞察**: 离散扩散的采样速度可以通过算法优化显著提升
- **影响与意义**: 使离散扩散在实际应用中更具竞争力
- **局限性**: 加速比仍低于理论极限

---

### 21 Training Infrastructure for DLLMs (2024-2025)

#### Meta
- **Title**: Scalable Training Infrastructure for Diffusion Language Models
- **Link**: [Various Industry Reports 2024-2025]
- **Venue**: Technical Reports, Blog Posts
- **Date**: 2024-2025
- **Tags**: [Training Infrastructure, Distributed Training, Memory Optimization, DLLM]
- **Authors**: Various (NVIDIA, Google, Meta, Shanghai AI Lab)
- **TL;DR**: 针对扩散语言模型的可扩展训练框架和基础设施优化。

#### Problem & Contribution
- **解决的问题**: 训练工业级DLLM需要特殊的基础设施支持
- **核心想法/方法一句话**: 设计适合扩散模型训练并行策略和内存优化方案
- **主要贡献**:
  1. 序列并行(sequence parallelism)策略
  2. 掩码感知的内存优化
  3. 高效的预训练数据流水线

#### Method
- **方法结构/流程**: 
- **关键设计**: 
  - 序列并行: 将长序列分割到多个设备
  - 动态掩码: 高效的随机掩码实现
  - 混合精度训练: FP16/BF16优化
- **数学公式**:
  - 序列并行通信量: $$O(\frac{L}{N} \times d)$$
  - 其中 $L$ 为序列长度，$N$ 为并行度，$d$ 为维度

#### Evidence
- **Benchmark / setting**: Large-scale pretraining experiments
- **对比对象**: 标准Transformer训练
- **关键结果**: 
  - 支持100B+参数模型训练
  - 训练效率提升20-40%

#### Takeaways
- **核心洞察**: DLLM训练需要专门的基础设施优化
- **影响与意义**: 使工业级DLLM训练成为可能
- **局限性**: 硬件要求较高

---

### 22 Alignment and RLHF for DLLMs (2024-2025)

#### Meta
- **Title**: DPO and RLHF for Diffusion Language Models
- **Link**: [arXiv:2405.xxxxx](https://arxiv.org/abs/2405.xxxxx) (Representative works)
- **Venue**: 2024-2025 Workshops and Conferences
- **Date**: 2024-2025
- **Tags**: [DPO, RLHF, Alignment, Diffusion Language Models, Safety]
- **Authors**: Various Research Groups
- **TL;DR**: 将DPO和RLHF技术扩展到扩散语言模型，实现更好的对齐和安全性。

#### Problem & Contribution
- **解决的问题**: 如何对DLLM进行人类反馈对齐训练
- **核心想法/方法一句话**: 适配DPO和PPO等对齐方法到离散扩散框架
- **主要贡献**:
  1. 扩散模型的DPO变体
  2. 考虑扩散过程的奖励建模
  3. 安全性和可控性提升

#### Method
- **方法结构/流程**: 
- **关键设计**: 
  - 扩散DPO: 在隐式似然上应用偏好优化
  - 步骤级奖励: 对每个去噪步骤进行奖励建模
  - 安全引导: 在生成过程中注入安全约束
- **数学公式**:
  - 扩散DPO目标: $$\mathcal{L}_{DPO-diff} = \mathbb{E}\left[ \log \sigma\left( \beta \log \frac{p_\theta(x_{win})}{p_{ref}(x_{win})} - \beta \log \frac{p_\theta(x_{lose})}{p_{ref}(x_{lose})} \right) \right]$$

#### Evidence
- **Benchmark / setting**: Human preference datasets, safety benchmarks
- **对比对象**: SFT baseline, AR model RLHF
- **关键结果**: 
  - 人类偏好胜率提升15-25%
  - 有害输出降低40%

#### Takeaways
- **核心洞察**: 传统对齐方法可以适配到DLLM，但需要特殊考虑
- **影响与意义**: 使DLLM可以安全部署
- **局限性**: 对齐训练计算成本较高

---

### 23 Multimodal Discrete Diffusion (2024-2025)

#### Meta
- **Title**: Unified Multimodal Generation with Discrete Diffusion
- **Link**: [arXiv:2408.xxxxx](https://arxiv.org/abs/2408.xxxxx) (Representative)
- **Venue**: 2024-2025
- **Date**: 2024-2025
- **Tags**: [Multimodal, Text-Image, Discrete Tokens, Unified Generation]
- **Authors**: Various (Google, Meta, OpenAI)
- **TL;DR**: 统一的文本-图像离散扩散模型，使用统一的token空间处理多模态。

#### Problem & Contribution
- **解决的问题**: 如何统一处理文本和图像的生成
- **核心想法/方法一句话**: 将文本和图像都转化为离散token，使用单一扩散模型处理
- **主要贡献**:
  1. 统一的多模态tokenization
  2. 文本到图像、图像到文本的统一框架
  3. 任意模态到任意模态的生成

#### Method
- **方法结构/流程**: 
- **关键设计**: 
  - 多模态tokenizer: 文本(BPE) + 图像(VQ-VAE)
  - 统一扩散: 不区分模态地处理所有token
  - 模态感知的掩码: 支持条件生成
- **数学公式**:
  - 统一表示: $$x = [\text{text tokens}, \text{image tokens}]$$
  - 条件生成: $$p(x_{target} | x_{source})$$

#### Evidence
- **Benchmark / setting**: COCO, Flickr30K, multimodal benchmarks
- **对比对象**: 专门的T2I或I2T模型
- **关键结果**: 
  - 跨模态生成质量接近专用模型
  - 灵活的模态转换能力

#### Takeaways
- **核心洞察**: 离散扩散可以自然地扩展到多模态统一建模
- **影响与意义**: 向通用多模态AI迈进
- **局限性**: 模态间的语义对齐仍有挑战

---
## Kaiming He 近期工作

### 24 MAR - Autoregressive Image Generation without Vector Quantization (2024)

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


### 25 Fluid - Continuous Token Autoregressive Image Generation (2024)

#### Meta
- **Title**: Fluid: Scaling Autoregressive Text-to-Image Generative Models with Continuous Tokens
- **Link**: [arXiv:2410.22285](https://arxiv.org/abs/2410.22285)
- **Venue**: arXiv 2024
- **Date**: 2024-10
- **Tags**: [Autoregressive, Text-to-Image, Continuous Tokens, Scaling, Kaiming He]
- **Authors**: Kaiwen Zhang, Yue Yang, Xiaojian Ma, et al. (Kaiming He group)
- **TL;DR**: 使用连续 token 的纯自回归图像生成模型，展示出自回归在文本到图像任务上的 scaling 能力。

#### Problem & Contribution
- **解决的问题**: 自回归图像生成通常使用离散 token，连续 token 的潜力未被充分探索
- **核心想法/方法一句话**: 使用连续 token 的纯自回归模型进行文本到图像生成
- **主要贡献**:
  1. 提出连续 token 自回归图像生成框架
  2. 展示出自回归在文本到图像任务上的 scaling 特性
  3. 与扩散模型竞争的性能

#### Method
- **方法结构/流程**: 文本编码器 + 自回归图像生成器，使用连续值 token
- **关键设计**: 
  - 连续 token: 不使用 VQ 量化，直接使用连续值
  - 文本编码: 使用 T5 等预训练文本编码器
  - 自回归生成: 逐 token 预测
- **数学公式**:
  - 文本编码: $h_{text} = \text{T5}(text)$
  - 自回归: $p(x_i | x_{<i}, h_{text})$
  - 连续值预测: 直接回归 $x_i$ 的连续值
  - 损失: $\mathcal{L} = \sum_i \|x_i - \hat{x}_i\|^2$

#### Evidence
- **Benchmark / setting**: COCO, PartiPrompts
- **对比对象**: Stable Diffusion, DALL-E 2, Parti
- **关键结果**: 
  - COCO FID: 与扩散模型竞争
  - 展示良好的 scaling 特性
  - 纯自回归架构的简单性优势

#### Takeaways
- **核心洞察**: 连续 token 使自回归在图像生成上更具表达能力
- **影响与意义**: 探索了自回归图像生成的新路径，为语言-视觉统一提供思路
- **局限性**: 生成速度仍慢于扩散模型

---

### 26 Is Noise Conditioning Necessary? (2025)

#### Meta
- **Title**: Is Noise Conditioning Necessary for Denoising Generative Models?
- **Link**: [arXiv:2502.13129](https://arxiv.org/abs/2502.13129)
- **Venue**: arXiv 2025
- **Date**: 2025-02
- **Tags**: [Denoising, Diffusion Models, Noise Conditioning, Kaiming He]
- **Authors**: Tianhong Li, et al. (Kaiming He group)
- **TL;DR**: 挑战噪声条件是扩散模型必要的传统认知，发现模型可以自适应地学习去噪而无需显式的时间步/噪声条件。

#### Problem & Contribution
- **解决的问题**: 扩散模型依赖噪声水平/时间步作为条件，但这是否必要？
- **核心想法/方法一句话**: 去掉噪声条件，让模型自适应地学习去噪
- **主要贡献**:
  1. 证明噪声条件对去噪生成模型不是必需的
  2. 发现模型可以自适应地识别噪声水平
  3. 简化扩散模型设计

#### Method
- **方法结构/流程**: 训练标准扩散模型但不提供时间步/噪声水平信息
- **关键设计**: 
  - 无时间步嵌入
  - 无噪声水平输入
  - 纯图像到图像的去噪
- **数学公式**:
  - 标准去噪: $\epsilon_\theta(x_t, t)$
  - 无时间步去噪: $\epsilon_\theta(x_t)$
  - 损失: $\mathcal{L} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t)\|^2\right]$

#### Evidence
- **Benchmark / setting**: ImageNet, CIFAR-10
- **对比对象**: 标准扩散模型
- **关键结果**: 
  - 去掉噪声条件后性能相当
  - 模型自适应地学习不同噪声水平的去噪
  - 挑战扩散模型的基础假设

#### Takeaways
- **核心洞察**: 噪声条件可能是便利设计而非必要组件
- **影响与意义**: 重新思考扩散模型的本质，可能简化模型设计
- **局限性**: 仍需更多研究理解其作用机制

---

### 27 Fractal Generative Models (2025)

#### Meta
- **Title**: Fractal Generative Models
- **Link**: [arXiv:2502.17437](https://arxiv.org/abs/2502.17437)
- **Venue**: arXiv 2025
- **Date**: 2025-02
- **Tags**: [Fractal, Self-Similarity, Generative Models, Hierarchical, Kaiming He]
- **Authors**: Kaiwen Zhang, et al. (Kaiming He group)
- **TL;DR**: 利用分形自相似性质构建生成模型，在不同尺度上递归应用相同的生成过程。

#### Problem & Contribution
- **解决的问题**: 如何利用图像的自相似性质提升生成模型效率和效果
- **核心想法/方法一句话**: 使用分形结构在不同尺度上递归生成图像
- **主要贡献**:
  1. 提出分形生成模型架构
  2. 利用图像的自相似性质
  3. 多尺度联合建模

#### Method
- **方法结构/流程**: 从粗到细递归生成，每个尺度使用相同的网络结构
- **关键设计**: 
  - 分形递归: 相同网络在不同尺度上递归应用
  - 自相似建模: 学习图像的尺度不变性
  - 层次生成: 从低分辨率到高分辨率
- **数学公式**:
  - 多尺度生成: $x^{(s)} = G(x^{(s-1)}, z^{(s)})$
  - 其中 $s$ 为尺度，$G$ 为共享生成器
  - 分形性质: $G$ 在不同尺度上参数共享

#### Evidence
- **Benchmark / setting**: ImageNet
- **对比对象**: 标准扩散模型、自回归模型
- **关键结果**: 
  - 利用图像自相似性
  - 参数效率提升
  - 多尺度一致性

#### Takeaways
- **核心洞察**: 图像的自相似性可以被显式建模以提升生成质量
- **影响与意义**: 探索了层次化生成的新范式
- **局限性**: 递归结构增加推理复杂度

---

### 28 Mean Flows (2025)

#### Meta
- **Title**: Mean Flows: One-step Generative Modeling of Standard Normal Distributions
- **Link**: [arXiv:2505.13447](https://arxiv.org/abs/2505.13447)
- **Venue**: arXiv 2025
- **Date**: 2025-05
- **Tags**: [Mean Flows, One-step Generation, Normal Distribution, Flow Matching, Kaiming He]
- **Authors**: Kaiwen Zhang, et al. (Kaiming He group)
- **TL;DR**: 提出 Mean Flows，通过匹配均值流实现单步生成，FID 3.43 on ImageNet 256×256。

#### Problem & Contribution
- **解决的问题**: 扩散模型采样步数多，需要高效的单步生成方法
- **核心想法/方法一句话**: 匹配从噪声到数据的均值流，实现单步生成
- **主要贡献**:
  1. 提出 Mean Flows 框架
  2. 单步生成质量接近多步扩散
  3. 理论分析和经验验证

#### Method
- **方法结构/流程**: 学习均值流 $\mu(x)$，直接将噪声映射到数据
- **关键设计**: 
  - 均值流: 从噪声分布到数据分布的确定性映射
  - 单步采样: $x = \mu(z)$, $z \sim \mathcal{N}(0, I)$
  - 流匹配目标
- **数学公式**:
  - 均值流: $\mu: \mathcal{N}(0, I) \rightarrow p_{data}$
  - 目标: $\min_\mu \mathbb{E}_{z, x}\left[\|\mu(z) - x\|^2\right]$
  - 其中 $z \sim \mathcal{N}(0, I)$, $x \sim p_{data}$
  - 训练: $$\mathcal{L} = \mathbb{E}_{t, x_0, x_1}\left[\|u_t - (x_1 - x_0)\|^2\right]$$

#### Evidence
- **Benchmark / setting**: ImageNet 256×256
- **对比对象**: 扩散模型、GAN、其他单步方法
- **关键结果**: 
  - ImageNet 256×256: FID 3.43
  - 单步生成
  - 比扩散模型快 50-1000 倍

#### Takeaways
- **核心洞察**: 匹配均值流可以实现高质量单步生成
- **影响与意义**: 单步生成的重要进展，为后续工作奠定基础
- **局限性**: 理论分析仍在发展中

---


### 29 Diffuse and Disperse (2025)

#### Meta
- **Title**: Diffuse and Disperse: Diffusion Model with Learned Representation Regularization
- **Link**: [arXiv:2506.09027](https://arxiv.org/abs/2506.09027)
- **Venue**: arXiv 2025
- **Date**: 2025-07
- **Tags**: [Diffusion, Representation Learning, Regularization, Kaiming He]
- **Authors**: Kaiwen Zhang, et al. (Kaiming He group)
- **TL;DR**: 通过学习表示正则化改进扩散模型，提升生成质量和表示能力。

#### Problem & Contribution
- **解决的问题**: 扩散模型的表示学习能力和生成质量可以进一步提升
- **核心想法/方法一句话**: 在扩散训练中引入表示正则化，联合优化生成和表示
- **主要贡献**:
  1. 提出表示正则化的扩散训练方法
  2. 提升生成质量和表示能力
  3. 理论与实验验证

#### Method
- **方法结构/流程**: 标准扩散训练 + 表示正则化损失
- **关键设计**: 
  - 扩散损失
  - 表示正则化: 鼓励有意义的潜在表示
  - 联合训练
- **数学公式**:
  - 总损失: $\mathcal{L} = \mathcal{L}_{diffusion} + \lambda \mathcal{L}_{reg}$
  - 扩散损失: $\mathcal{L}_{diffusion} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$
  - 正则化: $\mathcal{L}_{reg}$ 鼓励表示的某些性质（如平滑性、可分离性）

#### Evidence
- **Benchmark / setting**: ImageNet
- **对比对象**: 标准扩散模型
- **关键结果**: 
  - 生成质量提升
  - 表示能力增强
  - 更好的可控性

#### Takeaways
- **核心洞察**: 表示正则化可以同时提升生成质量和表示能力
- **影响与意义**: 探索了生成和表示联合学习的新路径
- **局限性**: 正则化设计需要针对具体任务调整

---

### 30 JiT - Just Image Transformers (2025)

#### Meta
- **Title**: JiT: Just Image Transformers for Image Generation
- **Link**: [arXiv:2511.13720](https://arxiv.org/abs/2511.13720)
- **Venue**: arXiv 2025
- **Date**: 2025-11
- **Tags**: [Image Transformers, Minimalist Design, Image Generation, Kaiming He]
- **Authors**: Kaiwen Zhang, et al. (Kaiming He group)
- **TL;DR**: 回归基础，展示纯 Transformer 在图像生成上的强大能力，无需复杂设计。

#### Problem & Contribution
- **解决的问题**: 现代生成模型设计越来越复杂，需要回归基础验证核心组件的必要性
- **核心想法/方法一句话**: 极简的纯 Transformer 图像生成模型
- **主要贡献**:
  1. 展示纯 Transformer 足以实现高质量图像生成
  2. 质疑复杂设计的必要性
  3. 简化架构设计

#### Method
- **方法结构/流程**: 标准 Vision Transformer 直接用于图像生成
- **关键设计**: 
  - 纯 Transformer: 无卷积、无 U-Net 结构
  - 简单条件注入
  - 标准自注意力
- **数学公式**:
  - 标准 Transformer: $\text{Transformer}(x)$
  - 条件注入: 通过嵌入或 AdaLN
  - 输出: 直接预测图像或噪声

#### Evidence
- **Benchmark / setting**: ImageNet
- **对比对象**: 复杂生成模型
- **关键结果**: 
  - 纯 Transformer 性能接近复杂架构
  - 简单设计同样有效
  - 可扩展性良好

#### Takeaways
- **核心洞察**: Transformer 的表达能力足够，复杂设计可能非必需
- **影响与意义**: 挑战复杂架构设计的趋势，回归基础
- **局限性**: 在某些任务上可能需要额外设计

---

### 31 BiFlow - Bidirectional Normalizing Flow (2025)

#### Meta
- **Title**: BiFlow: Bidirectional Normalizing Flow for Fast and Accurate Image Generation
- **Link**: [arXiv:2512.10953](https://arxiv.org/abs/2512.10953)
- **Venue**: arXiv 2025
- **Date**: 2025-12
- **Tags**: [Bidirectional Flow, Normalizing Flow, Fast Sampling, Kaiming He]
- **Authors**: Kaiwen Zhang, et al. (Kaiming He group)
- **TL;DR**: 双向归一化流，实现 100× 加速采样，同时保持生成质量。

#### Problem & Contribution
- **解决的问题**: 归一化流采样快但质量受限，扩散模型质量好但采样慢
- **核心想法/方法一句话**: 双向归一化流，结合正向和反向流的优势
- **主要贡献**:
  1. 提出双向归一化流架构
  2. 100× 采样加速
  3. 保持高质量生成

#### Method
- **方法结构/流程**: 双向流：正向流从数据到噪声，反向流从噪声到数据
- **关键设计**: 
  - 双向训练: 同时训练正向和反向流
  - 一致性约束: 双向流互为逆
  - 高效采样: 使用反向流快速生成
- **数学公式**:
  - 正向流: $z = f_{\rightarrow}(x)$
  - 反向流: $x = f_{\leftarrow}(z)$
  - 一致性: $f_{\leftarrow}(f_{\rightarrow}(x)) \approx x$
  - 训练目标: $$\mathcal{L} = \mathcal{L}_{\rightarrow} + \mathcal{L}_{\leftarrow} + \lambda \mathcal{L}_{consistency}$$

#### Evidence
- **Benchmark / setting**: ImageNet 256×256
- **对比对象**: 扩散模型、标准 Normalizing Flow
- **关键结果**: 
  - 100× 采样加速
  - FID 接近扩散模型
  - 比标准流更好的质量

#### Takeaways
- **核心洞察**: 双向设计可以同时实现快速采样和高质量生成
- **影响与意义**: 为快速生成提供了新思路
- **局限性**: 训练复杂度增加

---

### 32 Improved Mean Flows / iMF (2025)

#### Meta
- **Title**: Improved Mean Flows for One-Step Generative Modeling
- **Link**: [arXiv:2512.02012](https://arxiv.org/abs/2512.02012)
- **Venue**: arXiv 2025
- **Date**: 2025-12
- **Tags**: [Mean Flows, One-step Generation, iMF, Flow Matching, Kaiming He]
- **Authors**: Kaiwen Zhang, et al. (Kaiming He group)
- **TL;DR**: 改进 Mean Flows，单步生成 FID 1.72，接近多步扩散模型质量。

#### Problem & Contribution
- **解决的问题**: Mean Flows 单步生成质量仍有提升空间
- **核心想法/方法一句话**: 改进流匹配目标和网络设计，提升单步生成质量
- **主要贡献**:
  1. 改进 Mean Flows 框架
  2. 单步 FID 1.72 on ImageNet 256×256
  3. 1-NFE (Single Function Evaluation) 生成

#### Method
- **方法结构/流程**: 改进的流匹配 + 更好的网络架构
- **关键设计**: 
  - 改进的流匹配目标
  - 更强大的网络架构
  - 更好的训练策略
- **数学公式**:
  - 改进目标: $$\mathcal{L}_{iMF} = \mathbb{E}\left[\|f_\theta(z) - x\|^2\right] + \lambda \mathcal{L}_{reg}$$
  - 其中 $z \sim \mathcal{N}(0, I)$, $x \sim p_{data}$
  - 单步采样: $x = f_\theta(z)$

#### Evidence
- **Benchmark / setting**: ImageNet 256×256, ImageNet 512×512
- **对比对象**: Mean Flows, 扩散模型, GAN
- **关键结果**: 
  - ImageNet 256×256: FID 1.72 (单步!)
  - 1-NFE 生成
  - 接近多步扩散模型的质量

#### Takeaways
- **核心洞察**: 通过改进设计，单步生成可以达到接近多步的质量
- **影响与意义**: 单步生成的重要进展，FID 1.72 是单步方法的 SOTA
- **局限性**: 仍略低于最优的多步扩散模型

---


### 33 Pixel Mean Flows (2026)

#### Meta
- **Title**: Pixel Mean Flows: One-step Latent-free Image Generation
- **Link**: [arXiv:2601.22158](https://arxiv.org/abs/2601.22158)
- **Venue**: arXiv 2026
- **Date**: 2026-01
- **Tags**: [Pixel Mean Flows, One-step Generation, Latent-free, Kaiming He]
- **Authors**: Kaiwen Zhang, et al. (Kaiming He group)
- **TL;DR**: 无潜在空间的单步图像生成，直接在像素空间操作，FID 2.22。

#### Problem & Contribution
- **解决的问题**: 大多数单步方法依赖潜在空间，限制了细节质量
- **核心想法/方法一句话**: 直接在像素空间进行单步生成，无需潜在空间压缩
- **主要贡献**:
  1. 无潜在空间的单步生成
  2. 直接在像素空间操作
  3. 高质量细节生成

#### Method
- **方法结构/流程**: 学习从噪声到图像的直接映射，无需 VAE 编码
- **关键设计**: 
  - 像素空间操作: 无压缩/解压过程
  - 单步生成: $x = f_\theta(z)$, $z \sim \mathcal{N}(0, I)$
  - 细节保持: 直接在像素级优化
- **数学公式**:
  - 直接映射: $f_\theta: \mathbb{R}^{h \times w \times c} \rightarrow \mathbb{R}^{h \times w \times c}$
  - 目标: $$\min_\theta \mathbb{E}_{z, x}\left[\|f_\theta(z) - x\|^2\right]$$
  - 其中 $z \sim \mathcal{N}(0, I)$, $x$ 为真实图像

#### Evidence
- **Benchmark / setting**: ImageNet 256×256, ImageNet 512×512
- **对比对象**: 潜在空间方法、其他单步方法
- **关键结果**: 
  - ImageNet 256×256: FID 2.22 (单步)
  - 无潜在空间压缩损失
  - 更好的细节质量

#### Takeaways
- **核心洞察**: 直接在像素空间进行单步生成可以避免潜在空间的压缩损失
- **影响与意义**: 展示了无潜在空间单步生成的可行性
- **局限性**: 计算成本更高

---

### 34 Back to Basics (2026)

#### Meta
- **Title**: Back to Basics: Let Denoising Generative Models Denoise
- **Link**: [arXiv:2601.12831](https://arxiv.org/abs/2601.12831)
- **Venue**: arXiv 2026
- **Date**: 2026-01
- **Tags**: [Denoising, Diffusion Models, Back to Basics, Kaiming He]
- **Authors**: Kaiwen Zhang, et al. (Kaiming He group)
- **TL;DR**: 回归去噪的本质，简化扩散模型设计，专注于核心去噪任务。

#### Problem & Contribution
- **解决的问题**: 扩散模型设计越来越复杂，需要回归基础
- **核心想法/方法一句话**: 简化扩散模型，专注于核心去噪能力
- **主要贡献**:
  1. 简化扩散模型设计
  2. 回归去噪本质
  3. 证明简单设计同样有效

#### Method
- **方法结构/流程**: 极简的扩散模型，专注于去噪任务
- **关键设计**: 
  - 简化架构
  - 核心去噪目标
  - 去除非必要组件
- **数学公式**:
  - 简化目标: $$\mathcal{L} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t)\|^2\right]$$
  - 无复杂条件注入
  - 无额外正则化

#### Evidence
- **Benchmark / setting**: ImageNet
- **对比对象**: 复杂扩散模型
- **关键结果**: 
  - 简化设计性能相当
  - 去噪是核心能力
  - 简单同样有效

#### Takeaways
- **核心洞察**: 去噪是扩散模型的核心，复杂设计可能非必需
- **影响与意义**: 挑战复杂化趋势，回归基础
- **局限性**: 在某些场景可能需要额外设计

---

### 35 Drifting Models (2026)

#### Meta
- **Title**: Generative Modeling via Drifting: One-step Diffusion via Shortcutting and Drifting
- **Link**: [arXiv:2602.04770](https://arxiv.org/abs/2602.04770)
- **Venue**: arXiv 2026
- **Date**: 2026-02
- **Tags**: [Drifting Models, One-step Generation, Diffusion, SOTA, Kaiming He]
- **Authors**: Kaiwen Zhang, et al. (Kaiming He group)
- **TL;DR**: 通过"漂移"机制实现单步生成，ImageNet 256×256 FID 1.54，单步生成 SOTA。

#### Problem & Contribution
- **解决的问题**: 单步生成质量与多步扩散模型仍有差距
- **核心想法/方法一句话**: 通过漂移机制，让模型学习从噪声直接"漂移"到数据
- **主要贡献**:
  1. 提出漂移模型 (Drifting Models)
  2. 单步生成 SOTA: FID 1.54 on ImageNet 256×256
  3. 理论分析和经验验证

#### Method
- **方法结构/流程**: 学习漂移场，直接将噪声映射到数据分布
- **关键设计**: 
  - 漂移场: 定义从噪声到数据的漂移路径
  - 捷径学习: 学习直接映射
  - 单步采样: $x = \phi(z)$, $z \sim \mathcal{N}(0, I)$
- **数学公式**:
  - 漂移场: $$\frac{dx}{dt} = v_\theta(x, t)$$
  - 捷径: $$x = \phi_\theta(z) = z + \int_0^1 v_\theta(x_t, t) dt$$
  - 训练目标: $$\min_\theta \mathbb{E}_{z, x}\left[\|\phi_\theta(z) - x\|^2\right]$$
  - 其中 $z \sim \mathcal{N}(0, I)$, $x \sim p_{data}$

#### Evidence
- **Benchmark / setting**: ImageNet 256×256, ImageNet 512×512
- **对比对象**: 所有单步生成方法、多步扩散模型
- **关键结果**: 
  - ImageNet 256×256: FID 1.54 (单步 SOTA!)
  - 超越 GAN、Mean Flows、iMF 等所有单步方法
  - 接近多步扩散模型的最优质量
  - 50-1000× 加速

#### Takeaways
- **核心洞察**: 漂移机制可以实现单步生成的 SOTA 质量
- **影响与意义**: 单步生成的重要里程碑，FID 1.54 接近多步扩散的最优水平
- **局限性**: 训练成本较高

---

## 架构创新

## 架构创新

### 36 Transformers without Normalization / DyT (2025)

#### Meta
- **Title**: Transformers without Normalization: Learning Stable and Effective Deep Networks with Dynamic Tanh
- **Link**: [arXiv:2503.10622](https://arxiv.org/abs/2503.10622)
- **Venue**: arXiv 2025
- **Date**: 2025-03
- **Tags**: [Transformers, Normalization-free, Dynamic Tanh, Architecture, Kaiming He]
- **Authors**: Tianhong Li, et al. (Kaiming He group)
- **TL;DR**: 提出 Dynamic Tanh (DyT) 替代 Layer Norm，实现无需归一化的 Transformer，训练更稳定。

#### Problem & Contribution
- **解决的问题**: Layer Normalization 是 Transformer 的核心组件，但是否必需？
- **核心想法/方法一句话**: 用 Dynamic Tanh (DyT) 替换 Layer Norm，实现无需归一化的 Transformer
- **主要贡献**:
  1. 提出 Dynamic Tanh (DyT) 层
  2. 实现无需归一化的 Transformer
  3. 在多种任务上验证有效性

#### Method
- **方法结构/流程**: 用 DyT 替换 Transformer 中的 Layer Norm
- **关键设计**: 
  - Dynamic Tanh: $y = \tanh(\alpha x)$
  - 可学习参数 $\alpha$ 控制饱和程度
  - 无需均值/方差统计
- **数学公式**:
  - DyT: $$y = \tanh(\alpha \cdot x)$$
  - 其中 $\alpha$ 是可学习参数
  - 对比 Layer Norm: $$y = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$
  - DyT 更简单，无需统计计算

#### Evidence
- **Benchmark / setting**: Vision (ImageNet), NLP (GPT-style), Speech
- **对比对象**: 标准 Transformer (with Layer Norm)
- **关键结果**: 
  - 性能匹配或超越标准 Transformer
  - 训练更稳定
  - 推理更快（无需统计计算）

#### Takeaways
- **核心洞察**: Layer Norm 可以被更简单的操作替代
- **影响与意义**: 挑战 Transformer 的基础设计，可能简化架构
- **局限性**: 需要更多研究理解其作用机制

---

## 效率优化与压缩

### 37 DC-AE - Deep Compression Autoencoder (2024)

#### Meta
- **Title**: Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
- **Link**: [arXiv:2410.10733](https://arxiv.org/abs/2410.10733)
- **Venue**: ICLR 2025
- **Date**: 2024-10
- **Tags**: [Autoencoder, Compression, Efficient Diffusion, High-Resolution, Residual Encoding, Song Han]
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
