---
title: "Daily AI Papers - Feb 16, 2026"
published: 2026-02-16
description: "Curated AI papers from HuggingFace Daily Papers and arXiv"
tags: [Daily-Papers, RLVR, Reasoning, VLA, Efficient-LLM, AI-Infra]
category: Paper-Digest
draft: false
---

# Daily AI Papers - Feb 16, 2026

## 1. SLA2: Sparse-Linear Attention with Learnable Routing and QAT

**Authors**: Jintao Zhang et al.  
**arXiv**: [2602.12675](https://arxiv.org/abs/2602.12675)  
**Topic**: Efficient LLM / Attention Optimization

**Key Insight**: SLA2 introduces three improvements over Sparse-Linear Attention (SLA): (I) a learnable router that dynamically selects sparse vs. linear attention branches; (II) a more faithful sparse-linear attention formulation with learnable ratio combination; (III) sparse + low-bit attention via quantization-aware fine-tuning (QAT).

**Results**: 97% attention sparsity and 18.6x attention speedup on video diffusion models while preserving generation quality.

---

## 2. ARTS: Amortized Reasoning Tree Search

**Authors**: Zesheng Hong et al.  
**arXiv**: [2602.12846](https://arxiv.org/abs/2602.12846)  
**Topic**: RLVR / Reasoning

**Key Insight**: Identifies the "Normalization Squeeze" problem in RLVR—policy gradients systematically suppress rare but valid reasoning paths. ARTS decouples generation from verification and uses Flow Matching to estimate probability flow conservation, enabling robust navigation through sparse, high-entropy search spaces.

**Results**: 74.6% on MATH-500 (BoN@16), matching fully fine-tuned policies (74.7%). Uniquely recovers performance on long-tail subsets where coupled RL collapses to 0%.

---

## 3. R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training

**Authors**: Gengsheng Li et al.  
**arXiv**: [2602.13103](https://arxiv.org/abs/2602.13103)  
**Code**: [GitHub](https://github.com/Gengsheng-Li/R-Diverse)  
**Topic**: Reasoning / Self-Play

**Key Insight**: Self-play frameworks exhibit "Diversity Illusion"—training signals appear diverse but collapse into recurring patterns. Proposes Memory-Augmented Penalty (MAP) and Skill-Aware Measurement (SAM) to measure reasoning skill diversity rather than surface question variation.

**Results**: Consistently outperforms prior self-play methods across 10 math and general reasoning benchmarks.

---

## 4. ABot-M0: VLA Foundation Model for Robotic Manipulation

**Authors**: Yandan Yang et al. (Amap/CVLab)  
**arXiv**: [2602.11236](https://arxiv.org/abs/2602.11236)  
**Code**: [GitHub](https://github.com/amap-cvlab/ABot-Manipulation)  
**Topic**: VLA / Robotics

**Key Insight**: Proposes the Action Manifold Hypothesis—robot actions lie on a low-dimensional, smooth manifold governed by physical laws. Introduces Action Manifold Learning (AML) using a DiT backbone to predict clean, continuous action sequences directly.

**Results**: Unified VLA pre-training framework with 6M trajectories and 9,500 hours of data. Supports cross-platform knowledge transfer for general-purpose embodied intelligence.

---

## 5. DICE: Diffusion LLMs Excel at Generating CUDA Kernels

**Authors**: Haolei Bai et al.  
**arXiv**: [2602.11715](https://arxiv.org/abs/2602.11715)  
**Topic**: AI Infra / Code Generation

**Key Insight**: Introduces CuKe dataset and BiC-RL (bi-phase curated reinforcement learning) framework with two-stage training: CUDA kernel infilling followed by end-to-end generation.

**Results**: New state-of-the-art on KernelBench. Models at 1.7B, 4B, and 8B parameter scales significantly outperform both autoregressive and diffusion LLMs of comparable size.

---

## 6. What does RL improve for Visual Reasoning?

**Authors**: Xirui Li et al.  
**arXiv**: [2602.12395](https://arxiv.org/abs/2602.12395)  
**Topic**: RL Analysis / Multimodal

**Key Insight**: Frankenstein-style analysis framework: (i) causal probing for functional localization; (ii) parameter comparison for update characterization; (iii) model merging for transferability testing. RL primarily refines mid-to-late transformer computation rather than uniformly enhancing visual perception.

**Results**: Reveals RL's true contribution—systematic optimization of vision-to-reasoning alignment, not visual perception itself.

---

*Curated by Amy | Sources: HuggingFace Daily Papers + arXiv*
