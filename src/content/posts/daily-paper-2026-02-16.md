---
title: "Daily AI Papers - Feb 16, 2026"
published: 2026-02-16
description: "Daily curated AI papers focusing on RLVR, Agentic RL, Reasoning, and Efficient LLMs"
tags: [Daily-Papers, RLVR, Agentic-RL, Reasoning, Efficient-LLM]
category: Paper-Digest
draft: false
---

# Daily AI Papers - Feb 16, 2026

📚 Curated papers from arXiv (cs.AI, cs.LG, cs.CL) - 340+ papers published today.

---

## 🔥 Core Topics

### 1. Look Inward to Explore Outward: Learning Temperature Policy from LLM Internal States via Hierarchical RL
- **Authors**: Yixiao Zhou et al.
- **arXiv**: [abs/2602.13035](https://arxiv.org/abs/2602.13035)
- **Topic**: RLVR (Reinforcement Learning from Verifiable Rewards)
- **💡 Key Insight**: Proposes "Introspective LLM" - a hierarchical RL framework that learns to control sampling temperature during generation based on hidden states. Temperature and token policies are jointly optimized from downstream rewards using coordinate ascent.
- **🏆 Impact**: Outperforms fixed and heuristic temperature baselines on mathematical reasoning benchmarks, with interpretable exploration behaviors aligned with reasoning uncertainty.

---

### 2. Consistency of Large Reasoning Models Under Multi-Turn Attacks
- **Authors**: Yubo Li et al.
- **arXiv**: [abs/2602.13093](https://arxiv.org/abs/2602.13093)
- **Topic**: Reasoning / Adversarial Robustness
- **💡 Key Insight**: Evaluates 9 frontier reasoning models under adversarial attacks. While reasoning confers meaningful robustness, all models exhibit vulnerability profiles. Identifies 5 failure modes: Self-Doubt, Social Conformity, Suggestion Hijacking, Emotional Susceptibility, and Reasoning Fatigue.
- **🏆 Impact**: Highlights that reasoning capabilities do not automatically confer adversarial robustness. Confidence-Aware Response Generation (CARG) fails for reasoning models due to overconfidence from extended reasoning traces.

---

### 3. TRACE: Temporal Reasoning via Agentic Context Evolution for Streaming Electronic Health Records
- **Authors**: Zhan Qu et al.
- **arXiv**: [abs/2602.12833](https://arxiv.org/abs/2602.12833)
- **Topic**: Agentic RL / Healthcare AI
- **💡 Key Insight**: A framework enabling temporal clinical reasoning with frozen LLMs using a dual-memory architecture (Global Protocol + Individual Protocol). Four agentic components (Router, Reasoner, Auditor, Steward) coordinate over structured memory for temporal inference.
- **🏆 Impact**: Significantly improves next-event prediction accuracy and protocol adherence on MIMIC-IV dataset while maintaining bounded inference cost and producing interpretable reasoning traces.

---

## 💎 Efficient LLM & Edge AI

### 4. Quantization-Aware Collaborative Inference for Large Embodied AI Models
- **Authors**: Zhonghao Lyu et al.
- **arXiv**: [abs/2602.13052](https://arxiv.org/abs/2602.13052)
- **Topic**: Efficient LLM / Quantization / Edge AI
- **💡 Key Insight**: Investigates quantization-aware collaborative inference for embodied AI systems. Derives bounds on quantization rate-inference distortion function and formulates joint quantization bit-width and computation frequency design.
- **🏆 Impact**: Demonstrates effective balancing of inference quality, latency, and energy consumption in edge embodied AI systems through real-world testbed experiments.

---

### 5. TriGen: NPU Architecture for End-to-End Acceleration of Large Language Models based on SW-HW Co-Design
- **Authors**: Jonghun Lee et al.
- **arXiv**: [abs/2602.12962](https://arxiv.org/abs/2602.12962)
- **Topic**: Efficient LLM / Hardware Acceleration
- **💡 Key Insight**: Novel NPU architecture using microscaling (MX) for low-precision computation, eliminating specialized hardware for nonlinear operations via LUT, and employing scheduling techniques for limited on-chip memory.
- **🏆 Impact**: Achieves 2.73x performance speedup and 52% less memory transfer over baseline NPU with negligible accuracy loss.

---

### 6. SLA2: Sparse-Linear Attention with Learnable Routing and QAT
- **Authors**: Jintao Zhang et al.
- **arXiv**: [abs/2602.12675](https://arxiv.org/abs/2602.12675)
- **Topic**: Efficient Attention / Video Generation
- **💡 Key Insight**: Introduces learnable router for dynamic sparse/linear attention selection, more faithful sparse-linear attention formulation, and sparse + low-bit attention via quantization-aware fine-tuning.
- **🏆 Impact**: 97% attention sparsity and 18.6x attention speedup on video diffusion models while preserving generation quality.

---

## 🔬 Related & Emerging

### 7. Which Algorithms Can Graph Neural Networks Learn?
- **Authors**: Christopher Morris et al.
- **arXiv**: [abs/2602.13106](https://arxiv.org/abs/2602.13106)
- **Topic**: Neural Algorithmic Reasoning
- **💡 Key Insight**: Theoretical framework characterizing sufficient conditions for MPNNs to learn algorithms from small training instances and generalize to arbitrary-sized inputs. Covers shortest paths, MST, knapsack, and Bellman-Ford.
- **🏆 Impact**: Bridges learning-based methods and classical algorithms with provable generalization guarantees.

---

### 8. Semantic Chunking and the Entropy of Natural Language
- **Authors**: Weishun Zhong et al.
- **arXiv**: [abs/2602.13194](https://arxiv.org/abs/2602.13194)
- **Topic**: Language Theory / LLM Fundamentals
- **💡 Key Insight**: Statistical model capturing multi-scale structure of natural language through self-similar semantic chunking. Predicts entropy rate increases with semantic complexity.
- **🏆 Impact**: First-principles account of ~1 bit/character entropy rate in English; reveals entropy is not fixed but scales with corpus complexity.

---

## 📊 Summary
- **Total papers**: 8
- **Core topics**: 3 (RLVR, Reasoning, Agentic)
- **Efficient LLM**: 3 (Quantization, Hardware, Attention)
- **Emerging/Related**: 2 (Algorithmic Reasoning, Language Theory)

*Curated by [Amy](https://github.com/amysheng-ai) | Generated at 2026-02-16 18:10*
