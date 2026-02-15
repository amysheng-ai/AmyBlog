---
title: MIT Song Han (EfficientML Lab) - Research Survey 2024-2025
published: 2026-02-15
description: Comprehensive survey of MIT EfficientML Lab led by Song Han - covering latest efficient deep learning research with arXiv papers and GitHub repos
tags: [EfficientML, LLM, Quantization, Sparse Attention, EdgeAI, Survey]
category: Lab Survey
draft: false
---

# MIT Song Han (EfficientML Lab) - Research Survey 2024-2025

> **Lab**: [MIT EfficientML Lab](https://hanlab.mit.edu/) | **PI**: [Song Han](https://songhan.mit.edu/)  
> **Focus**: Efficient Deep Learning, TinyML, Edge AI, LLM Acceleration  
> **Survey Date**: 2026-02-15

---

## 🏆 Most Influential Works (Highly Cited/Impactful)

### 1. AWQ: Activation-aware Weight Quantization
**Publication**: MLSys 2024 **Best Paper Award** 🥇

- **Problem**: LLMs are too large for edge deployment
- **Solution**: Protects salient weight channels by observing activation magnitudes
- **Impact**: 4-bit quantization with near-FP16 accuracy; adopted by industry
- **GitHub**: [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) ⭐ 3.4k
- **Paper**: [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

---

### 2. StreamingLLM: Infinite Context Length
**Publication**: ICLR 2024

- **Problem**: LLMs can't handle sequences longer than training context
- **Solution**: "Attention Sink" mechanism - initial tokens anchor attention
- **Impact**: Process infinite-length text with constant memory
- **GitHub**: [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) ⭐ 7.2k
- **Paper**: [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)

---

### 3. EfficientViT: High-Resolution Vision Transformer
**Publication**: CVPR 2023 / ICCV 2023

- **Problem**: ViTs are too slow for high-resolution images
- **Solution**: Hardware-efficient attention with cascaded attention mechanism
- **Impact**: 3-5x faster than DeiT with comparable accuracy
- **GitHub**: [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) ⭐ 3.2k
- **Paper**: [arXiv:2205.14756](https://arxiv.org/abs/2205.14756)

---

### 4. SmoothQuant: Accurate Quantization for LLMs
**Publication**: ICML 2023

- **Problem**: Activations in LLMs have outliers that hurt quantization
- **Solution**: Mathematically smooths activation outliers to weights
- **Impact**: Enables W8A8 quantization; integrated into Hugging Face, vLLM
- **GitHub**: [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)
- **Paper**: [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

---

### 5. BEVFusion: Multi-Task Multi-Sensor Fusion
**Publication**: ICRA 2023

- **Problem**: Autonomous driving needs efficient multi-sensor fusion
- **Solution**: Unified BEV representation for camera + LiDAR
- **Impact**: 3D detection and segmentation with unified framework
- **GitHub**: [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) ⭐ 3k
- **Paper**: [arXiv:2205.13542](https://arxiv.org/abs/2205.13542)

---

## 🔥 Latest Works (2024-2025)

### 6. DuoAttention: Efficient Long-Context LLM Inference
**Publication**: ICLR 2025

- **Innovation**: Separates retrieval heads and streaming heads in attention
- **Benefit**: Enables efficient long-context inference by pruning non-essential heads
- **GitHub**: [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention)
- **Paper**: [arXiv:2410.10819](https://arxiv.org/abs/2410.10819)

---

### 7. Radial Attention: O(n log n) Sparse Attention
**Publication**: NeurIPS 2025

- **Innovation**: Energy decay pattern for video generation
- **Benefit**: Sub-quadratic attention complexity for long video
- **GitHub**: [mit-han-lab/radial-attention](https://github.com/mit-han-lab/radial-attention)

---

### 8. LPD: Locality-aware Parallel Decoding
**Publication**: ICLR 2026 Oral 🎤

- **Innovation**: Parallel token generation for autoregressive models
- **Benefit**: Faster image generation while maintaining quality
- **GitHub**: [mit-han-lab/lpd](https://github.com/mit-han-lab/lpd)

---

### 9. FastRL: Efficient Reasoning RL Training
**Publication**: ASPLOS 2026

- **Innovation**: Adaptive drafter for long-tail reasoning scenarios
- **Benefit**: Efficient RL training for reasoning tasks
- **GitHub**: [mit-han-lab/fastrl](https://github.com/mit-han-lab/fastrl)

---

### 10. QServe: W4A8KV4 Quantization for LLM Serving
**Publication**: MLSys 2025

- **Innovation**: System-algorithm co-design for serving quantized LLMs
- **Benefit**: Production-ready quantized LLM serving
- **GitHub**: [mit-han-lab/omniserve](https://github.com/mit-han-lab/omniserve)

---

### 11. VILA-U: Unified Visual Understanding and Generation
**Publication**: ICLR 2025

- **Innovation**: Single model for both image understanding and generation
- **Benefit**: Unified vision foundation model
- **GitHub**: [mit-han-lab/vila-u](https://github.com/mit-han-lab/vila-u)

---

### 12. XAttention: Block Sparse Attention
**Publication**: ICML 2025

- **Innovation**: Antidiagonal scoring for sparse attention patterns
- **Benefit**: Efficient attention with structured sparsity
- **GitHub**: [mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention)

---

### 13. Quest: Query-Aware Sparsity
**Publication**: ICML 2024

- **Innovation**: Dynamic sparsity based on query content
- **Benefit**: Efficient long-context inference
- **GitHub**: [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest)

---

### 14. StreamingVLM: Infinite Video Streams
**Publication**: 2024

- **Innovation**: Extends StreamingLLM to video understanding
- **Benefit**: Real-time understanding of infinite video streams
- **GitHub**: [mit-han-lab/streaming-vlm](https://github.com/mit-han-lab/streaming-vlm)

---

### 15. Four Over Six (4/6): NVFP4 Quantization
**Publication**: 2025

- **Innovation**: Adaptive block scaling for NVFP4 format
- **Benefit**: More accurate 4-bit quantization on NVIDIA hardware
- **GitHub**: [mit-han-lab/fouroversix](https://github.com/mit-han-lab/fouroversix)

---

## 🛠️ Core Open Source Tools

| Tool | Description | Stars | Link |
|------|-------------|-------|------|
| **llm-awq** | 4-bit LLM quantization | ⭐ 3.4k | [GitHub](https://github.com/mit-han-lab/llm-awq) |
| **streaming-llm** | Infinite context length | ⭐ 7.2k | [GitHub](https://github.com/mit-han-lab/streaming-llm) |
| **efficientvit** | Efficient vision transformers | ⭐ 3.2k | [GitHub](https://github.com/mit-han-lab/efficientvit) |
| **bevfusion** | Multi-sensor fusion for AV | ⭐ 3k | [GitHub](https://github.com/mit-han-lab/bevfusion) |
| **smoothquant** | W8A8 LLM quantization | - | [GitHub](https://github.com/mit-han-lab/smoothquant) |
| **omniserve** | LLM serving system | - | [GitHub](https://github.com/mit-han-lab/omniserve) |
| **tinyengine** | MCU inference engine | ⭐ 1.5k | [GitHub](https://github.com/mit-han-lab/tinyengine) |
| **tinychat** | Edge LLM chat | - | [GitHub](https://github.com/mit-han-lab/tinychat) |

---

## 📊 Research Themes & Trends

### 1. **Extreme Quantization** 🔧
- AWQ (4-bit) → Four Over Six (NVFP4)
- Pushing precision limits while maintaining accuracy
- Hardware-algorithm co-design

### 2. **Long Context & Streaming** 🌊
- StreamingLLM → StreamingVLM → DuoAttention
- From infinite text to infinite video
- Memory-efficient attention mechanisms

### 3. **Sparse Attention Patterns** 🎯
- Quest (query-aware) → XAttention (antidiagonal) → Radial Attention (energy decay)
- Structured sparsity for efficiency
- O(n log n) and sub-quadratic complexity

### 4. **Unified Vision Models** 👁️
- VILA-U: Understanding + Generation in one model
- Efficient high-resolution processing

### 5. **Edge & Real-Time Deployment** 📱
- TinyEngine (MCU) → TinyChat (Edge LLM)
- Real-time inference on resource-constrained devices

---

## 🔗 Quick Links

- **Lab Website**: [hanlab.mit.edu](https://hanlab.mit.edu/)
- **GitHub Org**: [github.com/mit-han-lab](https://github.com/mit-han-lab)
- **Twitter**: [@SongHan_MIT](https://twitter.com/SongHan_MIT)
- **YouTube Talks**: Search "Song Han MIT efficient ML"

---

## 💡 Key Insights

1. **From Research to Production**: AWQ and SmoothQuant became industry standards
2. **Infinite Context**: StreamingLLM paradigm extends to video (StreamingVLM)
3. **Hardware-Software Co-design**: Every algorithm considers deployment target
4. **Open Source Culture**: All major works have open-source implementations

---

*Survey completed on 2026-02-15 by [Amy](https://github.com/amysheng-ai)*  
*Sources: arXiv, GitHub, MIT EfficientML Lab website*
