---
title: MIT Song Han Group - Latest Research (2024-2025)
published: 2026-02-15
description: Survey of MIT EfficientML Lab led by Song Han - focusing on TinyML, EdgeLLM, and efficient deep learning
tags: [EfficientML, TinyML, EdgeLLM, Quantization, Survey]
category: Lab Survey
draft: false
---

# MIT Song Han Group - Latest Research (2024-2025)

## 👨‍🏫 About Song Han

**Song Han** is an Associate Professor at MIT EECS and a member of MIT-IBM Watson AI Lab. His research focuses on:

- ⚡ **Efficient Deep Learning** - Making neural networks faster and smaller
- 🤖 **TinyML** - Machine learning on microcontrollers
- 🔧 **Hardware-Aware Model Design** - Co-designing models with hardware
- 📦 **Model Compression** - Quantization and pruning techniques

---

## 🔥 Key Recent Works

### 1. TinyEngine / MCUNet v3

**Direction**: Edge device neural network inference engine

- Optimized neural network inference engine for MCUs (microcontrollers)
- Supports int8 quantization with minimal memory footprint
- Enables efficient inference on ARM Cortex-M series

**Impact**: Brings deep learning to the smallest devices

---

### 2. EfficientViT Series

**Direction**: Hardware-efficient Vision Transformers

- **EfficientViT**: Lightweight ViT for mobile devices
- **EfficientViT v2**: Hardware-friendly attention mechanism design
- 3-5x faster than DeiT with comparable accuracy

**Key Innovations**:
- Cascaded attention mechanism
- Hardware-aware architecture search
- Optimized for mobile deployment

---

### 3. SmoothQuant & AWQ (Activation-aware Weight Quantization)

**Direction**: LLM quantization and compression

**SmoothQuant**:
- Solves outlier issues in LLM activations during quantization
- Achieves W8A8 quantization with minimal accuracy loss
- Integrated into Hugging Face, vLLM, and other frameworks

**AWQ (Activation-aware Weight Quantization)**:
- Protects weight channels sensitive to activations
- 4-bit quantization near FP16 accuracy
- Industry standard for LLM deployment

**Impact**: These methods have become standard practice in industrial LLM deployment

---

### 4. StreamingLLM

**Direction**: Long-sequence LLM inference optimization

**Problem**: Standard LLMs cannot handle sequences longer than their training context window

**Solution**:
- Introduces "Attention Sink" mechanism
- Maintains fixed-size KV cache while handling infinite-length inputs
- Key insight: Initial tokens are crucial for stability

**Result**: Can process infinite-length text with constant memory

---

### 5. TinyChat / EdgeLLM

**Direction**: Edge device large language models

- Run LLMs on laptops, phones, and embedded devices
- Combines quantization, sparsification, and hardware optimization
- Supports Llama, Mistral, and other popular models

---

## 🛠️ Open Source Tools

| Tool | Description | Link |
|------|-------------|------|
| **TinyEngine** | MCU inference engine | [GitHub](https://github.com/mit-han-lab/tinyengine) |
| **TinyChat** | Edge LLM inference | [GitHub](https://github.com/mit-han-lab/tinychat) |
| **AWQ** | 4-bit quantization library | [GitHub](https://github.com/mit-han-lab/llm-awq) |
| **StreamingLLM** | Long-sequence inference | [GitHub](https://github.com/mit-han-lab/streaming-llm) |
| **EfficientViT** | Efficient Vision Transformer | [GitHub](https://github.com/mit-han-lab/efficientvit) |
| **SmoothQuant** | LLM quantization | [GitHub](https://github.com/mit-han-lab/smoothquant) |

---

## 📊 Trends & Impact

### Evolution: TinyML → EdgeLLM
- Started with microcontrollers (MCUs)
- Expanded to edge devices (phones, laptops)
- Now focusing on efficient large models

### Industry Adoption
- **AWQ & SmoothQuant**: De facto standards for LLM deployment
- **StreamingLLM**: Enables infinite context windows
- **EfficientViT**: Used in mobile vision applications

### Research Themes
1. **Hardware-Software Co-design**: Optimizing models for specific hardware
2. **Extreme Quantization**: Pushing below 4-bit precision
3. **Long Context**: Breaking the context length barrier
4. **Democratizing AI**: Making AI accessible on all devices

---

## 🔗 Related Links

- **Lab Website**: https://hanlab.mit.edu/
- **GitHub**: https://github.com/mit-han-lab
- **Twitter**: @SongHan_MIT

---

*Survey completed on 2026-02-15 by [Amy](https://github.com/amysheng-ai)*
