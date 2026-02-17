# Daily AI Papers - 深度分析模板

## Frontmatter

```yaml
---
title: "Daily AI Papers - YYYY年MM月DD日"
published: YYYY-MM-DD
description: "精选AI论文日报"
tags: [Daily-Papers, {tag1}, {tag2}]
category: Paper-Digest
draft: false
---
```

## Body Structure

```markdown
# Daily AI Papers - YYYY年MM月DD日

## 今日预览

今天筛选出 **N篇高质量论文**...

**必读推荐**：
- **{Paper Title}**: {一句话亮点}

---

## 论文详解

### 1. {Paper Title}

#### Meta
- **Title**: {完整英文标题}
- **Link**: [arXiv:{id}](https://arxiv.org/abs/{id}) | [Code]({github})（如有）
- **Venue**: {会议/期刊，或 arXiv preprint}
- **Date**: {发表日期}
- **Tags**: {标签，如 RLVR, Reasoning, VLA}
- **推荐度**: ⭐⭐⭐ 必读 / ⭐⭐ 可选 / ⭐ 跳过（{原因一句话}）
- **TL;DR**: {一句话总结 Problem & Contribution}

#### Problem & Contribution
- **解决的问题**: {中文描述}
- **核心想法/方法一句话**: {中文描述}
- **主要贡献**（≤3条）:
  1. {贡献1}
  2. {贡献2}
  3. {贡献3}

#### Method
- **方法结构/流程**: 
  {3-5行描述核心流程}

- **关键设计/公式/模块**（最关键的 2–3 点）:
  - {关键设计1}
  - {关键设计2}
  - {关键设计3}

- **训练/推理成本**:
  - 数据: {数据规模}
  - 参数: {模型大小}
  - 计算: {计算资源}
  - 依赖: {关键依赖}

#### Evidence
- **Benchmark / setting**: {数据集/设置}
- **对比对象（baselines）**: {对比方法}
- **关键结果（数字/提升幅度）**: 
  - {指标1}: {数值} (+{提升})
  - {指标2}: {数值}
- **消融/失败案例/局限**: {一句话总结}

#### Takeaways
- **可以迁移到什么场景**: {应用场景}
- **风险/注意点**: {潜在问题}
- **下一步动作**: {复现 / 读前置 / 做实验}（具体到一条）

---

## 总结

| 论文 | 推荐度 | TL;DR | 下一步 |
|------|--------|-------|--------|
| {Title} | ⭐⭐⭐ | {一句话} | {动作} |
...

**今日趋势观察**：
1. {趋势1}
2. {趋势2}

---

*Curated by Amy 🤖*
```

## 语言规范

### ✅ DO:
- 正文用**中文**
- 论文标题保持**英文**（不翻译）
- 专业术语可用英文（RLVR, VLA, Action Manifold）
- 作者名、arXiv ID、链接保持英文
- 数字、指标保持原文

### ❌ DON'T:
- 不要把论文标题翻译成中文
- 不要把专业术语强行翻译成中文
- 不要用英文写正文（除了必要的技术术语）

### 推荐度标准
- **⭐⭐⭐ 必读**: 核心领域突破、顶级机构、扎实实验
- **⭐⭐ 可选**: 相关领域有价值、方法有新意
- **⭐ 跳过**: 增量小、非核心领域、质量一般

## Git Config

**IMPORTANT**: Set git identity before commit:
```bash
git config user.name "Amy"
git config user.email "amysheng.ai@outlook.com"
```

## Workflow

1. Set proxy: `export http_proxy=http://127.0.0.1:7890`
2. Fetch papers from HF Daily Papers + arXiv (weekdays only)
3. Filter based on topics and institution criteria
4. For each selected paper, fill in all sections of the template
5. Write blog post
6. Clone AmyBlog and set git config (Amy)
7. Copy post to `src/content/posts/`
8. Commit and push
9. Send same content to Feishu

## Filter Criteria

### Topics (Priority):
- RLVR (Reinforcement Learning with Verifiable Rewards)
- Reasoning (Chain-of-Thought, Test-time Compute)
- Agentic RL
- VLA (Vision-Language-Action Models)
- Efficient LLM (Quantization, Sparsity, Attention)
- AI Infra (Training frameworks, Code generation)

### Exclusions:
- Vertical applications (medical, chemistry, finance, etc.)
- GNN (Graph Neural Networks)
- Pure theory without experiments
- Hardware (unless paradigm-changing)
- Non-top-tier institutions without significant novelty

### Institution Priority:
Top-tier: MIT, Stanford, CMU, Tsinghua, etc. / 知名研究机构 / 大厂研究院
