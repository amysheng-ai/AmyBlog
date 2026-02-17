# Daily AI Papers Skill

> End-to-end workflow for curating, analyzing, and publishing daily AI paper digests.

## Overview

This skill provides a complete workflow for:
- Fetching papers from HuggingFace Daily Papers and arXiv
- Filtering based on research interests and quality criteria
- Generating deep analysis reports in Chinese with English titles
- Publishing to AmyBlog (GitHub Pages) and delivering to Feishu

## Schedule

| Day | Source | Notes |
|-----|--------|-------|
| Mon-Fri | HuggingFace + arXiv | arXiv updates at 08:00 Beijing Time |
| Sat-Sun | HuggingFace only | arXiv doesn't update on weekends |

## Prerequisites

- Network proxy: `http://127.0.0.1:7890` for HuggingFace access
- Git identity: `user.name="Amy"`, `user.email="amysheng.ai@outlook.com"`
- Blog repo: `https://github.com/amysheng-ai/AmyBlog`
- Target Feishu user: `ou_168ea1a1162ad66582b40ec15e5a2950`

## Workflow

### Phase 1: Setup & Configuration

#### 1.1 Set Network Proxy
```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

Or use clashctl:
```bash
source /home/ubuntu/clashctl/scripts/cmd/clashctl.sh && clashon
```

#### 1.2 Verify Git Identity
```bash
git config user.name "Amy"
git config user.email "amysheng.ai@outlook.com"
```

### Phase 2: Fetch Papers

#### 2.1 HuggingFace Daily Papers
```bash
# By date
web_fetch "https://huggingface.co/papers/date/YYYY-MM-DD" --maxChars 5000

# Latest (default page)
web_fetch "https://huggingface.co/papers" --maxChars 5000
```

**Extract:**
- Paper titles
- Authors and affiliations
- arXiv IDs
- GitHub links (if available)
- Abstract summaries

#### 2.2 arXiv (Weekdays Only)
```bash
# Recent AI/ML papers
web_fetch "https://arxiv.org/list/cs.AI/recent" --maxChars 8000
web_fetch "https://arxiv.org/list/cs.LG/recent" --maxChars 8000
web_fetch "https://arxiv.org/list/cs.CL/recent" --maxChars 8000
```

**Note:** arXiv updates at 08:00 Beijing Time (UTC 00:00), Monday-Friday only.

### Phase 3: Filter & Select

#### 3.1 Inclusion Criteria (Topic)

**✅ Include:**
- RLVR (Reinforcement Learning with Verifiable Rewards)
- Reasoning (Chain-of-Thought, Test-time compute)
- Agentic RL (Multi-agent, Tool use)
- VLA (Vision-Language-Action models)
- Efficient LLM (Quantization, Pruning, Distillation)
- AI Infra (Training frameworks, Inference optimization)

**❌ Exclude:**
- GNN (Graph Neural Networks)
- Vertical applications (Medical, Chemistry, Finance, Weather)
- Pure theory without experimental validation
- Hardware (unless paradigm-changing)

#### 3.2 Institution Priority

**🏛️ Must be top-tier:**
- MIT, Stanford, CMU, Berkeley, Google DeepMind, OpenAI, Anthropic
- Tsinghua (THUNLP), Peking University, Shanghai AI Lab
- FAIR, Microsoft Research, Google Research

**⚠️ Non-top-tier:** Only include if significant novelty

#### 3.3 Quality Signals

**✅ Positive signals:**
- Has open-source code/GitHub repo
- Published at top venues (NeurIPS, ICML, ICLR, ACL)
- Strong experimental validation
- Novel method with clear contribution

**❌ Negative signals:**
- Pure application/dataset paper
- Incremental method with weak experiments
- ArXiv-only without code

### Phase 4: Deep Analysis

For each selected paper, use the **deep-paper-analysis** skill structure:

#### 4.1 Paper Meta
```markdown
| Attribute | Content |
|-----------|---------|
| **arXiv** | [ID](https://arxiv.org/abs/ID) |
| **Title** | {English title} |
| **Authors** | {Names} ({Affiliation}) |
| **Code** | [GitHub](URL) ⭐ {stars} |
| **Direction** | {RLVR/Reasoning/VLA/etc.} |
| **Rating** | ⭐⭐⭐ 必读 / ⭐⭐ 可选 / ⭐ 跳过 |
```

#### 4.2 Core Analysis Sections

**Problem & Contribution:**
- What problem does it solve?
- What's the key insight?
- Why does it matter?

**Method:**
- High-level approach
- Key technical innovations
- Design choices and rationale

**Evidence:**
- Experimental setup (datasets, baselines, metrics)
- Key results with specific numbers
- Ablation studies

**Takeaways:**
- Best applied when...
- Limitations
- Next action (read/follow/star)

### Phase 5: Write Report

#### 5.1 Report Structure

```markdown
# Daily AI Papers - YYYY年MM月DD日

## 今日预览
[3-4句中文亮点速览，论文标题用英文]

---

## 论文详解

### 1. {Paper Title in English}
**作者**: {Authors} 等  
**链接**: [arXiv:{id}](...) | [代码](...)  
**方向**: {中文分类}

**核心创新**:
[中文详细描述]

**实验结果**:
[具体数字]

---

## 总结
| 论文 | 主题 | 核心贡献 |
|------|------|----------|
| {英文标题} | {中文主题} | {中文贡献} |
...

**今日趋势观察**:
1. {中文}
2. {中文}
```

#### 5.2 Language Rules

| Element | Format | Example |
|---------|--------|---------|
| Body text | Chinese | 使用 DiT 骨干网络... |
| Paper titles | English (no translation) | SLA2: Sparse-Linear Attention... |
| Technical terms | English allowed | RLVR, VLA, Action Manifold |
| Author names | English | Ning Ding, Andrej Karpathy |
| arXiv IDs | English | arXiv:2602.12125 |

#### 5.3 Rating System

| Rating | Meaning | Action |
|--------|---------|--------|
| ⭐⭐⭐ | 必读 | Core breakthrough, top-tier, strong evidence |
| ⭐⭐ | 可选 | Related value, solid but not core interest |
| ⭐ | 跳过 | Outside scope or weak contribution |

**Always include one-sentence reason for ⭐ ratings**

### Phase 6: Publish & Deliver

#### 6.1 Create Blog Post

**File naming:**
```
src/content/posts/daily-paper-YYYY-MM-DD.md
```

**Required frontmatter (CRITICAL ⚠️):**
```yaml
---
title: Daily AI Papers - YYYY年MM月DD日
published: YYYY-MM-DD
description: {3-4 sentence preview}
tags: [Daily Papers, AI, {topics}]
category: Papers
draft: false
---
```

**CRITICAL:** Without frontmatter, GitHub Pages deployment will fail!

#### 6.2 Commit & Push

```bash
cd /path/to/AmyBlog
git add src/content/posts/daily-paper-YYYY-MM-DD.md
git commit -m "Add daily papers for YYYY-MM-DD"
git push origin main
```

#### 6.3 Send to Feishu

Send the full blog post content (or summary with link) to:
- Feishu user: `ou_168ea1a1162ad66582b40ec15e5a2950`
- Channel: feishu

### Phase 7: Verification

#### 7.1 Pre-commit Checklist
- [ ] YAML frontmatter present and correct
- [ ] Paper titles in English (not translated)
- [ ] Body text in Chinese
- [ ] All arXiv links work
- [ ] GitHub repos linked (if available)
- [ ] Ratings assigned with reasons
- [ ] No "filtering stats" or meta commentary

#### 7.2 Post-publish Verification
- [ ] GitHub Actions deployment successful
- [ ] Blog post accessible at expected URL
- [ ] Feishu message delivered

## Cron Job Setup

### Create Daily Cron

```bash
openclaw cron add \
  --name "Daily AI Papers - {user}" \
  --cron "0 8 * * *" \
  --tz "Asia/Shanghai" \
  --session isolated \
  --message "Daily AI Papers task: Read templates/daily-papers-template.md and TOOLS.md, then execute the daily paper curation workflow." \
  --announce \
  --channel feishu \
  --to "{user_id}"
```

### Template Files

**Read at execution time:**
- `templates/daily-papers-template.md` - Full template structure
- `TOOLS.md` - Standards and criteria (sections: "Daily AI Papers - 数据源", "推送 schedule", "写作标准", "筛选标准")

## Data Sources Reference

### HuggingFace Daily Papers
- **Format:** `https://huggingface.co/papers/date/YYYY-MM-DD`
- **Example:** https://huggingface.co/papers/date/2026-02-16
- **Default:** https://huggingface.co/papers
- **Update:** Daily

### arXiv
- **CS.AI:** https://arxiv.org/list/cs.AI/recent
- **CS.LG:** https://arxiv.org/list/cs.LG/recent
- **CS.CL:** https://arxiv.org/list/cs.CL/recent
- **Update:** Weekdays (Mon-Fri) at 08:00 Beijing Time
- **Weekends:** No updates

## Common Pitfalls

### ❌ Frontmatter Missing
**Mistake:** Starting with `# Title` directly  
**Fix:** Always include YAML frontmatter first

### ❌ Wrong Date Format
**Mistake:** `published: 2026-02-16 08:00`  
**Fix:** `published: 2026-02-16`

### ❌ Paper Title Translated
**Mistake:** `SLA2: 稀疏线性注意力...`  
**Fix:** `SLA2: Sparse-Linear Attention...`

### ❌ Body Text in English
**Mistake:** `Using DiT backbone...`  
**Fix:** `使用 DiT 骨干网络...`

### ❌ Including Meta Commentary
**Mistake:** "筛选了20篇，排除原因..."  
**Fix:** Remove filtering stats, focus on selected papers only

### ❌ Missing Proxy
**Mistake:** HuggingFace fetch fails  
**Fix:** Always set `export http_proxy=http://127.0.0.1:7890`

## Tools Integration

### With ai-paper-survey
- Use survey skill for lab/group overviews
- Identify key researchers to watch
- Cross-reference with daily papers

### With deep-paper-analysis
- Flag ⭐⭐⭐ papers for deep dive
- Generate detailed analysis for important works
- Build personal knowledge base

## Example Output

See completed daily digests:
- `src/content/posts/daily-paper-2026-02-16.md`
- Published: https://amysheng-ai.github.io/AmyBlog/posts/daily-paper-2026-02-16

## Related Files

- `templates/daily-papers-template.md` - Report template
- `TOOLS.md` - Data sources, schedule, writing standards, filter criteria
- `skills/deep-paper-analysis/SKILL.md` - Single paper deep dive
- `skills/ai-paper-survey/SKILL.md` - Lab/group surveys

---

*Skill created: 2026-02-16 based on established workflow*
