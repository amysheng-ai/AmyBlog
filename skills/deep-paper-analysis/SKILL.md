# Deep Paper Analysis Skill

> Comprehensive framework for in-depth analysis and summary of single AI papers.

## Overview

This skill provides a structured approach to deeply analyze AI research papers, going beyond surface-level summaries to extract core insights, methodology details, experimental findings, and research implications.

## Use Cases

- Deep dive into seminal papers
- Literature review preparation
- Research idea generation
- Paper discussion/reading group notes
- Personal knowledge base building

## Prerequisites

- `web_fetch` - For fetching paper PDFs and abstracts
- `web_search` - For finding related work (optional)
- `notion` - For storing structured notes (optional)

## Analysis Framework

### Phase 1: Paper Card (Quick Overview)

Create a concise summary card with essential information:

```markdown
| Attribute | Content |
|-----------|---------|
| **arXiv** | [ID](link) |
| **Title** | Full paper title |
| **Authors** | Names + Affiliations |
| **Code** | GitHub link (if available) |
| **Tags** | 3-5 key topic tags |
| **TL;DR** | One-sentence core contribution |
```

### Phase 2: Motivation & Context

#### 2.1 Background Chain
- **What existed before?** (2-3 key prior works)
- **What was the limitation?** (Why previous methods insufficient)
- **Why now?** (What changed to enable this work)

#### 2.2 Core Problem Statement
> Clear articulation of the specific problem being solved

### Phase 3: Core Insights (The "Aha!" Moments)

#### 3.1 Main Theoretical/Conceptual Insight
- The key discovery that makes the paper work
- Usually 1-2 major insights
- Explain intuition before math

#### 3.2 Why It Matters
- Implications for the field
- Connections to broader trends
- Potential impact

### Phase 4: Method Deep Dive

#### 4.1 Framework Overview
- High-level architecture/algorithm
- Key components
- Data flow

#### 4.2 Key Design Choices
Present as comparison table:

| Design Choice | Options | Why This Choice | Impact |
|--------------|---------|----------------|--------|
| Component A | Option 1 vs 2 | Reasoning | Effect |

#### 4.3 Technical Details
- Critical hyperparameters
- Implementation nuances
- Tricks that make it work

### Phase 5: Experiments & Results

#### 5.1 Experimental Setup
- **Tasks/Datasets**: What benchmarks
- **Baselines**: What compared against
- **Metrics**: How evaluated
- **Implementation**: Key details

#### 5.2 Key Findings
Present main results with interpretation:

**Finding 1: [Name]**
- Setup: What experiment
- Result: Numbers/outcomes
- Analysis: Why this happens
- Implications: What it means

**Finding 2: [Name]**
- ...

#### 5.3 Ablation Studies
- Which components matter most
- Sensitivity analysis
- Failure modes

### Phase 6: Deep Analysis

#### 6.1 Why It Works (Intuition)
- Mechanistic explanation
- Visualization/conceptual diagrams (described)
- Edge cases where it might fail

#### 6.2 Connections & Implications
- Links to concurrent works
- Impact on downstream research
- Practical deployment considerations

### Phase 7: Critical Evaluation

#### 7.1 Limitations (Honest Assessment)
- **Technical limitations**: What can't it do?
- **Scope limitations**: Where not applicable?
- **Resource limitations**: Compute/data requirements?
- **Evaluation limitations**: Benchmark coverage?

#### 7.2 Open Questions
- What remains unsolved?
- Natural extensions
- Future work directions

### Phase 8: Related Work Context

#### 8.1 Position in Literature
| Method | Relationship | Key Difference |
|--------|--------------|----------------|
| Prior Work A | Extension | What's new |
| Concurrent B | Alternative | Trade-offs |
| Follow-up C | Foundation | Builds on this |

#### 8.2 Citation Graph
- 2-3 must-read prerequisites
- 2-3 important follow-up works

## Output Structure Template

```markdown
# [Paper Title]

> **TL;DR**: One-sentence summary

---

## 📋 Paper Card

| Attribute | Content |
|-----------|---------|
| **arXiv** | [ID](link) |
| ... | ... |

---

## 🎯 Motivation

### Background
...

### Problem Statement
> ...

---

## 💡 Core Insights

### Insight 1: [Name]
...

### Insight 2: [Name]
...

---

## 🔧 Method

### Framework Overview
...

### Key Design Choices
| Choice | Options | Decision | Rationale |
|--------|---------|----------|-----------|
| ... | ... | ... | ... |

### Technical Details
...

---

## 🔬 Experiments

### Setup
- **Tasks**: ...
- **Baselines**: ...
- **Metrics**: ...

### Key Findings

#### Finding 1: [Name]
- **Setup**: ...
- **Result**: ...
- **Analysis**: ...

#### Finding 2: [Name]
...

---

## 📊 Deep Analysis

### Why It Works
...

### Connections
...

---

## ⚠️ Limitations & Open Questions

### Limitations
1. ...
2. ...

### Open Questions
- ...

---

## 🔗 Related Work

| Method | Relationship | Key Difference |
|--------|--------------|----------------|
| ... | ... | ... |

---

## 📝 Summary

**Key Takeaways**:
1. ...
2. ...
3. ...

**Best Applied When**:
- Scenario 1
- Scenario 2

**Impact**: [High/Medium/Low] - Why
```

## Quality Guidelines

### Do:
- ✅ Start with intuition, then math
- ✅ Use tables for comparisons
- ✅ Include specific numbers when available
- ✅ Explain "why" not just "what"
- ✅ Connect to broader research trends
- ✅ Be honest about limitations
- ✅ Use consistent formatting

### Don't:
- ❌ Just copy abstract
- ❌ Skip methodology details
- ❌ Omit negative results
- ❌ Overclaim contributions
- ❌ Ignore related work context

## Example Application

See completed analysis:
- `memory/papers/g-opd-2602.12125.md` - G-OPD deep dive example

## Variations by Paper Type

### Theory Papers
- Emphasize: proofs, assumptions, implications
- Add section: Theoretical guarantees

### Empirical Papers
- Emphasize: experimental design, results, ablations
- Add section: Reproducibility checklist

### Survey Papers
- Emphasize: categorization, trends, gaps
- Add section: Taxonomy/map of field

### Position Papers
- Emphasize: arguments, evidence, counter-arguments
- Add section: Debate/controversy context

## Tools Integration

### With Notion
```bash
# After analysis, create structured page
curl -X POST "https://api.notion.com/v1/pages" \
  -H "Authorization: Bearer $NOTION_KEY" \
  -H "Notion-Version: 2025-09-03" \
  -d '{
    "parent": {"database_id": "YOUR_DB_ID"},
    "properties": {
      "标题": {"title": [{"text": {"content": "[arXiv ID] Paper Title"}}]},
      "Topics": {"multi_select": [{"name": "Tag1"}, {"name": "Tag2"}]},
      "阅读状态": {"status": {"name": "已读完"}},
      "Type": {"multi_select": [{"name": "Theory/Empirical"}]}
    }
  }'
```

### With Daily Digest
- Extract key findings for daily paper summaries
- Flag high-impact papers for deeper dive

## Related Skills

- `ai-paper-survey` - For lab/group-level surveys
- `notion` - For structured note storage
- `web_search` - For finding related work

---

*Skill created: 2026-02-15 based on G-OPD analysis*
