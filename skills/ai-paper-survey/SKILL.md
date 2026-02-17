# AI Paper Survey Skill

> Systematic methodology for researching AI labs, research groups, and paper collections.

## Overview

This skill provides a structured approach to survey AI research groups, track their latest work, and compile comprehensive reports with proper citations.

## Use Cases

- Lab/group surveys (e.g., MIT Song Han, Berkeley Sergey Levine)
- Research topic deep dives
- Paper digest compilation
- Trend analysis

## Prerequisites

- `web_fetch` - For fetching arXiv/GitHub pages
- `web_search` - For discovering papers (if API available)
- `exec` with `agent-browser` - For dynamic web scraping
- GitHub CLI (`gh`) - For repository queries

## Workflow

### Phase 1: Discovery (Data Collection)

#### 1.1 GitHub Organization Search
```bash
# List recent repositories from a lab's GitHub org
gh api /orgs/{org-name}/repos --paginate | jq '.[] | {name, description, pushed_at}'

# Or use search API
https_proxy=http://127.0.0.1:7890 curl -s "https://api.github.com/search/repositories?q=org:{org-name}+created:>{date}&sort=updated&order=desc"
```

**What to extract:**
- Repository name
- Description (often contains paper venue/year)
- Star count (popularity indicator)
- Last updated (recency)

#### 1.2 arXiv Search
```bash
# Direct arXiv API query
https_proxy=http://127.0.0.1:7890 curl -s "http://export.arxiv.org/api/query?search_query=au:{LastName}_{FirstName}+AND+affiliation:{Institution}&sortBy=submittedDate&max_results=30"
```

**Alternative**: Use `web_fetch` on arXiv search results page

#### 1.3 Huggingface Daily Papers
```bash
# Fetch daily papers feed
web_fetch https://huggingface.co/papers --maxChars 5000

# Or search for specific author/topic
agent-browser open "https://huggingface.co/papers?search={author_name}"
```

**What to extract:**
- Trending papers with abstracts
- Author affiliations
- GitHub links (often included)
- Community discussions

#### 1.4 Lab Website Scan
```bash
# Fetch lab website for overview
web_fetch https://{lab}.mit.edu/ or https://{lab}.berkeley.edu/
```

### Phase 2: Analysis (Data Processing)

#### 2.1 Categorize Papers

**By Impact:**
- 🏆 **Most Influential**: Highly cited, award-winning, industry adoption
- 🔥 **Recent (Last 2 years)**: Latest developments
- 📚 **Foundational**: Seminal works that defined the field

**By Topic:**
- Group related papers under themes
- Identify research evolution trends

#### 2.2 Extract Key Metadata

For each paper, collect:

| Field | Source | Example |
|-------|--------|---------|
| **Title** | arXiv/GitHub | "AWQ: Activation-aware Weight Quantization" |
| **Venue** | GitHub description, arXiv | "MLSys 2024 Best Paper" |
| **arXiv ID** | arXiv URL | 2306.00978 |
| **GitHub Repo** | GitHub search | mit-han-lab/llm-awq |
| **Stars** | GitHub API | ⭐ 3.4k |
| **Problem** | Abstract/Description | "LLMs too large for edge" |
| **Solution** | Abstract/Description | "4-bit quantization" |
| **Impact** | Citations/Adoption | "Industry standard" |

### Phase 3: Synthesis (Report Writing)

#### 3.1 Report Structure

```markdown
# {Lab Name} - Research Survey {Year}

> Lab: [Website](URL) | PI: [Name](URL)  
> Focus: {Research areas}  
> Survey Date: {Date}

---

## 🏆 Most Influential Works

### 1. {Paper Title}
**Publication**: {Venue} {Awards}

- **Problem**: {One sentence}
- **Solution**: {One sentence}
- **Impact**: {Industry/citation impact}
- **GitHub**: [org/repo](URL) ⭐ {stars}
- **Paper**: [arXiv:{id}](https://arxiv.org/abs/{id})

---

## 🔥 Latest Works ({Year Range})

### {N}. {Paper Title}
**Publication**: {Venue}

- **Innovation**: {What's new}
- **Benefit**: {Why it matters}
- **GitHub**: [org/repo](URL)
- **Paper**: [arXiv:{id}](URL)

---

## 🛠️ Open Source Tools

| Tool | Description | Stars | Links |
|------|-------------|-------|-------|
| {name} | {desc} | ⭐ {n} | [GitHub](URL) |

---

## 📊 Research Themes & Trends

### 1. **{Theme Name}** {emoji}
- Evolution: {Paper A} → {Paper B}
- Key insight: {Summary}

---

## 🔗 Quick Links

- **Lab Website**: [url](URL)
- **GitHub Org**: [github.com/{org}](URL)
- **Twitter**: @{handle}

---

## 💡 Key Insights

1. {Insight 1}
2. {Insight 2}

---

*Survey completed on {date} by [Amy](https://github.com/amysheng-ai)*
```

#### 3.2 Writing Guidelines

**Do:**
- Include arXiv links for every paper
- Include GitHub repos when available
- Note conference venues and awards
- Categorize by influence and recency
- Add trend analysis section
- Include star counts for popularity

**Don't:**
- List papers without links
- Mix influential and recent without labeling
- Skip the problem/solution summary
- Forget to mention industry impact

### Phase 4: Verification

#### 4.1 Link Check
```bash
# Verify arXiv links work
curl -I https://arxiv.org/abs/{id}

# Verify GitHub repos exist
gh repo view {org}/{repo}
```

#### 4.2 Completeness Check
- [ ] All papers have arXiv links
- [ ] All repos have GitHub links
- [ ] Venues are specified
- [ ] At least 5 influential works
- [ ] At least 5 recent works
- [ ] Trend analysis included

#### 4.3 Blog Post Format Check (CRITICAL ⚠️)
For Fuwari template, MUST include frontmatter at the top:
```yaml
---
title: [Post Title]
published: [YYYY-MM-DD]
description: [Brief description]
tags: [Tag1, Tag2, Tag3]
category: [Category]
draft: false
---
```

**Common mistakes:**
- ❌ Starting with `# Title` directly (missing frontmatter)
- ❌ Wrong date format
- ❌ Missing required fields

**Always verify** the post has frontmatter before committing to AmyBlog.

## Tools & Commands Reference

### GitHub API Queries

```bash
# List org repos
gh api /orgs/{org}/repos --paginate

# Search repos by date
gh api "/search/repositories?q=org:{org}+created:>2024-01-01&sort=updated&order=desc"

# Get repo details
gh repo view {owner}/{repo} --json name,description,stargazersCount,pushedAt
```

### arXiv API

```bash
# Author + affiliation search
curl "http://export.arxiv.org/api/query?search_query=au:{Last}_{First}+AND+affiliation:{Inst}&sortBy=submittedDate"

# Direct paper access
curl "https://export.arxiv.org/api/query?id_list={id1},{id2}"
```

### Web Scraping

```bash
# Fetch and extract
web_fetch https://github.com/{org} --maxChars 5000

# Dynamic content (JavaScript)
agent-browser open "{url}"
agent-browser snapshot
```

## Example Output

See completed surveys:
- `src/content/posts/sergey-levine-survey-2025-2026.md`
- `src/content/posts/mit-songhan-survey-2024-2025.md`

## Troubleshooting

### arXiv API returns empty
- Check author name format: `Han_S` vs `Song_Han`
- Try affiliation-only search
- Use web_fetch on arXiv search results page instead

### GitHub API rate limited
- Use authenticated requests (token)
- Add delays between requests
- Cache results locally

### Can't find paper on arXiv
- Check if it's a conference paper (not preprint)
- Search by title on Google Scholar
- Check lab website publications page

## Related Skills

- `github` - For repository operations
- `web_search` - For discovery (if API available)

---

*Skill created: 2026-02-15 by Amy*
