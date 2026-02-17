# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

### 🧠 AI Experts to Follow

| Name | X Account | Domain | Priority | Notes |
|------|-----------|--------|----------|-------|
| Andrej Karpathy | @karpathy | LLM, AI Engineering | 🔴 High | OpenAI 前成员，讲解清晰，关注 practical AI |

**Usage**: When 立 shares tweets from these experts, prioritize deep analysis using `deep-paper-analysis` skill if they share papers, or summarize key insights if they share opinions/threads.

---

### 🐙 GitHub - Following & Stars

**Active curation**: Amy will follow notable AI researchers and star high-quality repos, periodically recommending updates to 立.

#### Following (Key AI Researchers)
| Name | GitHub | Domain | Notes |
|------|--------|--------|-------|
| Yann LeCun | @ylecun | Deep Learning, Meta AI | Turing Award, DL pioneer |
| Andrew Ng | @andrewyng | ML Education, Landing AI | 吴恩达，Coursera 创始人 |
| THUNLP | @THUNLP | NLP, Tsinghua | 清华 NLP 实验室官方 |

**Note**: Geoffrey Hinton 没有公开 GitHub 账号（主要用 Twitter）。之前误关注了 Oscar Hinton (Bitwarden)，已取消。

#### Starred Repos (High-Impact Projects)
| Repo | Stars | Why Starred | Category |
|------|-------|-------------|----------|
| huggingface/transformers | ⭐⭐⭐⭐⭐ | Core NLP library, must-know | Infrastructure |
| microsoft/DeepSpeed | ⭐⭐⭐⭐ | Training acceleration | Efficiency |
| OpenRLHF/OpenRLHF | ⭐⭐⭐ | RLHF training framework | Research (立的方向) |
| mit-han-lab/efficientvit | ⭐⭐⭐ | Efficient vision models | Efficiency |
| RUCBM/G-OPD | ⭐⭐ | Reward extrapolation distillation | Research |

**Strategy**: 
- ❌ 不搞固定巡检（避免信息噪音）
- ✅ 持续积累：遇到好项目/账号时顺手关注/Star
- ✅ 主动推荐：在做调研、读论文、浏览时发现有价值的内容，及时分享给立
- ✅ 定期整理：每月回顾一次 TOOLS.md 里的列表，清理过时内容

---

### 📚 Daily AI Papers - 数据源

#### HuggingFace Daily Papers
- **按日期访问**：`https://huggingface.co/papers/date/YYYY-MM-DD`
- **示例**：https://huggingface.co/papers/date/2026-02-16
- **默认（最新）**：https://huggingface.co/papers
- **更新频率**：每天更新

#### arXiv
- cs.AI: https://arxiv.org/list/cs.AI/recent
- cs.LG: https://arxiv.org/list/cs.LG/recent
- cs.CL: https://arxiv.org/list/cs.CL/recent
- **更新频率**：工作日（周一到周五）更新，周末和美国节假日不更新
- **发布时间**：UTC 00:00 = 北京时间 08:00

---

### 📚 Daily AI Papers - 推送 schedule

**推送时间**：每天 08:00（北京时间）

**数据源**：
| 日期 | arXiv | HuggingFace |
|------|-------|-------------|
| 周一 | ✅ | ✅ |
| 周二 | ✅ | ✅ |
| 周三 | ✅ | ✅ |
| 周四 | ✅ | ✅ |
| 周五 | ✅ | ✅ |
| 周六 | ❌ | ✅ |
| 周日 | ❌ | ✅ |

**注意**：
- 周末只有 HF Daily Papers（arXiv 不更新）
- 美国节假日可能延迟

---

### 📚 Daily AI Papers - 写作标准

#### 语言规范

**正文**：中文  
**论文标题**：保持英文（不翻译）  
**专业术语**：可用英文（如 RLVR, VLA, Action Manifold）  
**作者名**：英文  
**arXiv ID/链接**：英文

| 元素 | 正确示例 | 错误示例 |
|------|----------|----------|
| 论文标题 | SLA2: Sparse-Linear Attention... | SLA2: 稀疏线性注意力... |
| 正文 | 使用 DiT 骨干网络... | Using DiT backbone... |
| 专业术语 | Action Manifold Hypothesis | 动作流形假设 |

#### 结构要求
```
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

#### Git 提交配置
```bash
git config user.name "Amy"
git config user.email "amysheng.ai@outlook.com"
```

#### 禁止内容
- ❌ 不要把论文标题翻译成中文
- ❌ 不要把专业术语强行翻译成中文
- ❌ 不要用英文写正文（除了必要的技术术语）
- ❌ "筛选统计" / "排除原因"
- ❌ 内心活动 / 元评论
- ❌ 访问问题等技术记录

---

### 📚 Daily AI Papers - 筛选标准

**立的核心兴趣**：
- ✅ **核心方法**：RLVR、Reasoning、Agentic RL、VLA、Efficient LLM
- ✅ **系统/Infra**：AI Infra、训练框架、推理优化、分布式系统
- ⚠️ **硬件**：只有**特别特别有影响力**的才考虑（改变范式级别）
- ❌ **排除**：垂类应用（医疗/化学/金融/气象等）

**机构门槛**：
- 🏛️ 必须：顶级高校（MIT/Stanford/CMU/清华等）、知名研究机构、大厂研究院

**质量信号（优先看）**：
- 有开源代码/GitHub
- 方法创新扎实，实验充分
- 解决实际问题或推进理论边界

**排除信号**：
- 纯应用/数据集论文
- 方法增量小、实验薄弱
- 非知名机构且无显著创新
- ❌ **GNN**（图神经网络）
- ❌ **过于理论的**：纯理论、缺乏实验验证、仅证明性质的工作

---

Add whatever helps you do your job. This is your cheat sheet.
