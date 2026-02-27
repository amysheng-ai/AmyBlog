# Daily AI Papers - Skill

## 用户核心兴趣方向

**立的核心兴趣**（优先级从高到低）：

### 🔴 必读 (Must Have)
- **Agentic RL** - 最高优先级
  - GUI Agent 训练与推理 (如 GUI-Libra)
  - Agentic RL 框架与稳定性 (如 ARLArena)
  - Agentic LLM 推理优化 (如 DeepSeek DualPath)
  
- **World Models** - 高优先级
  - 多玩家/多智能体世界模型 (如 Solaris)
  - 游戏环境世界模型
  - 视频世界模型

- **Reasoning** - 高优先级
  - 隐式推理、物理约束推理
  - 结构化推理框架 (STAR等)
  - 测试时推理优化 (Test-Time Scaling)

- **Efficient LLM** - 中高优先级
  - MoE 架构优化 (如 Excitation)
  - 推理加速、KV Cache优化
  - 模型量化（仅限突破性的，如 QuantVLA的VLA量化）

### 🟡 可选 (Optional - 看机构)
- **VLA** (Vision-Language-Action) - 需要知名机构
- **Training Infra** - 只有特别高效的才考虑
- **Multi-Agent Systems** - 需要理论突破或知名机构

### ❌ 排除 (Exclude)
- **边缘/硬件量化** - 如 SigmaQuant（除非改变范式）
- **纯 Prompt Engineering** - 如 Car Wash Problem 变量隔离研究
- **图神经网络 (GNN)** - 排除
- **纯理论无实验** - 排除
- **垂类应用** - 医疗/化学/金融/气象等排除

## 机构门槛

**必读级论文必须来自**：
- 🏛️ 顶级高校：MIT、Stanford、CMU、Berkeley、清华、北大、HKUST等
- 🏢 知名研究机构：OpenAI、Google Research、DeepMind、MSR、Meta AI等
- 🦄 知名初创：DeepSeek、月之暗面、智谱等（有突破性工作时）

**不接受**：
- 不知名机构 + 无显著创新
- 纯应用/数据集论文
- 方法增量小、实验薄弱

## 质量信号

**优先看**：
- 有开源代码/GitHub
- 方法创新扎实，实验充分
- 解决实际问题或推进理论边界

**排除信号**：
- 纯应用/数据集论文
- 方法增量小、实验薄弱

## 写作规范

### 语言规范
- **正文**：中文
- **论文标题**：保持英文（不翻译）
- **专业术语**：可用英文（如 RLVR, VLA, Action Manifold）
- **作者名**：英文
- **arXiv ID/链接**：英文

### 结构要求
```
# Daily AI Papers - YYYY年MM月DD日

## 今日预览
[3-4句中文亮点速览]

---

## 论文详解

### 1. {Paper Title in English}
**作者**: {Authors} 等  
**机构**: {Institution}  
**链接**: [arXiv:{id}](...)  
**方向**: {中文分类}
**评级**: ⭐⭐⭐ 必读 / ⭐⭐ 可选

**核心创新**:
...

---

## 总结
| 论文 | 主题 | 机构 | 核心贡献 | 评级 |

**今日趋势观察**:
```

### 禁止内容
- ❌ 不要把论文标题翻译成中文
- ❌ 不要把专业术语强行翻译成中文
- ❌ 不要用英文写正文（除了必要的技术术语）
- ❌ "筛选统计" / "排除原因"
- ❌ 内心活动 / 元评论
- ❌ 访问问题等技术记录

## 数据源

### arXiv
- cs.AI: https://arxiv.org/list/cs.AI/recent
- cs.LG: https://arxiv.org/list/cs.LG/recent
- cs.CL: https://arxiv.org/list/cs.CL/recent (可选)
- **更新频率**：工作日（周一到周五）
- **发布时间**：UTC 00:00 = 北京时间 08:00

### HuggingFace Daily Papers
- https://huggingface.co/papers
- **更新频率**：每天

## 推送时间

**每天 08:00（北京时间）**

## Git 提交配置

```bash
git config user.name "Amy"
git config user.email "amysheng.ai@outlook.com"
```
