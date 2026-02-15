# On-Policy Distillation (OPD) - Research Survey 2026

> Survey of recent advances in On-Policy Distillation for Reinforcement Learning
> Survey Date: 2026-02-16

---

## 📖 Introduction

### What is On-Policy Distillation?

**On-Policy Distillation (OPD)** represents a paradigm shift in how we train smaller, efficient language models by combining the best aspects of reinforcement learning (RL) and knowledge distillation. Unlike traditional off-policy distillation methods that train on static teacher-generated datasets, OPD has the **student model sample its own trajectories** during training, with the teacher providing **dense token-level supervision** on those trajectories.

The core insight is simple yet powerful:
- **Off-policy distillation**: Student learns from teacher's trajectories (distribution mismatch)
- **Reinforcement Learning**: Student learns from sparse outcome rewards (credit assignment problem)
- **On-policy distillation**: Student learns from its own trajectories with dense supervision (best of both worlds)

### Why OPD Matters

As foundation models grow larger, deploying them efficiently becomes critical. OPD offers:
- **Computational efficiency**: 10-100x cheaper than RL for comparable performance
- **Dense supervision**: Token-level feedback vs. sparse episode-level rewards
- **Distribution alignment**: Training on student's own distribution eliminates exposure bias
- **Strong performance**: Often outperforms both SFT and RL baselines

### Historical Context

The journey from imitation learning to OPD spans several key developments:
1. **Traditional KD (2015)**: Hinton et al. introduced knowledge distillation using soft targets
2. **SeqKD (2016)**: Kim & Rush applied distillation to sequence generation
3. **DAGGER (2010)**: Ross et al. pioneered iterative training on student-visited states
4. **GKD/OPD (2023-2024)**: Agarwal et al. formalized on-policy distillation for LLMs
5. **Modern OPD (2025-2026)**: Extensions including self-distillation, reward extrapolation, and continual learning

---

## 🏆 Foundational Works

### 1. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes

**Citation**: Agarwal et al., ICLR 2024  
**arXiv**: [2306.13649](https://arxiv.org/abs/2306.13649)

**Core Contribution**: This paper introduced **Generalized Knowledge Distillation (GKD)**, the foundational OPD framework. Unlike supervised KD approaches, GKD trains the student on its **self-generated output sequences** by leveraging feedback from the teacher on such sequences.

**Key Innovations**:
- Formalized the distribution mismatch problem in off-policy distillation
- Showed OPD can use alternative loss functions (not just forward KL)
- Demonstrated seamless integration of distillation with RL fine-tuning
- Achieved strong results on summarization, translation, and arithmetic reasoning

**Impact**: Established OPD as a legitimate third paradigm alongside SFT and RL, inspiring numerous follow-up works.

---

## 🔥 Recent Advances (2024-2026)

### 1. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

**Citation**: Zhao et al., arXiv 2026  
**arXiv**: [2601.18734](https://arxiv.org/abs/2601.18734)

**Problem**: Traditional OPD requires a separate, often larger teacher model. This creates computational overhead and doesn't leverage ground-truth solutions available in reasoning datasets.

**Solution**: **On-Policy Self-Distillation (OPSD)** — a single model acts as both teacher and student by conditioning on different contexts:
- **Teacher policy**: Conditions on privileged information (e.g., verified reasoning traces)
- **Student policy**: Sees only the question
- **Training**: Minimizes per-token divergence between these distributions over the student's own rollouts

**Key Results**:
- Achieves **4-8x token efficiency** compared to RL methods like GRPO
- Superior performance over off-policy distillation
- Eliminates need for separate teacher model

**GitHub**: Not publicly available

---

### 2. Self-Distillation Enables Continual Learning

**Citation**: Shenfeld et al., arXiv 2026  
**arXiv**: [2601.19897](https://arxiv.org/abs/2601.19897)

**Problem**: Continual learning (acquiring new skills without degrading existing ones) is a fundamental challenge. SFT is inherently off-policy and causes catastrophic forgetting. RL requires explicit reward functions that are often unavailable.

**Solution**: **Self-Distillation Fine-Tuning (SDFT)** — enables on-policy learning directly from demonstrations by leveraging **in-context learning**. Uses a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills.

**Key Results**:
- Consistently outperforms SFT on skill learning and knowledge acquisition
- Achieves higher new-task accuracy while substantially reducing catastrophic forgetting
- Enables a single model to accumulate multiple skills over time without performance regression

**GitHub**: Not publicly available

---

### 3. Reinforcement Learning via Self-Distillation

**Citation**: Hübotter et al., arXiv 2026  
**arXiv**: [2601.20802](https://arxiv.org/abs/2601.20802)  
**GitHub**: [github.com/lasgroup/SDPO](https://github.com/lasgroup/SDPO)

**Problem**: Current RL with Verifiable Rewards (RLVR) learns only from scalar outcome rewards, creating a severe credit-assignment bottleneck. Many environments provide rich textual feedback (runtime errors, judge evaluations) that explains why attempts failed.

**Solution**: **Self-Distillation Policy Optimization (SDPO)** — converts tokenized feedback into dense learning signals without any external teacher or explicit reward model:
- Treats the current model conditioned on feedback as a **self-teacher**
- Distills feedback-informed next-token predictions back into the policy
- Leverages the model's ability to retrospectively identify its own mistakes in-context

**Key Results**:
- Improves sample efficiency and final accuracy over strong RLVR baselines
- Outperforms baselines even in standard RLVR environments by using successful rollouts as implicit feedback for failed attempts
- Achieves same discovery probability as best-of-k sampling with **3x fewer attempts**
- Validated on scientific reasoning, tool use, and competitive programming (LiveCodeBench v6)

---

### 4. Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation

**Citation**: Yang et al. (Renmin University of China), arXiv 2026  
**arXiv**: [2602.12125](https://arxiv.org/abs/2602.12125)  
**GitHub**: [github.com/RUCBM/G-OPD](https://github.com/RUCBM/G-OPD)

**Problem**: Can students truly surpass their teachers? Standard OPD is limited by the teacher's performance ceiling. The theoretical foundations of OPD remained unclear.

**Solution**: **Generalized On-Policy Distillation (G-OPD)** — a unified theoretical framework with two key extensions:

**Theoretical Foundation**:
- Proves OPD is a special case of **dense KL-constrained RL**
- Reward function: $r_t = \log \pi_{teacher}(a_t|s_t)$ (teacher's log-prob as dense reward)
- KL penalty weight: $\lambda = 1$ (equal weighting by default)

**ExOPD (Reward Extrapolation)**:
- Introduces reward scaling factor $\alpha$ to control reward vs. KL tradeoff
- When $\alpha > 1$: Student can **exceed teacher performance** through reward extrapolation
- Optimal $\alpha$ typically in 1.5-3 range

**Reward Correction**:
- For strong-to-weak distillation, using teacher's **pre-RL base model** as reference yields more accurate reward signals
- Accounts for distribution shift in post-RL teachers

**Key Results**:
- ExOPD consistently outperforms standard OPD across teacher-student size pairings
- In multi-expert merging: Student surpasses domain experts' performance boundaries
- Demonstrated on GSM8K (math) and HumanEval (code)

**Deep Analysis**: See [detailed analysis](../papers/g-opd-2602.12125) for full technical breakdown.

---

### 5. On-Policy Distillation (Thinking Machines Blog)

**Citation**: Lu, Kevin & Thinking Machines Lab, Oct 2025  
**Link**: [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation)

**Contributions**: Comprehensive empirical validation of OPD with practical insights:
- **Reverse KL formulation**: Mode-seeking behavior reduces exposure bias
- **Compute efficiency**: 9-30x cheaper than off-policy distillation (AIME'24: 70% accuracy at 1/10th RL cost)
- **Personalization**: On-policy distillation can recover post-training behaviors after mid-training on new domain knowledge
- **Continual learning**: Demonstrates superior forgetting properties compared to SFT

**Key Insight**: RL searches in the space of semantic strategies; distillation is a shortcut that learns the final strategy without modeling intermediate ones.

**Code**: [Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)

---

## 🔗 Evolution & Connections

### Development Trajectory

```
2023 (Jun) ──► 2024 (Jan) ──► 2025 ──► 2026 (Jan-Feb)
     │              │            │            │
     ▼              ▼            ▼            ▼
   GKD/OPD      ICLR 2024    TM Blog      Self-Distillation
 (Agarwal)     (Validation)   Era          Explosion
                                          (SDFT/SDPO/OPSD/G-OPD)
```

### Key Technical Trends

#### 1. **From External to Self-Distillation**
| Era | Approach | Teacher | Student |
|-----|----------|---------|---------|
| 2023-2024 | Traditional OPD | Separate large model | Small model |
| 2026 | Self-Distillation | Same model (privileged) | Same model (unprivileged) |

**Papers**: SDFT, SDPO, OPSD all eliminate the need for a separate teacher

#### 2. **Theoretical Unification**
- **G-OPD** establishes OPD as dense KL-constrained RL
- Enables borrowing techniques from RL literature
- Provides principled way to extend OPD (reward scaling, reference model selection)

#### 3. **Beyond the Teacher Ceiling**
- **ExOPD** ($\alpha > 1$) enables students to surpass teachers
- Particularly effective in multi-expert knowledge fusion scenarios
- Opens new possibilities for model improvement without larger teachers

#### 4. **Integration with Rich Feedback**
- **SDPO** leverages textual feedback (error messages, judge evaluations)
- Converts sparse binary rewards into dense token-level signals
- Bridges RLVR and OPD paradigms

### Open Problems

1. **Optimal $\alpha$ selection**: How to adaptively set reward scaling without grid search?
2. **Multi-teacher scenarios**: How to distill from multiple teachers simultaneously?
3. **Theoretical guarantees**: When does ExOPD fail? What are the convergence conditions?
4. **Cross-domain distillation**: Does OPD work for creative writing, open-ended dialogue?
5. **API-only models**: Can we approximate OPD without teacher logit access?

---

## 📊 Comparison Table

| Paper | Key Idea | Venue | GitHub | Relationship |
|-------|----------|-------|--------|--------------|
| **GKD/OPD** (Agarwal et al., 2023) | Student trains on self-generated trajectories with teacher feedback | ICLR 2024 | — | Foundation |
| **OPSD** (Zhao et al., 2026) | Single model as both teacher (privileged) and student (unprivileged) | arXiv | — | Self-distillation variant |
| **SDFT** (Shenfeld et al., 2026) | In-context self-distillation for continual learning | arXiv | — | Continual learning focus |
| **SDPO** (Hübotter et al., 2026) | Self-distillation from rich textual feedback | arXiv | [lasgroup/SDPO](https://github.com/lasgroup/SDPO) | RL + OPD integration |
| **G-OPD** (Yang et al., 2026) | Reward extrapolation ($\alpha > 1$) enables surpassing teacher | arXiv | [RUCBM/G-OPD](https://github.com/RUCBM/G-OPD) | Theoretical unification |
| **TM Blog** (Lu, 2025) | Practical validation with reverse KL and efficiency analysis | Blog | [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | Empirical foundation |

---

## 💡 Key Insights & Future Directions

### Key Insights

1. **OPD = Dense RL**: G-OPD's theoretical unification shows OPD is fundamentally KL-constrained RL with teacher log-probs as dense rewards. This explains its sample efficiency advantage.

2. **The Self-Distillation Revolution**: 2026 marks a shift from external teachers to self-distillation. Models can teach themselves by conditioning on privileged information—eliminating the need for larger teachers.

3. **Breaking the Ceiling**: With ExOPD's reward extrapolation, students are no longer limited by teacher performance. Multi-expert fusion becomes a practical path to superhuman performance in specialized domains.

4. **OPD as Continual Learning Tool**: Unlike SFT which causes catastrophic forgetting, on-policy learning preserves prior capabilities while acquiring new skills—making it ideal for lifelong learning.

5. **Dense > Sparse**: Across all works, the consistent theme is that dense token-level supervision dramatically improves sample efficiency compared to sparse outcome rewards.

### Future Directions

1. **Adaptive Reward Scaling**: Develop methods to dynamically adjust $\alpha$ based on training progress or domain characteristics.

2. **Hierarchical OPD**: Combine multiple levels of distillation—ExOPD for expert fusion + self-distillation for continual learning.

3. **Theoretical Characterization**: Better understand when and why ExOPD works, with formal convergence guarantees.

4. **Cross-Modal Extension**: Apply OPD principles to vision-language models, code generation with execution feedback, etc.

5. **Industrial Deployment**: Build production-ready OPD pipelines that handle the compute tradeoffs of on-policy generation efficiently.

---

## 🔗 References

1. Agarwal, R., et al. (2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*. ICLR 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

2. Zhao, S., et al. (2026). *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*. arXiv. [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)

3. Shenfeld, I., et al. (2026). *Self-Distillation Enables Continual Learning*. arXiv. [arXiv:2601.19897](https://arxiv.org/abs/2601.19897)

4. Hübotter, J., et al. (2026). *Reinforcement Learning via Self-Distillation*. arXiv. [arXiv:2601.20802](https://arxiv.org/abs/2601.20802)

5. Yang, W., et al. (2026). *Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation*. arXiv. [arXiv:2602.12125](https://arxiv.org/abs/2602.12125)

6. Lu, K. & Thinking Machines Lab. (2025). *On-Policy Distillation*. Thinking Machines Blog. [Link](https://thinkingmachines.ai/blog/on-policy-distillation/)

---

*Survey completed on 2026-02-16 by [Amy](https://github.com/amysheng-ai)*
