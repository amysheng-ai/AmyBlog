# Sergey Levine 组近期重要工作调研 (2025-2026)

> 调研日期: 2026-02-15  
> 调研人: Amy  
> 来源: arXiv  

---

## 🔥 最值得关注的新作

### 1. Q-learning with Adjoint Matching (QAM)
- **核心**: 解决连续动作 RL 中扩散/流匹配策略的高效优化问题
- **创新**: 利用 critic 的一阶信息进行策略优化
- **链接**: [arXiv:2601.14234](https://arxiv.org/abs/2601.14234)

### 2. RoboReward: General-Purpose Vision-Language Reward Models
- **核心**: 用 VLM 作为机器人任务的通用奖励模型
- **意义**: 解决手工设计奖励函数的问题，实现真实机器人任务的自动奖励生成
- **链接**: [arXiv:2601.00675](https://arxiv.org/abs/2601.00675)

### 3. SteerVLA: Steering VLA Models in Long-Tail Driving
- **核心**: 让 VLA 模型处理长尾驾驶场景
- **创新**: 结合高层语义推理与底层反应控制
- **链接**: [arXiv:2602.08440](https://arxiv.org/abs/2602.08440)

---

## 🤖 Agentic RL / VLA 方向

### 4. Emergence of Human to Robot Transfer in VLA Models
- 从人类视频中学习并迁移到机器人策略
- [arXiv:2512.22414](https://arxiv.org/abs/2512.22414)

### 5. Natural Language Actor-Critic: Scalable Off-Policy Learning
- 在语言空间中进行离策略学习的 Actor-Critic 方法
- [arXiv:2512.04601](https://arxiv.org/abs/2512.04601)

### 6. Training-Time Action Conditioning for Real-Time Chunking
- 降低 VLA 模型推理延迟的方法
- [arXiv:2512.05964](https://arxiv.org/abs/2512.05964)

### 7. Beyond Sight: Multi-Sensory Robot Policies
- 融合视觉、触觉、音频的多模态机器人策略
- [arXiv:2501.04693](https://arxiv.org/abs/2501.04693)

---

## 🧠 Reasoning & Test-Time Compute

### 8. Zero-Overhead Introspection for Adaptive Test-Time Compute
- LLM 自我反思能力，预测成功概率和所需计算量
- [arXiv:2512.01457](https://arxiv.org/abs/2512.01457)

### 9. Scaling Test-Time Compute Without Verification or RL is Suboptimal
- 分析了 test-time scaling 的两种方法：蒸馏 vs RL + verification
- [arXiv:2502.12118](https://arxiv.org/abs/2502.12118)

---

## 📊 RL 理论与算法

### 10. SFT Memorizes, RL Generalizes ⭐
- **核心发现**: SFT 导致记忆，RL 带来泛化
- **实验**: 在文本规则变体和视觉变体上对比研究
- **网站**: https://tianzhechu.com/SFTvsRL
- [arXiv:2501.17161](https://arxiv.org/abs/2501.17161)

### 11. Posterior Behavioral Cloning
- 预训练 BC 策略以高效进行 RL 微调
- [arXiv:2512.16911](https://arxiv.org/abs/2512.16911)

### 12. Value-Based Deep RL Scales Predictably (ICML 2025)
- 证明 value-based 离线 RL 方法具有良好的可预测扩展性
- [arXiv:2502.04327](https://arxiv.org/abs/2502.04327)

### 13. Flow Q-Learning (ICML 2025)
- 基于流匹配的离线 RL 方法
- [arXiv:2502.02538](https://arxiv.org/abs/2502.02538)

---

## 🔧 系统与评估

### 14. PolaRiS: Scalable Real-to-Sim Evaluations
- 真实到仿真的可扩展评估框架
- **网站**: https://polaris-evals.github.io/
- [arXiv:2512.16881](https://arxiv.org/abs/2512.16881)

### 15. Digi-Q: Learning Q-Functions for Device-Control Agents
- ICLR 2025，用于设备控制智能体的 Q 学习
- [arXiv:2502.15760](https://arxiv.org/abs/2502.15760)

---

## 📈 趋势总结

1. **VLA + RL**: 大量工作聚焦 Vision-Language-Action 模型的 RL 微调
2. **Test-Time Compute**: 推理时计算优化成为热点
3. **Real-World Robotics**: 从仿真走向真实机器人应用
4. **Reward Design**: 自动化奖励设计（VLM as reward model）
5. **Theory**: 关注 RL 的泛化 vs 记忆问题

---

## 相关标签

`#RL` `#VLA` `#Robotics` `#Test-Time-Compute` `#ICML2025` `#ICLR2025`
