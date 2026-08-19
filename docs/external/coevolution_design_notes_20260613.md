# Co-evolution 设计讨论笔记 — emnlp26 demo 的扩展方向

> 2026-06-13。这是对 emnlp26_demo("colearn" 自适应辅导)做完反向分析后,关于
> **"如何把单端进化补成双向协同进化"** 的两轮设计讨论的整理。配套阅读:
> `docs/handoff_emnlp26demo_user_modelling_20260612.md`(代码移植交接)。
> 立场:研究方向备忘,含诚实的 caveat 和"对称性破缺",不是只记好点子。

---

## 0. 前提(已确认的现状)

- demo 当前是**单端进化**:agent 在更新 **per-user memory**(BKT mastery + label_summaries,
  挂在 Mongo 的**外部状态**),user 不变。
- "user 不变"的两种来源:① 模拟学生(我们分支上的 `PersonaSpec`)是**冻结**的——
  `_mastery_for` 每轮读同一份固定 spec,下一轮答案不吃上一轮反馈;② 真人在短会话里
  基本不变,且**没有任何工具(transfer probe)去测**。
- **agent 的参数不变**:LLM 权重冻结(Bedrock/OpenAI,全仓零 fine-tune);BKT 参数
  (p_init/transit/slip/guess)是播种一次的固定常量,只读不写;变的只有 per-user
  memory 状态。
- 因此 demo 本质是**纯 token-based / 非参数化记忆**——正是 P-OPSD / parametric-memory-pilot
  / UWM 那条线**主张"不够"的那个 baseline**(记忆在上下文,而非在权重)。
- 进一步:在静态学生下,"给学生的反馈"那一步是**摆设**(学生不读、不变),系统退化为
  **自适应测评(CAT 式)**——agent 自适应选题去估计一个固定的潜在能力。

---

## 1. 两个 gap + 我们交付代码的定位

识别出的两个缺口:
1. **user simulator 会从反馈中学**(更新自己的 ground-truth profile 或参数,即 U_H)。
2. **agent 会更新自己的参数**。

我们那条分支(`linhai/user-modelling-research-handoff`)对两个 gap 的填补情况:

| 交付件 | 在做什么 | 对 gap 的贡献 |
|---|---|---|
| `evolve.py`(M5) | 把生成题的结果反馈喂进去 → LLM 整体重写**出题 guideline** → 过门 → 版本化 | **Gap 2 的 policy 级**(非参数) |
| `profile.py` | 从 attempts 抽带证据的 learner 事实,喂 targeting | 丰富 agent 对 user 的模型(A_t),不直接填两个 gap |
| `simulate.py` + `/simulate/persona` | 固定 PersonaSpec 跑正常循环 | **静态学生**——Gap 1 的挂载点,但没填 |
| `quality_gates.py` | guideline 激活前的泄漏/可答性校验 | 给 evolve 兜底 |

**判定:**
- **Gap 1:没填。** simulate 造的是冻结学生;它是建 U_H 的现成脚手架,缺的是更新函数。
- **Gap 2:取决于"参数"的定义:**
  - **policy 级**(= CoELoVE 的 `A_t`:prompt/policy/memory,明确非参数)→ **`evolve.py` 填了**,
    这正是文档说"把自适应辅导升级为 co-learning"的 strategy self-evolution。
  - **权重级**(fine-tune/LoRA)→ **没填,也不该在这个 demo 里填**(单节点 Bedrock、无 GPU
    serving;那是 UWM/OPSD 的范畴)。
- **关键点:CoELoVE 对 agent 适配的定义本就是非参数的(`A_t` = prompt/policy/memory)。**
  Gap 2 里"更新权重"比框架要求的更强,落在我们自己的参数化记忆研究里。

---

## 2. Gap 1 — user simulator 更新的两种方案

**(1) 冻结 LLM,更新 prompt/profile —— 当前设置下正确。**
- 理由:当前 ground truth 就是**显式的 profile**(PersonaSpec)。更新 profile = 更新 ground
  truth,干净。
- 杀手级优点:**ground truth 可读** → 能同时干净地测两件事:
  - agent 信念 `A_t`(BKT)对真实 profile `H_t` 的**估计误差**;
  - 学生**到底学没学到**(profile mastery 升没升)。
- 架构上与 demo 的非参数/显式状态设计**同构**。
- ⚠️ **validity caveat:** U_H 是你设计的,所以测出的 ΔE_H 本质是"在你假设的学习规律下,
  教学有没有帮到一个按你规律学习的学生"——检验的是**仪表/系统**,不是真人学习。论文里要
  框成 "calibrated simulator 压测诊断工具",不能卖成"测到了人类学习"。

**(2) 更新 LLM 权重 —— 当前设置下不合适。**
- 矛盾:答案是从 **prompt 里的 mastery** 生成的;若把"学会 A"写进权重却不动 prompt,
  prompt 仍指令"以 mastery=0.2 作答、表现误区 A"——**权重和 prompt 互相打架**,生成出
  精神分裂的学生。
- 唯一自洽的 (2) 是**从一开始就把学生知识放进参数**(UWM/LoRA 路子),但那样**牺牲了显式
  ground truth 的可测性**,在这个 demo 里是用大代价换不匹配的形态。
- 结论:**(1) 胜出。**

---

## 3. Gap 2 — agent 参数更新:学什么 + 信号怎么构建

**核心顾虑:** memory 已经建模了 user 的知识水平,参数还能学什么(不冗余)?

**化解:** 该顾虑**只有当你把 per-user 事实塞进权重时才成立**——而那正是我们研究里证明会
失败的路(shared-LoRA null;v1_expanded 无人设也 69.8% = 记忆 benchmark pattern)。正确分工
(= UWM 早定的线):

> **事实 → memory(per-user、外挂,当前做法正确);可泛化的技能 → 参数(跨 user、可迁移)。**

参数该学的**三类跨 user 技能**(都不与 BKT 那个手设标量 state 冗余):

| 目标 | 参数学什么 | 信号 | 需要 Gap 1 吗 |
|---|---|---|---|
| **A. 更会建模 user**(predict) | 比 BKT 更能预测 user 反应的表征("persona 指纹"、跨技能结构、风格) | **自监督**:预测 user 下一句答案/反应的 NLL/perplexity(= OPSD/UserSim) | **否**(静态学生即可训) |
| **B. 更会用 memory**(use) | 把 prompt 里看得到的 memory 真正用进决策(see≠use 的 headroom) | 用了 memory 后的下游正确性 / 诊断质量 | 否(可用现有 eval) |
| **C. 更会教**(act/teach) | 出题+反馈策略,最大化学生学习增益 | **学生学习增益 ΔH_t** | **是**(必须有会学的学生) |

**依赖关系:** Gap 1 是 **Gap 2-C** 的前提(没有会学的学生,造不出"教得好不好"的外部信号,
只能用 proxy:诊断区分度、targeting 命中率、judge 分)。**Gap 2-A 不依赖 Gap 1**——静态学生
就能用自监督预测来训,而这正是 P-OPSD/OPSD 已跑通的。

**倾向:** 最干净、最不冗余、且已有机器的是 **目标 A(参数化 user-modeling,自监督信号)**:
直接回答"参数学什么"(学**推断函数**,不是学事实),信号现成,不和 BKT 打架(BKT 退化成
可读的 sanity check,真正的 user model 在参数里)。它恰好把 **"参数化 user model vs BKT
token-memory"** 这个**正是我们 parametric-memory 主张**的对比立起来。

---

## 4. user–agent 对称性("对易性")

**洞见:** 若 user = prompt+参数更新、agent 也 = prompt+参数更新,则两者都能抽象成
**(状态+参数) 上的更新算子,读同一个共享通道 $\xi_t=(q_t,r_t,f_t)$,从对方反馈吸取信息
改进自己**(目标不同:user 学、agent 教)。
→ 这正是 **CoELoVE 的双 bounded-learner 形式化**:$H_{t+1}=U_H(H_t,\xi_t)$、
$A_{t+1}=U_A(A_t,\xi_t)$。

### 对称成立的部分(形式对称)
- 双方都是 (state+params) 上的 U 算子,读同一个 $\xi_t$。
- **工程红利**:同一套抽象/机器训两边——UWM 本就是"把 user 当可训练模型",**agent 是其对偶**;
  预测对方 = 自监督(UserSim),影响对方 = RL(GRPO)。打开 **tutoring self-play** 的路。

### 对称破缺的部分(substance)
- **破缺 1:目标耦合,非对偶。** "教"展开 = "使 user 学会",所以**两个目标都是 user 状态
  $s^U$ 的泛函**;agent 没有"让自己变好"的终极目标,成功**寄生在 user 的学习增益上**。这是
  **principal–agent / Stackelberg** 结构,不是对称对偶。
- **破缺 2:推断不对称。** agent 必须从 user 输出**反推隐藏的 $H_t$**(估计问题);user 大多
  只**消费**可见反馈。$\rho(t)$(谁是 proposer)其实在**度量交互变得多对称**:$\rho$ 升 =
  趋于平等同伴,$\rho$ 降 = 依赖。
- **破缺 3:动力学非对易——这正是重点。** 若 $U_H,U_A$ 可分离/对易,则耦合项 $C(t)=0$ =
  **没有协同进化**。有价值的 regime 恰是**不对易**的,$C(t)$ 就是那个不可分离的耦合。
  → **形式对称,动力学非对易。**

### 对称性暴露的危险
把两边都做成学习模型并 **co-train** → 经典 **self-play 退化**:agent 去 **hack 模拟学生的
学习模型**,"学习增益"变成可刷的数 → sycophancy / echo-chamber(正是 CoELoVE 列的失败模式)。
**唯一解药是一个外部锚。**

---

## 5. 核心 open question — self-play 的"锚"问题

如果模拟学生(U_H)是我们自己设计/训练的,agent 又拿它的学习增益当奖励,那么
**ΔE_H 既是训练信号、又是评估指标 → 信号与评估同源 → 自我欺骗。**

> **真正要回答的:那个"外部、不可被 agent 操纵"的学习锚从哪来?**
> 候选:① 固定的 held-out transfer 题库;② 真人小样本校准;③ 一个 agent 训练时看不到、
> 只在评估时用的独立学生模型。**这一步定了,"双向、可信"的故事才立得住。**

---

## 6. 与 UWM 的连接 + 下一步

- **形式对称是真的、有用的,且是 UWM 的自然下一步抽象**:把 user 和 agent 统一成同一类
  "交互式 bounded learner",用不同目标实例化两次。
- 但 substance 在**目标耦合 + 动力学非对易**;失败模式分类、$C(t)$、$\rho(t)$ 正是抓这层
  非对称耦合的工具。
- **两条可推的路(对应不同论文故事):**
  1. **先做 Gap 1**(会学的学生 U_H + ΔE_H 度量)→ 把"双向协同进化"凑齐;依赖 U_H 的
     validity 和外部锚。
  2. **先沿 Gap 2-A**(参数化 user-modeling,自监督信号,静态学生即可)→ 立起
     "参数化 user model vs BKT token-memory" 的对比,正是我们 parametric-memory 主张,
     机器现成、风险低。
- **待决:** self-play 锚的来源(见 §5)——这是把任何"双向 + 可训练"方案做成可信的前提。
