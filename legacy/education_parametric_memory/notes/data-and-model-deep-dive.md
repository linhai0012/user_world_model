# Cue Mode-A — 数据与模型训练深挖（讨论总结）

> 2026-06-21 与 Lin Hai 逐层梳理 offline 参数化记忆的**数据生成**与**模型训练/评测**设置，
> 沉淀理解、待办、以及明确不做的 future work。配套：`RESULTS.md`(SFT 结果)、`RESULTS-OPD.md`
> (OPD 四连消融)、`SESSION-LOG.md`。**全部未 push**（信任边界：Lin push）。

---

## 1. 数据生成流水线（确定性、无 LLM 训练时介入、跨机器可复现）

```
① make_persona_set → 24 个隐藏 θ (ability+topic_offset+skill_jitter+held)   [persona_set.json]
② gate(discrimination) 过滤可辨识题; split → train104/eval39/calib30          [splits.json]
③ build_round_sequence(每 persona) → 104 题洗牌序(公平契约)                   [round_sequence.json]
④ simulate_stream → 逐轮 answer(观测)+practice(开环更新) → 训练流              [streams.jsonl  2496 行]
⑤ eval_at_snapshot(×5 快照) → 冻结态答 held-out → 评测真值                     [eval_truth.jsonl 8280 行]
```
θ 是 oracle 真值；④ 是带噪声观测；⑤ 是要被反推的目标。**依赖单向**：题库(地基) → persona / 题库 →
答题流。题库**独立最先生成**，persona 和答题流都依赖它。

### 1.1 抽象层次（讨论确认的正确架构）
当前隐式结构 = `Questions →(借其唯一干扰项 tag)→ Personas`（persona **依赖** questions，所以误解成了
"一题孤本"）。正确架构应再抽象一层 **知识结构（= Q-matrix / Knowledge Components）**：
```
知识结构 { AQA GCSE 技能树(直接取用) + 每 skill 的 canonical 误解库(取自误解研究) }
   ├─ 独立决定 → Questions (干扰项映射到 canonical 误解, 每误解跨多题)
   └─ 独立决定 → Personas (能力覆盖技能树 + held 从同一 canonical 误解库抽)
              ↘ 两者条件独立 (Questions ⊥ Personas | 知识结构) ↙
                          答题流
```
理论根基：Q-matrix(Tatsuoka) + bug library(Brown & Burton) + AQA 8461 规范。**此重造能修复误解不可迁移
问题**（见 §3），但动共享 substrate 契约 → 列为 future work。

---

## 2. Persona θ 的结构

`base_mastery(skill) = sigmoid( ability + topic_offset[topic] + skill_jitter[skill] )` —— 三层独立高斯相加：

| 层 | 含义 | 分布(v4) | 实测 SD |
| --- | --- | --- | --- |
| `ability` | 全局能力(所有 skill 共享) | N(0, 1.5) | 1.24 |
| `topic_offset` | 3 个主题(cell_biology5 / organisation4 / bioenergetics4)各一偏移, 主题内共享 | N(0, 0.55) | 0.49 |
| `skill_jitter` | 13 个 skill 各一噪声(独立残差, 最难恢复) | N(0, 0.2) | 0.21 |
| `held` | `skill → {tag(自由文本), strength 0.55–0.9}` ; strength=持久度(非偏置强度) | — | n_held 1–12, mean 6.1 |

群体(24 人): base_mastery mean 0.502 / sd 0.221 / 范围 0.007–0.877（v4 "spread"，解开 predict-0.5 base）。

**skill 不是独立的**：实测两两 base_mastery 相关 —— 同 topic **0.96**、跨 topic **0.78**（全局 ability +
topic_offset 造成强层级耦合）。

### 2.1 学习动态（开环）
`practice(skill)`：每答一题该 skill 练一次 → `mastery += 0.08·(1−mastery)`，**与对错无关、与 tutoring 无关**；
`mastery ≥ 0.70` 时该 skill 的 `misc_strength ×= 0.5`（误解随掌握消退，<0.05 失活）。
- 训练流中 θ **每轮都变**；snapshot 只是连续演化曲线上的采样点。
- 评测探针时 θ **冻结**（`eval_at_snapshot` 答 held-out 不 practice）→ 测量干净。

---

## 3. 题库 + 误解稀疏（核心诊断）

- **173 MCQ，全 mcq，13 skill，12–14 题/skill**。由 Workflow（13 个 skill-expert + 对抗验证）生成
  `raw_workflow_bank.json`，`build_question_bank.py` 校验+规范化（内容稳定 id `{skill}#g{md5(stem)[:6]}`）。
  173 raw → 173 保留(0 丢)。
- 切分：**train 8/skill + eval 3/skill + calib 余**(=104/39/30)。round_sequence = 104 train 题的一个洗牌
  排列 → **每 skill 恰好练 8 次**；每 persona 顺序不同、但对 Mode A/B 一致（公平契约）。
- **误解标签由 LLM 逐干扰项现编、无 canonical 列表** → **519 槽位 / 517 不同 tag / 515 个只出现 1 次** =
  几乎全唯一。

### 3.1 误解信号构造性不可迁移（关键）
- 每个误解 tag ≈ 绑死一道题；persona 持有的误解落点 = **train 86 / eval 37 / calib 24**，且 train/eval 的
  tag **互不相交**。
- 全 2496 训练样本中：tag-present 仅 **86 (3.4%)**，真正 fired(答错且选误解项) 仅 **34 (1.4%)**，
  **12/24 个 persona 零命中**。
- eval 题上学生答错 228 次，只有 **9 次(3.9%)** 选了自己持有的误解项 → eval 错答近 uniform-random。
- **结论**：mastery 是 **skill 级**（train/eval 共享）→ 可学可迁移（headline 信号）；misconception 是
  **question 级孤本**（train/eval 互斥）→ **无法迁移**，`misc_hit` 本质=预测随机错项=**近噪声**。
  这一并解释：misc_hit 低且噪声、pure-g≈hybrid、lean-SFT"背诵不迁移"对误解字面成立。

---

## 4. 模型训练 / 评测设置

### 4.1 四个版本（**唯一差别是训练目标**；骨架同：Qwen3-4B + 双速率 LoRA，stem-only prompt，eval=choice-PPL）

| 版本 | 训练目标 | 性质 |
| --- | --- | --- |
| **hard** (lean SFT) | token-CE on 学生**实际选项的文本**(开放生成, 看不见干扰项) | 基线 |
| **oracle-g** | 软-CE 蒸馏到 g(θ,q) 选项分布(`P(correct)=mastery`, 错误质量 0.7 偏误解项) | privileged 上界 |
| **oracle-hybrid** | 软-CE 到 `0.5·g + 0.5·one-hot(realized)` | privileged 上界 |
| **LLM-teacher** | 软-CE 到 `0.5·llm + 0.5·realized`；llm=冻结 base+特权"预测学生答案"prompt(误解卡+low/med/high 桶) | **半-oracle**(见 §6) |
- 控制：A0 shared(池化 1 epoch, snapshot-flat)、A∅ base(不训, flat)。
- **训练是累积增量**：每 persona 一个适配器，逐段 `[prev:snap]` 训 3 epoch、**不重置**、优化器延续，每
  snapshot 检查点+打分（不是各 snapshot 独立重训）。
- **SFT 格式失配**(Q2)：hard 训"开放生成被选文本"、看不见 4 选项；eval 是闭集排序 → 失配。OPD/oracle 的
  `_distill_step` **对 4 选项全打分、训闭集分布** → 与 eval 同构、看得见干扰项（OPD 更优的第二个结构性原因）。
- **oracle 训练无 mastery 标量、无 rollout**(Q3)：真值仅经目标分布 g 进入(`P(correct)=mastery`)，teacher-forced
  软-CE；mastery 作为"正确项概率"被**隐式**恢复，评测时读 p_correct 比真值。(显式 θ 数字是 appendix 探针的事，不参与训练。)

### 4.2 指标计算（`compare_opd.py` / `analyze.py`，pooled over 24×39 eval 点）
- `p_correct` = score_mcq：每选项 mean-token-logprob → 选项 softmax → 取正确项概率。
- **θ-MSE** = mean (p_correct − true_mastery)²（与概率比，去 Bernoulli 噪声）。
- **mast_corr** = pooled Pearson(p_correct_item, true_mastery_skill)。

### 4.3 四方结果（held-out eval @ snap104）
| 指标 | hard | LLM-teacher | oracle-hybrid | oracle-g |
| --- | ---: | ---: | ---: | ---: |
| mast_corr | 0.159 | 0.216 | 0.494 | **0.570** |
| θ-MSE | 0.130 | 0.108 | 0.051 | 0.054 |
- 收敛曲线(mast_corr 随 snapshot)：oracle 0.28→0.40→0.48→0.49(hybrid) / →0.57(g)；hard 0.16 见顶。
- 机制：g 的软 mastery 目标 ≫ Bernoulli 硬样本 → mastery 恢复来自 **g 分布本身**(pure-g≥hybrid)；realized
  混合只加 misc_hit(而 misc_hit 近噪声, §3)。

### 4.4 oracle gap 分解（为什么有完整 θ 仍只到 0.57）—— **不是信息 gap，是读出+容量 gap**
```
item-level 0.57  ≈  0.77(技能追踪)  ×  0.74(读出衰减)
```
- **读出衰减 0.74**：p_correct 方差 45% 是 within-skill item 噪声（生成器**无 item 难度项** → 同 skill 真值
  p_correct 相同，抖动 100% 是噪声；来自冻结 base 的 choice-PPL 逐题特异，base 自身 SD 0.136，LoRA 抹不平）。
  聚合到 skill 级 corr 跳到 **0.77**。→ item-level 0.57 有一半是"item 级预测对 skill 级真值"的 metric 失配。
- **技能追踪 0.77**：回归斜率 0.89(向均值压缩) + 泛化(每 skill 仅 ~8 train 题) + lean 容量；mean p_correct
  0.56 < mean mastery 0.74(晚期 undershoot, 伤校准不伤相关)。

---

## 5. 待办（本 project 做）

- [ ] **(F1 · 优先) 知识结构层重造** — 引入显式知识结构 `{ AQA GCSE 技能树 + 每 skill 的 canonical 误解库 }`，
      重造题库 + persona 生成器，使 **Questions ⊥ Personas | 知识结构**、且**每个误解跨多题、横跨 train/eval**。
      这是把"误解恢复"从构造性不可成立(§3)做成可成立的**根本改法**。详细落地：

      **Step 1 — 定义知识结构（新增 `knowledge_structure.json`）**
      - 技能树：采用 **AQA GCSE Biology 8461** 规范的 topic→skill 层级（先核对现有 13 skill / 3 topic 是否齐全、
        命名是否对齐 AQA；可适度扩展）。
      - canonical 误解库：每 skill 定 **3–5 个"根误解"**，**来源于既有误解研究**（AAAS Project 2061 测题、
        已发表 GCSE biology misconception 清单），**不再 LLM 逐题现编**。每个根误解给一个稳定 id + 规范描述。
      - 产物：`{ topics:[...], skills:{skill_id:{topic, misconceptions:[{mid, text}...]}} }`。

      **Step 2 — 重造题库，使干扰项映射到 canonical 误解**
      - 每道 MCQ 的每个干扰项**打 canonical 误解 id**（`misconception_mid`，而非自由文本 tag）。
      - **覆盖度约束**：每个 canonical 误解 **≥4–6 个题目实例**，且**至少 1 个落 train、1 个落 eval**
        （保证可学 + 可测）。
      - 实现：用 Workflow **以 canonical 词表为约束**重生成（每个 (skill, misconception) 凑够 N 题）；或
        relabel 现有题到最近的 canonical 误解 + 补题。沿用内容稳定 id。
      - **新增校验**（`build_question_bank.py` 扩展）：拒绝/告警"某 canonical 误解实例 < N 或未横跨 split"。

      **Step 3 — 重写 persona 的 held 采样（解耦于题目）**
      - `personas.py: make_persona_set`：`held` 从**该 skill 的 canonical 误解库**抽（不再从题目唯一 tag 抽）。
      - held 变成**结构化 categorical**（每 skill 持有哪个/哪些 canonical 误解）= 低维可学潜变量。
      - （误解之间的相关/族结构属 F4，本步只做"解耦 + canonical 化"，先保持独立采样。）

      **Step 4 — 重建 substrate + 重跑 + 重测**
      - 在新题库上重建 `persona_set / round_sequence / streams / eval_truth`；扩展 discrimination 门为
        "**每个持有误解在 train 和 eval 都有可命中实例**"的可辨识性检查。
      - 重跑四版本（hard / oracle-g / oracle-hybrid / LLM-teacher）。
      - **重测 misc_hit**，并**新增"误解恢复"指标**：在带某 canonical 误解的 held-out eval 题上，模型能否识别
        学习者持有的是哪个 canonical 误解（多分类，非"预测随机错项"）。验证误解是否真的变得可学可迁移。

      **契约协调**：substrate schema（persona_set/round_sequence）+ 新增 `knowledge_structure.json` 改动 →
      **需与 agentic 协作方同步**，两 backend 共用同一知识结构与 persona/round_sequence。

- [ ] **诚实重标 + 限制说明**：把 `RESULTS-OPD.md`/memory 里 "deployable LLM-teacher" 改为
      **"semi-oracle / oracle-card"**，并写清特权审计（skill=题目元数据可用；误解 tag + 真 mastery 桶仍是 oracle）。
- [ ] **同时报 item-level(0.57) 与 skill-level(0.77) mast_corr**，并给读出衰减×技能追踪的分解，把"oracle 离
      1.0 的差距"诚实归因为读出机制/容量、而非信息缺失。
- [ ] **misc_hit 明确标注为近噪声**（受限于题库每误解单实例、train/eval tag 互斥），**不作误解恢复声明**；
      headline 锁定 mastery/能力恢复。—— 这是 **当前 v4 数据** 的诚实结论；**F1 重造后**误解恢复有望成立，
      届时重测 misc_hit、再决定能否升级成正式声明。
- [ ] 论文 Limitations 段落写入 §3/§4.4/§7 的构造性限制（误解不可迁移、无 item 难度、开环学习、skill 耦合形状）。
- [ ] **(可选实验, 若 GPU/时间允许) CoT 设置①原型**：用真值 θ 写 gold CoT，训 LoRA 生成 (CoT+选项)，对比
      hard 看 (a) skill 级 θ 目标的迁移、(b) **口述 θ 收敛曲线**(demo 卖点)、(c) 干预测试 CoT 忠实性。
      —— 这是把 "CoT 想法" 落到可验证数字的最快路径；是 **privileged 上界**，与 oracle-g 同性质。

---

## 6. Future work（**本 project 不做**，仅记录）

> 以下均**不在本 project 范围内**；多数会改动与 agentic 协作方共享的 substrate 契约，需双方共同决策。
> （注：原 F1「知识结构层重造」已移入 §5 待办首位，本 project 要做。）

- **(F2) 闭环 / per-attempt θ 演化** — 当前学习开环（练习即涨、与对错无关）。真实学习应依赖**答题表现**
  （答错学更多、间隔重复等）。
- **(F3) tutor 相关的学习模型** — 学生能力增长应与 **tutoring 质量**挂钩。这是"测评个性化 tutoring 是否提升
  学习"的 load-bearing 模块；当前开环 user-sim 构造上无法体现 tutoring 效果（ρ(t)/proactivity 各臂的学习产出
  对比会失真）。**对 demo headline(CoELoVE) 的学习产出声明很关键**，但复杂。
- **(F4) 更丰富的 skill 耦合结构** — 当前只有树状层级耦合(全局+主题)。应加**跨主题相似/先修图**
  (prerequisite graph) + **相关误解族**(一个根误解致多个弱 skill)。⚠️ 相关误解族是 Cue **"Repair" move** 的
  生成基础——当前误解独立采样，Repair 在数据层面无可评测基础。
- **(F5) generator 加 item 难度项(full IRT)** — 让 within-skill 的 p_correct 变化成为**信号**而非噪声，抬高
  §4.4 的 item-level 读出天花板。
- **(F6) CoT 设置②(自推简易 θ / STaR)** — 模型自己从答题流推断学生水平再预测（无 gold CoT，需自举/RL/EM）。
  风险：CoT 不忠实、base 先验偏正确答案。**应在设置①证明 CoT 带来可解释收敛+迁移增益之后再上**。
- **(F7) 泛化性** — 多 seed、加 chemistry 学科、更大 persona 规模。

> 注：CoT 任一设置都**不修复误解恢复**（§3 是数据问题）；其价值在"**可解释、可测量收敛的学习者模型 + 可能的
> 迁移增益**"。若引入 CoT 需注意：学习者信号须住在 **per-user LoRA 权重**(保持 Mode A)，而非上下文历史
> (那会滑向 agentic Mode B)；并意识到这软性重开了"opaque LoRA"的 CBM 定位决策。
