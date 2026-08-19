# Project Status — user_world_model(general/PersonaMem 轨道快照, 截至 2026-05-29)


> **历史快照,路径已过时。** 2026-08-09 仓库按 domain 重组:`common/` 只剩跨域基础设施,
> 各域代码在 `domains/{general,health,education}/`,入口脚本在 `scripts/<domain>/`。

> ⚠️ **历史快照。** 2026-06-03 仓库已重组为 all-purpose 框架(见
> [`project_summary.md`](project_summary.md));本文描述的 PersonaMem 原型代码现位于
> [`legacy/general_personamem/`](legacy/general_personamem/)。本文保留为 general 轨道的
> 现状综述与经验记录,路径按重组前书写。
>
> 重启快照,基于通读 `legacy/general_personamem/EXPERIMENTS.md`(§1–§13)、文档、
> `teacher_sft/` + `student_opd/` + `data_prep/` 代码、及 `outputs/` 383 个结果文件。

---

## 1. 项目是什么

用 **on-policy distillation** 训练「每用户模拟器」(per-user UserSim)。核心命题:
把一个特定用户的全部偏好压进一个 **per-user LoRA** 的权重里,推理时**不给任何聊天
历史**,只给一张人物卡(demographics)。

**部署范式**:Agent 对一个 query 生成 N 个候选回复 → UserSim 用「下一句用户话」的
perplexity 给每个候选打分 → Agent 取 `argmin PPL`。

- **数据**:PersonaMem-v1,20 个 persona × 三种上下文版本(32k / 128k / 1M)。
  跨版本测试把「persona 知识」与「具体事件记忆」分离。
  - 32k → 589 MCQ,128k → 2727 MCQ,1M → 2674 MCQ;K=3 prefix 上下文上限。
- **基座**:`Qwen/Qwen3-4B-Instruct-2507`(262k 原生上下文,无 thinking 模式)。
- **姊妹项目**:[P-OPSD](https://github.com/linhai0012/P-OPSD)(agent-modeling 轨道,
  PersonaMem-v2),已独立开发。本 repo 是 **user-modeling 轨道**。

---

## 2. 仓库结构(快照时布局;现已整体移至 `legacy/general_personamem/`)

> 下列目录在 2026-06-03 重组后位于 `legacy/general_personamem/` 下;根目录的活跃工作见 `CONVENTIONS.md`。

```
data_prep/      PersonaMem 加载、episode 切分、K-session 窗口、SFT tokenize (6 .py)
teacher_sft/    教师 SFT(Instruct-2507, K=3, user-only loss)+ 5 套评估范式 (11 .py)
student_opd/    per-user dual-LoRA 学生 + OPD/OPSD 训练 + 全套 eval/judge/报表 (28 文件)
outputs/        383 结果 JSON + 8 jsonl + 6 评审 xlsx(已提交参考结果)
EXPERIMENTS.md  全实验日志(Phase 0–2b + OPSD, §1–§13, 115KB)
phase2b_experiment_plan.md   R1→2c 的 dual-LoRA + gated-KL 设计
verbal_eval_summary.md       verbal-feedback(范式 II)三次失败复盘
mcq_examples.{md,xlsx}        per-qtype MCQ 样本
```
代码 ~10.7k 行 / 40 个 .py。工作树干净,`main` 与 `origin/main` 同步。

---

## 3. 实验进展时间线

| 阶段 | 内容 | 关键结果 |
|---|---|---|
| **P1 / R1 教师** | Qwen3-4B-hybrid, K=20, user-only SFT | NLL −1.08 nats,但 **MCQ-PPL 打平**(RoPE 外推 + thinking 污染) |
| **P1 / R3 教师**(生产) | 换 **Instruct-2507**, K=3, 262k 上下文 | **MCQ-PPL +14.6pp**(0.345→0.491);swap-persona 证明真学到 persona(+0.008 nats, 35/40 符号一致, p<1e-6);强于 track_evolution(+35pp)/suggest_new(+28pp),弱于 generalize(−26pp) |
| **P2 / 单 LoRA OPD** | rank-32 单 LoRA, reverse-KL, 4 焦点 persona | best-step closure:Lisa 84% / Jordan 92% / Kanoa 50% / Leilani 61% → 宏平均 **76% closure, 零上下文**。三指标(NLL/Judge/MCQ)互相背离 |
| **P2b / R1**(失败) | dual-LoRA + 2-turn 上下文, slow_lr=1e-5 | closure 崩到 26%(2-turn 上下文有毒 + slow_lr 太低 MLP 不动) |
| **P2b / R1b** ★最佳配方 | dual `s32f16` + **demo-only 上下文** + **slow_lr=5e-5** + ungated reverse-KL | 4 persona 恢复 **+78% closure** 且更稳(波动 12pp vs R1 的 27pp);**全 20 persona:+95.7% best-step closure**,8/20 超 teacher_k3,18/20 终点净正,仅 Leilani 单调衰减 |
| **P2b / R2a**(失败) | entropy-gated KL | gate 第 30 步崩到 ~7%(Instruct 基座过度自信),反而**加速** Leilani 衰减 |
| **P2b / R2c** | joint-gate `(H_t<τ) ∧ argmax 分歧` | +47% closure,稳定;但无法区分「教师自信且对」vs「教师自信但错」(§11.7 根本极限) |
| **跨版本泛化** | 128k 训练 → 1M 测试 | **+128% closure**,4/4 persona 超 teacher_k3 → LoRA 学的是 persona「指纹」非具体事件 |
| **§13 OPSD** | GT 当作教师 prefix 里「已完成的上一句用户话」 | 收敛快 2–3×;整体 verbal-judge 1.81 < R1b 2.00,但判别型 qtype 碾压(track_evolution **1.00**, generalize 0.82 vs R1b 0.18)。**per-qtype oracle max(R1b,OPSD)≈0.59 vs R1b 单独 0.48 → +11pp 集成空间** |

**头条结果**(README):20-persona / 128k MCQ 上,Base 30.6% → **R1b 学生 38.8%(零上下文)**
→ Teacher_k3(带 K=3 历史)39.8%。学生几乎追平带历史的教师。

---

## 4. 三种评估范式状态

- **范式 I — NLL/PPL 打分**:✅ 唯一**跑通**的端到端主线,数字干净(54–95.7% closure),有清晰 ablation。
- **范式 II — Verbal-feedback Agent**:❌ R1 两次尝试均失败,**结构性反相关**(SFT 把最投入的反应给了错选项),非 prompt 可修。已搁置。唯一相邻成功:相似度 judge(student 在 3/4 persona 上超 teacher_demo)。
- **范式 III — 直接问(zero-shot MCQ)**:❌ LoRA 破坏指令跟随(parse 失败 5.4%→48%),不可行,已放弃。

---

## 5. 当前状态 & 剩余储备

**刚完成 / 已验证**:R3 生产教师(§9.6)、R1b 全 20-persona 通用配方(§11.11)、
OPSD sanity-check + benchmark(§13)。
**进行中**:full-param SFT 基线(§13.7)—— 判断「是 LoRA 容量瓶颈还是任务本身在 4B 规模就这么难」。

**最大的未动用储备**(§11.9–11.11):
1. **CE-on-GT 混合损失** —— 救 Leilani 的 `acknowledge_latest` 结构性失败(teacher 0.10 < base 0.20)。
2. **per-qtype loss 加权** / **per-persona 自适应 gating**(cherry-pick 每轮最佳 +94% vs 单轮 +78%)。
3. **Teacher K=10**(只改 `build_opd_data.py` 一个常数)。
4. **minimal-demographics ablation** —— 当前富人设抬高 closure 分母,paper baseline 对比需要它。

---

## 6. 关键缺口(重启首要任务)

> ⚠️ **缺少明确的 token-based memory baseline。** 目前所有 closure% 都是相对
> `teacher_k3`(带 K=3 历史的同一基座)度量的,**没有一个「把记忆放进上下文」的
> 标准外部方法**作为对照。因此「per-user LoRA 这套到底相对 SOTA 记忆方法强不强」
> 这个问题**当前无法回答**。重启第一步:固定 token-based memory baseline。

> **注(2026-06-03)**:该 token-based memory baseline 缺口现由根目录 `baselines/`
> (oracle / trivial / token-memory)承接,归入 `project_summary.md` §8.2 的消融骨架
> (`base` vs `+profile` vs `+memory` vs `+per-user weights`)。上述 PersonaMem 代码与
> 产物已随重组移入 `legacy/general_personamem/`。

---

## 7. 环境与规模

- 集群:Isambard-AI Phase 2(ARM aarch64, GH200, Slurm, 24h walltime)/ KCL CREATE(A100/H100/H200/B200)。
- LoRA 规模:dual `s32f16` = slow(MLP gate+up, r=32, α=64)+ fast(Attn q/k/v/o, r=16, α=32),
  合计 ~68M ≈ 4B 的 ~1.7%。
- 训练:1 epoch,每 200 步存档;4 persona 各占 1 GPU 并行(非 DDP);K=3 约 50s/step。
- reverse-KL 约定:`KL(student‖teacher) = Σ_v P_s(v)·(log P_s − log P_t)`(Thinking Machines
  on-policy distillation 约定,**勿**用 `F.kl_div(student_lp, teacher_p)` 写反方向)。
