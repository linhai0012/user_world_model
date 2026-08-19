# DRIVE-Health CDT PhD Project Description — Draft

> **Internal draft, 2026-08-17 (rev. 2)** — for the DRIVE-Health CDT call (2027 entry), Theme 3:
> Complex Simulations and Digital Twins. Prepared for Yulan's review; deliberately kept at
> a public-safe level of detail (no architecture or method specifics). Sections mirror the
> CDT template exactly — copy each block into the corresponding field of the Google Doc.

---

## Project Title

**Virtual Patients as Digital Twins: Simulating Individual Health Behaviour to Develop and Evaluate AI-Supported Care**

---

## Background

Patient-facing AI systems, including conversational health assistants, digital coaching, and behaviour-change support, are moving rapidly from research into practice, increasingly powered by large language models (LLMs). Their development faces a fundamental bottleneck: such systems learn from, and must be evaluated through, interaction with people, yet real patient interaction is slow, costly, and ethically constrained, and sensitive health data cannot be freely shared. This is precisely the setting in which simulation helps. A *virtual patient* is a generative model that simulates how an individual responds to a health-support action, capturing both what the person says and how their health-related state, such as mood, fatigue, sleep quality, and engagement, evolves. Such a model allows patient-facing AI to be trained, stress-tested, and compared safely and at scale before deployment, and supports "what-if" exploration of intervention strategies that could not be trialled directly.

Digital twins are attracting growing attention in healthcare [1], and the simulation of human behaviour with LLMs has advanced on two fronts. Generative agents have shown that LLM-driven characters can reproduce strikingly believable social behaviour [2], and a growing line of work builds LLM-based user world models, which predict how a specific user will react to a system's next action so that candidate strategies can be simulated and assessed in advance [3,4]. In parallel, LLMs have been shown to interpret wearable-sensor data and to deliver personalised sleep and fitness coaching grounded in it [5,6]. The open challenge is the step from *plausible* to *faithful* simulation: virtual patients that track a specific individual over time, remain consistent with that individual's history, and whose fidelity can be rigorously validated against strong baselines. Without such validation, health AI developed in simulation cannot be trusted. This project addresses that gap, developing methods that are grounded in healthcare and designed to carry over to other settings where AI systems interact with individuals over time, such as education.

**Key references**

1. Katsoulakis et al. (2024). Digital twins for health: a scoping review. *npj Digital Medicine* 7:77.
2. Park et al. (2023). Generative agents: interactive simulacra of human behavior. *UIST*.
3. He et al. (2025). Simulating before planning: constructing intrinsic user world model for user-tailored dialogue policy planning. *SIGIR*.
4. Wang et al. (2025). User behavior simulation with large language model-based agents. *ACM Transactions on Information Systems* 43(2).
5. Kim et al. (2024). Health-LLM: large language models for health prediction via wearable sensor data. *CHIL*.
6. Khasentino et al. (2025). A personal health large language model for sleep and fitness coaching. *Nature Medicine* 31.

---

## Aims and Objectives

**Aim:** to develop and validate methods for building LLM-based virtual patients, which serve as individual-level digital twins of health behaviour, and to use them as simulation environments for developing and evaluating patient-facing AI.

**Objectives:**

1. **Construction.** Develop methods for building virtual patients from heterogeneous longitudinal data, including user profiles, interaction history, wearable-derived signals, and self-reports, producing both natural-language responses and structured trajectories of health-related state.
2. **Validation.** Establish an evaluation framework for individual-level fidelity: predictive accuracy against strong baselines, temporal consistency, calibration, and controls that distinguish genuine personalisation from population-level plausibility.
3. **Simulation for AI development.** Use validated virtual patients as simulated environments in which AI health-support agents can be developed, stress-tested, and compared, providing systems simulation and predictive analytics over simulated patient populations.
4. **Counterfactual analysis.** Investigate "what-if" simulation of intervention strategies (e.g., the timing, framing, and intensity of digital health interventions), drawing on causal-inference methods and carefully characterising the limits of causal conclusions drawn from simulated environments.

---

## Planned research methods, skills required, and additional training provided

**Planned research methods.** Large-language-model adaptation (fine-tuning and parameter-efficient methods); retrieval and long-horizon memory mechanisms; joint modelling of text and time-series data; structured prediction of user state; simulation-based evaluation design; causal inference; and human evaluation studies. Experiments will use public and synthetic datasets on King's HPC facilities (CREATE), and the resulting methods are expected to find application both in digital-health research at King's and in neighbouring domains such as education, where AI tutors interact with learners over time.

**Skills required.** A strong quantitative background (computer science, AI/ML, mathematics, engineering, or a related discipline); solid Python programming; and familiarity with machine learning fundamentals. Prior NLP/LLM or health-data experience is desirable but not essential.

**Additional training provided.** DRIVE-Health CDT cohort training in data-driven health; training in LLM/NLP research methods within the King's NLP group; large-scale experimentation on HPC; responsible AI, research ethics, and health-data governance; and opportunities to engage with King's digital-health research programmes and their clinical and industry collaborators.

---

## Project summary for the CDT website

### Project background

Imagine being able to test a new AI health coach on a thousand realistic patients, each with their own personality, routines, history, and day-to-day ups and downs, before it ever speaks to a real person. AI systems that support people in managing their health, from wellbeing coaching to long-term condition self-management, are advancing rapidly on the back of large language models. But building them responsibly runs into a hard constraint: every design choice ideally needs to be evaluated through interaction with people, and real patient interaction is slow, expensive, and ethically sensitive, while the health data such systems depend on is rightly protected.

Simulation offers a way through. A *virtual patient* is a generative model that plays the role of a specific individual: it responds the way that person would, both in what they say and in how their health-related state (mood, sleep, fatigue, engagement) changes over time in response to advice, interventions, and everyday life. With faithful virtual patients, researchers can train and stress-test patient-facing AI safely and at scale, compare design alternatives on the same simulated population, and explore what-if questions that could never be trialled directly: what if the coaching message had arrived in the evening, or the goal had been smaller?

Today's language models are strikingly good at simulating *plausible* people, but plausible is not faithful: a simulator that produces a generic "typical user" tells you little about how a particular individual, with their particular history, will actually respond. Making virtual patients faithful to individuals, and knowing how to measure that faithfulness, is the frontier this project tackles. The project is grounded in health, but the same simulation-and-validation toolkit applies wherever AI systems interact with people over time, from AI-supported care to AI tutoring in education.

### Project aims and objectives

The aim of this PhD is to develop and validate methods for building LLM-based virtual patients that act as individual-level digital twins of health behaviour, and to use them as simulation environments for developing patient-facing AI. The project is organised around four research questions:

- **RQ1 — Construction.** How can a virtual patient be built from the heterogeneous, longitudinal data an individual generates, from conversation history and self-reports to wearable-derived signals, so that it captures both what the person says and how their state evolves?
- **RQ2 — Validation.** How do we measure whether a virtual patient is faithful to its individual? What evaluation protocols distinguish genuine individual-level fidelity from population-level plausibility, and what baselines must a simulator beat before it can be trusted?
- **RQ3 — Simulation for AI development.** What does a validated simulator buy us? Can AI health-support agents developed and evaluated against simulated patient populations be shown to transfer to real-world criteria?
- **RQ4 — Counterfactual analysis.** Can virtual patients support principled "what-if" exploration of intervention strategies, and where are the limits of causal conclusions drawn from simulation?

Expected outcomes include new methods for individual-level behaviour simulation, an open evaluation framework and benchmark suite for virtual-patient fidelity, and openly released synthetic patient cohorts that let the wider community build on this work without access to sensitive data.

### Suitable background

This project would suit a graduate in computer science, artificial intelligence, mathematics, engineering, or a related quantitative discipline. Solid Python programming and a good grounding in machine learning fundamentals are expected. Experience with deep-learning frameworks, NLP/LLMs, time-series data, or statistics is an advantage but not a requirement, and no clinical background is needed. The ideal candidate is curious about how people behave, careful about evaluation, and motivated by the idea of making health AI safer to build.

### Skills and experience provided

The student will graduate with a skill set spanning some of the most sought-after areas in AI research and the digital health sector: adapting and fine-tuning large language models; designing and running large-scale experiments on high-performance computing clusters; building simulation environments and agent-based evaluations; applying causal-inference thinking to intervention design; and designing rigorous evaluation methodology, a skill in short supply as AI systems increasingly interact with people. Beyond the technical, the student will work at the intersection of the DRIVE-Health CDT cohort, the King's NLP group, and King's wider digital-health research community, gaining experience of interdisciplinary collaboration with clinicians and industry partners, training in responsible AI and health-data governance, and the communication skills that come from publishing at leading AI/NLP venues. These capabilities transfer directly to careers in academic research, the growing digital-health industry, and AI research and engineering more broadly.

---

## How does your project fit EPSRC's remit?

The project's core contributions lie in engineering and physical sciences, specifically in artificial intelligence, machine learning, and natural language processing methodology: new methods for generative simulation of individual human behaviour, new evaluation methodology for simulation fidelity, and simulation-based engineering of interactive AI systems. These sit squarely within EPSRC's ICT/AI portfolio and speak directly to EPSRC's cross-cutting interest in digital twinning as an emerging engineering paradigm.

Healthcare is the *application* domain: it motivates the design constraints (longitudinal heterogeneous data, privacy, safety) and provides the evaluation setting, but the project is not clinical research. It involves no clinical trial, no patient recruitment for clinical outcomes, and no evaluation of a medical treatment, questions that would sit within MRC or NIHR remit. The methodological advances in user simulation, personalisation, fidelity validation, and counterfactual analysis are general computational-science contributions that transfer to any domain where AI systems interact with people, including education and assistive technology. This methods-led profile with a health application is exactly the mode in which EPSRC pursues its "transforming health and healthcare" ambitions: through engineering and information-science research that underpins, rather than conducts, clinical innovation.

---

## What data does the project require?

Longitudinal, individual-level data that links a person's *behaviour* to their *state* over time: conversational interactions with digital assistants or coaches; self-reported wellbeing measures; wearable-derived activity and sleep signals; and profile or contextual information. None of this needs to be identifiable: the project is designed to run on public, synthetic, and de-identified data throughout.

### What data is already available?

- **Public datasets**, including PMData (Simula), which pairs five months of wearable logging with daily wellness self-reports, and public long-horizon personalised-dialogue benchmarks such as PersonaMem, alongside large public human–assistant interaction corpora for population-level modelling.
- **Synthetic longitudinal health-interaction data** derived from public sources in the supervisory team's prior work on user simulation and digital twins, available to the student from day one.

No sensitive or identifiable data is required to begin the project.

### What data do you expect to gain or collect during the project?

- **Large-scale synthetic virtual-patient cohorts and benchmark suites** generated during the research, released openly where possible as a community resource.
- **Human evaluation data** from small-scale user studies assessing simulator fidelity and downstream utility, conducted under King's ethics approval.
- **Potentially**, subject to governance and ethics approvals, access to de-identified wearable and interaction data from ongoing digital-health research at King's in later stages of the PhD; this would provide valuable real-world grounding but is not a dependency of the core research plan.
