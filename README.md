# 🎓 PARIX — Student Intervention Analytics  
### Turning Classroom Data into Actionable Dialogue

> **PARIX is not a grading system.**  
> It is a **decision-support engine** that helps teachers identify *why* a student is struggling and *what intervention is most likely to help* — before failure becomes irreversible.

## 🔗 Live Demo

🖥️ **Interactive Teacher Dashboard (Streamlit)**  
https://parix-for-teachers.streamlit.app/

---

## 🧩 The Problem We’re Solving

Teachers already have access to:
- Grades
- Attendance
- Behavioral records

What they *don’t* have is:
- Time to analyze root causes
- Confidence in choosing the *right* intervention
- A defensible way to explain decisions to parents

As a result, most interventions are **generic, late, or misaligned**.

**PARIX addresses this gap** by transforming raw classroom data into:
- Early risk signals
- Root-cause–driven insights
- Clear, low-effort intervention guidance

---

## 🧠 Our Philosophy: *Predictive Empathy*

Instead of asking:  
> *“How well did this student perform?”*

PARIX asks:  
> **“Given this student’s effort and environment, how should they have performed — and where is the learning process breaking down?”**

This shift prevents:
- One-size-fits-all labeling  
- Bias against students with structural constraints  
- Reactive, end-of-term interventions  

---

## 🏗 System Overview

PARIX follows a **four-layer analytical pipeline**, where each layer answers a different teacher-facing question:

| Layer | Question Answered | Output |
|-----|------------------|--------|
| Diagnose | *Is this student underperforming relative to their context?* | Effort–Outcome Gap |
| Understand | *Why is this happening?* | Failure-Mode Persona |
| Decide | *What should I change first?* | Primary Intervention Lever |
| Act | *How do I communicate and intervene?* | AI Playbooks |

---

## 🔍 Feature Deep Dive (What, Why, How)

### 1️⃣ Effort–Outcome Gap Analysis  
**Purpose:** Detect *silent struggle* early

| Aspect | Description |
|------|-------------|
| **What** | A Linear Regression model estimates an **Expected Exam Score** using 16 behavioral and contextual features. |
| **Why** | Raw scores are misleading. Two students with the same marks may face very different challenges. |
| **How** | The gap between actual and expected score is standardized as a **Z-score**, flagging underperformance relative to individual context. |

📌 **Key Insight:**  
This approach reduces bias by comparing students *against their own conditions*, not against each other.

---

### 2️⃣ Failure-Mode Persona Clustering  
**Purpose:** Replace labels with patterns

| Aspect | Description |
|------|-------------|
| **What** | K-Means clustering groups students into 4 distinct **failure modes**. |
| **Why** | Teachers intervene more effectively when they understand *patterns*, not isolated metrics. |
| **How** | Clustering is performed on performance gaps + key drivers (sleep, motivation, attendance, resources). |

**Example Personas**
- Overworked Strugglers  
- Disengaged Despite Resources  
- Structurally Constrained Learners  
- Stable Performers  

📌 Personas describe **problems**, not people.

---

### 3️⃣ Resource–Performance Mismatch Index  
**Purpose:** Avoid misdirected interventions

| Aspect | Description |
|------|-------------|
| **What** | A normalized composite index of access-related features (Internet, Resources, Income). |
| **Why** | Academic struggle is often mistaken for lack of effort when it is actually lack of support. |
| **How** | Resource index is compared against the Effort–Outcome Gap to flag mismatches. |

📌 Helps distinguish *behavioral* issues from *structural* ones.

---

### 4️⃣ Primary Intervention Lever Detection  
**Purpose:** Force clarity, prevent overload

| Aspect | Description |
|------|-------------|
| **What** | An interpretable rule-based decision engine assigns **one dominant intervention lever** per student. |
| **Why** | Teachers need focus, not long checklists. |
| **How** | Deterministic logic prioritizes the factor with highest modeled impact (sleep, attendance, tutoring, etc.). |

📌 One student → one primary lever → one clear action.

---

### 5️⃣ Intervention Simulation (What-If Analysis)  
**Purpose:** Support decision-making, not guesswork

| Aspect | Description |
|------|-------------|
| **What** | Simulates expected score improvement if a lever is modestly adjusted. |
| **Why** | Teachers want to know *what is worth their effort*. |
| **How** | Feature values are perturbed and passed through the trained model to estimate sensitivity. |

📌 These are **model-estimated scenarios**, not causal guarantees.

---

### 6️⃣ GenAI Playbook Engine  
**Purpose:** Translate analytics into human dialogue

| Aspect | Description |
|------|-------------|
| **What** | A strict-contract GenAI layer generates teacher playbooks and parent communication drafts. |
| **Why** | Insights are useless if they can’t be communicated clearly and responsibly. |
| **How** | Only structured analytical outputs (no raw data) are passed into GLM-4.5 via OpenRouter. |

**Outputs include:**
- 2-sentence root-cause explanation  
- Actionable intervention steps (owner, metric, timeframe)  
- Non-judgmental parent message draft  

📌 AI explains decisions — it does not make them.

---

## 📊 Teacher-Facing Dashboard

**Weekly Priority View**
- Risk distribution across the class
- Priority matrix highlighting who needs attention *now*
- Top 3 students selected to prevent overload

**Student Deep Dive**
- Persona + primary lever
- Expected improvement
- Radar chart vs class average
- One-click AI playbook generation

---

## 🧠 What PARIX Explicitly Does *Not* Do

To maintain ethical clarity and trust:

- ❌ Does not diagnose learning disabilities  
- ❌ Does not replace teacher judgment  
- ❌ Does not guarantee score improvements  

**PARIX supports decisions — it does not automate them.**

---

## 🏗 Technical Architecture

```text
ps_2/
├── src/
│   ├── data/           # Preprocessing & ID generation
│   ├── models/         # Expected-score model logic
│   ├── features/       # Gaps, personas, simulations
│   └── explainability/ # GenAI payload & prompts
├── ui/                 # Streamlit teacher dashboard
├── models/             # Serialized artifacts (.joblib)
└── notebooks/          # EDA & experimentation
