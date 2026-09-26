# Prompt Engineering Evaluation Lab

A hands-on educational application demonstrating the tangible impacts of prompt engineering techniques on Large Language Model (LLM) responses using customer support scenarios.

---

## 📌 Prompt Evolution

### 1. Prompt V1: Basic Instruction (Zero-Shot)
- **Techniques Used**: Zero-shot prompting.
- **Characteristics**: Simple direct command without explicit boundaries, role framing, or output rules.
- **Drawbacks**: Often produces generic responses, inconsistent tone, unverified commitments, and unstructured outputs.

### 2. Prompt V2: Role-Based & Structured
- **Techniques Added**: Role Prompting, Context, Structured Output.
- **Improvements**:
  - Sets a distinct identity (*"Senior Customer Support Specialist at Apex Tech Solutions"*).
  - Enforces a 4-step structural blueprint (*Greeting $\rightarrow$ Empathy $\rightarrow$ Solution $\rightarrow$ Next Steps*).
- **Result**: Dramatically improves empathy, tone consistency, and visual scanability.

### 3. Prompt V3: Role, Constraints & Few-Shot Demonstrations
- **Techniques Added**: Constraints, Few-Shot Prompting, Hallucination Prevention.
- **Improvements**:
  - Adds negative constraints (*"Do NOT make concrete financial promises"*).
  - Enforces conciseness limits (*Under 150 words*).
  - Provides concrete, multi-turn few-shot examples illustrating policy adherence.
- **Result**: Produces high-precision, low-risk responses optimized for real enterprise production environments.

---

## 📊 Evaluation Methodology

The lab evaluates generated outputs across 6 core qualitative criteria:

1. **Relevance**: Direct alignment with the user's specific problem statement.
2. **Instruction Following**: Adherence to given constraints and behavioral instructions.
3. **Professional Tone**: Appropriate empathy, courtesy, and company alignment.
4. **Completeness**: Addressing all user sub-questions without omitting critical context.
5. **Format Compliance**: Following layout rules, schemas, and word counts.
6. **Hallucination Risk**: Absence of unverified policy promises or fabricated facts.

---

## 🚀 Quickstart Guide

1. Clone or download this repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate