import os
import json
import csv
import io
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1. Configuration (Groq API Setup)
# ------------------------------------------------------------------------------
load_dotenv()

# Define API key and model variables properly
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Initialize Groq Client safely
client = Groq(api_key=API_KEY) if API_KEY else None

st.set_page_config(
    page_title="Prompt Engineering Evaluation Lab",
    page_icon="🧪",
    layout="wide",
)


# ------------------------------------------------------------------------------
# 2. Session State
# ------------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ------------------------------------------------------------------------------
# 3. Styling
# ------------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* General spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Response cards */
    .response-card {
        border-radius: 14px;
        padding: 18px;
        min-height: 230px;
        color: #f8fafc;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }

    .response-card h3 {
        margin-top: 0;
        margin-bottom: 14px;
        color: #ffffff;
    }

    .response-card p {
        white-space: pre-wrap;
        line-height: 1.6;
    }

    .v1-card {
        background: #172554;
        border-left: 5px solid #3b82f6;
    }

    .v2-card {
        background: #2e1065;
        border-left: 5px solid #8b5cf6;
    }

    .v3-card {
        background: #052e16;
        border-left: 5px solid #22c55e;
    }

    .error-card {
        background: #450a0a;
        border-left: 5px solid #ef4444;
    }

    /* Score cards */
    .score-card {
        background: #171923;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        border: 1px solid #30343f;
    }

    .score-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 4px;
    }

    .score-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #60a5fa;
    }

    /* What changed */
    .diff-card {
        background: #171923;
        border-radius: 12px;
        padding: 18px;
        border-left: 5px solid #60a5fa;
        min-height: 180px;
    }

    .diff-card h4 {
        color: #ffffff;
        margin-top: 0;
    }

    .diff-card li {
        margin-bottom: 8px;
        color: #e2e8f0;
    }

    /* Small project badge */
    .project-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #172554;
        color: #93c5fd;
        font-size: 0.82rem;
        margin-right: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 4. Prompt Loading
# ------------------------------------------------------------------------------
def load_prompt(filename: str, fallback: str) -> str:
    filepath = Path("prompts") / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return fallback


DEFAULT_V1 = """Respond to the following customer inquiry:

{customer_message}"""


DEFAULT_V2 = """You are a senior customer support specialist at Apex Tech Solutions.

CUSTOMER MESSAGE:
{customer_message}

RESPONSE STRUCTURE:
1. Greeting: Address the customer respectfully.
2. Empathy Statement: Acknowledge their situation or frustration directly.
3. Solution / Direct Action: Provide actionable steps or a status update.
4. Next Steps / Closing: Offer further assistance politely.

Maintain a polite, professional, and efficient tone."""


DEFAULT_V3 = """You are a senior customer support specialist at Apex Tech Solutions.

CUSTOMER MESSAGE:
{customer_message}

CONSTRAINTS:
- Do not make unconfirmed financial promises.
- Do not invent account access, refunds, IDs, URLs, or actions that have not actually been performed.
- Keep the response under 150 words.
- Maintain a warm, professional, and concise tone.
- If information is missing, clearly say what the customer should provide or do next.

FEW-SHOT EXAMPLE 1
Customer: "I was charged twice for my subscription."
Response: "I understand how concerning a duplicate charge can be. Please check whether both charges are posted rather than pending. If both are confirmed, contact support with the transaction details so the duplicate charge can be reviewed."

FEW-SHOT EXAMPLE 2
Customer: "I cannot log into my account."
Response: "I'm sorry you're having trouble accessing your account. Please try the password-reset option and check your email, including your spam folder. If the reset email does not arrive, contact support with the account email so the issue can be investigated."

Write a helpful response to the customer message above."""


prompt_v1_raw = load_prompt("prompt_v1.txt", DEFAULT_V1)
prompt_v2_raw = load_prompt("prompt_v2.txt", DEFAULT_V2)
prompt_v3_raw = load_prompt("prompt_v3.txt", DEFAULT_V3)


def inject_customer_message(prompt: str, customer_message: str) -> str:
    return prompt.replace("{customer_message}", customer_message)


# ------------------------------------------------------------------------------
# 5. Evaluation Schema
# ------------------------------------------------------------------------------
Rating = Literal["Excellent", "Good", "Needs Improvement"]


class EvaluationCriterion(BaseModel):
    label: Rating = Field(
        description="Rating must be exactly Excellent, Good, or Needs Improvement."
    )
    explanation: str = Field(
        description="Brief 1-2 sentence explanation supporting the rating."
    )


class PromptEvaluation(BaseModel):
    version_name: str = Field(
        description="Prompt version evaluated, such as Prompt V1."
    )
    relevance: EvaluationCriterion
    instruction_following: EvaluationCriterion
    professional_tone: EvaluationCriterion
    completeness: EvaluationCriterion
    format_compliance: EvaluationCriterion
    hallucination_risk: EvaluationCriterion


LABEL_WEIGHTS = {
    "Needs Improvement": 1,
    "Good": 3,
    "Excellent": 5,
}


# ------------------------------------------------------------------------------
# 6. Groq Generation
# ------------------------------------------------------------------------------
def generate_response(prompt: str) -> str:
    if not client:
        return (
            "CONFIGURATION ERROR: GROQ_API_KEY is missing. "
            "Add it to your .env file and restart Streamlit."
        )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=500,
        )

        text = response.choices[0].message.content

        if not text:
            return "API ERROR: The model returned an empty response."

        return text.strip()

    except Exception as exc:
        return f"API ERROR: {exc}"


def is_error_response(text: str) -> bool:
    return text.startswith(("API ERROR:", "CONFIGURATION ERROR:"))


# ------------------------------------------------------------------------------
# 7. AI Judge (Groq Evaluation Implementation)
# ------------------------------------------------------------------------------
def evaluate_response(
    version: str,
    prompt_used: str,
    customer_msg: str,
    response_text: str,
) -> PromptEvaluation:

    if is_error_response(response_text) or not client:
        dummy = EvaluationCriterion(
            label="Needs Improvement",
            explanation="Evaluation failed due to missing API key or generation error.",
        )
        return PromptEvaluation(
            version_name=version,
            relevance=dummy,
            instruction_following=dummy,
            professional_tone=dummy,
            completeness=dummy,
            format_compliance=dummy,
            hallucination_risk=dummy,
        )

    eval_prompt = f"""
You are an objective prompt-engineering evaluator.

Evaluate the generated customer-support response against the customer's message and the prompt that produced it.

CUSTOMER MESSAGE:
{customer_msg}

PROMPT USED:
{prompt_used}

GENERATED RESPONSE:
{response_text}

Evaluate these six criteria:
1. Relevance
2. Instruction Following
3. Professional Tone
4. Completeness
5. Format Compliance
6. Hallucination Risk

Use only these labels:
- Excellent
- Good
- Needs Improvement

Respond strictly in raw JSON adhering to this schema:
{json.dumps(PromptEvaluation.model_json_schema(), indent=2)}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": eval_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_completion_tokens=1000,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("The evaluator returned an empty response.")

        return PromptEvaluation.model_validate_json(content)

    except Exception as exc:
        dummy = EvaluationCriterion(
            label="Needs Improvement",
            explanation=f"Evaluation failed: {exc}",
        )
        return PromptEvaluation(
            version_name=version,
            relevance=dummy,
            instruction_following=dummy,
            professional_tone=dummy,
            completeness=dummy,
            format_compliance=dummy,
            hallucination_risk=dummy,
        )


def compute_scorecard(eval_obj: PromptEvaluation) -> float:
    criteria = [
        eval_obj.relevance,
        eval_obj.instruction_following,
        eval_obj.professional_tone,
        eval_obj.completeness,
        eval_obj.format_compliance,
        eval_obj.hallucination_risk,
    ]

    total = sum(LABEL_WEIGHTS.get(item.label, 1) for item in criteria)
    return round(total / len(criteria), 1)


# ------------------------------------------------------------------------------
# 8. Export Helpers
# ------------------------------------------------------------------------------
def create_json_export(export_data: dict) -> str:
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def create_csv_export(export_data: dict) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Scenario", export_data["scenario"]])
    writer.writerow(["Customer Message", export_data["customer_message"]])
    writer.writerow(["Model", export_data["model"]])
    writer.writerow([])

    writer.writerow(
        [
            "Prompt Version",
            "Average Score",
            "Generated Response",
            "Relevance Label",
            "Relevance Rationale",
            "Instruction Following Label",
            "Instruction Following Rationale",
            "Professional Tone Label",
            "Professional Tone Rationale",
            "Completeness Label",
            "Completeness Rationale",
            "Format Compliance Label",
            "Format Compliance Rationale",
            "Hallucination Risk Label",
            "Hallucination Risk Rationale",
        ]
    )

    for key in ["V1", "V2", "V3"]:
        version_data = export_data["versions"][key]
        evaluation = version_data["evaluation"]

        writer.writerow(
            [
                key,
                f"{version_data['score']}/5",
                version_data["response"],
                evaluation["relevance"]["label"],
                evaluation["relevance"]["explanation"],
                evaluation["instruction_following"]["label"],
                evaluation["instruction_following"]["explanation"],
                evaluation["professional_tone"]["label"],
                evaluation["professional_tone"]["explanation"],
                evaluation["completeness"]["label"],
                evaluation["completeness"]["explanation"],
                evaluation["format_compliance"]["label"],
                evaluation["format_compliance"]["explanation"],
                evaluation["hallucination_risk"]["label"],
                evaluation["hallucination_risk"]["explanation"],
            ]
        )

    return output.getvalue()


# ------------------------------------------------------------------------------
# 9. Header
# ------------------------------------------------------------------------------
st.title("🧪 Prompt Engineering Evaluation Lab (Groq)")
st.caption(
    "Demonstrating how prompt engineering techniques affect LLM responses "
    "in customer-support scenarios."
)

st.markdown(
    """
<span class="project-badge">Python</span>
<span class="project-badge">Streamlit</span>
<span class="project-badge">Groq API</span>
<span class="project-badge">Pydantic</span>
""",
    unsafe_allow_html=True,
)

if not API_KEY:
    st.warning(
        "⚠️ GROQ_API_KEY was not found. Add it to your .env file and restart the app."
    )


# ------------------------------------------------------------------------------
# 10. Preset Scenarios
# ------------------------------------------------------------------------------
PRESET_SCENARIOS = {
    "Custom Input": "",
    "Delayed Refund": (
        "I returned my item over 12 days ago (Order #8831) and still haven't "
        "received my refund! This is unacceptable service. I want my money back immediately!"
    ),
    "Login Failure": (
        "I'm locked out of my portal right before an important client presentation. "
        "Resetting password isn't sending any email. Fix this now!"
    ),
    "Duplicate Charge": (
        "I saw two identical charges of $89.00 on my credit card statement today "
        "for order #5510. Please reverse one immediately."
    ),
    "Subscription Cancellation": (
        "I tried canceling my subscription online but the button is greyed out. "
        "I do not want to be billed for next month!"
    ),
    "Delivery Delay": (
        "My package was supposed to arrive 3 days ago according to tracking "
        "(#TRK9021), but it hasn't moved. Where is it?"
    ),
    "Password Reset": (
        "Hi, I forgot my password and can't access my account. How do I request a reset link?"
    ),
    "Angry Customer": (
        "Your app crashed and deleted 2 hours of my unsaved work! This is garbage "
        "software and I am demanding a full annual refund right now!"
    ),
    "Missing Order": (
        "My order #1092 says 'Delivered' on the portal, but there is nothing on my porch. Please help."
    ),
    "Account Locked": (
        "Received an alert saying my account was locked due to security reasons. "
        "I need access restored ASAP."
    ),
    "Unclear Customer Request": "it broken need fix fast send email back.",
}


# ------------------------------------------------------------------------------
# 11. Sidebar
# ------------------------------------------------------------------------------
st.sidebar.header("🎯 Preset Test Cases")

selected_preset = st.sidebar.selectbox(
    "Choose a scenario:",
    list(PRESET_SCENARIOS.keys()),
)

st.sidebar.markdown("---")
st.sidebar.header("📋 Evaluation History")

if st.session_state.history:
    for item in reversed(st.session_state.history):
        st.sidebar.markdown(f"**✓ {item['scenario']}**")
        st.sidebar.caption(
            f"V1: {item['s1']}/5  |  V2: {item['s2']}/5  |  V3: {item['s3']}/5"
        )
else:
    st.sidebar.caption("No tests run yet in this session.")


# ------------------------------------------------------------------------------
# 12. Customer Input
# ------------------------------------------------------------------------------
customer_message = st.text_area(
    "Customer Message:",
    value=PRESET_SCENARIOS[selected_preset],
    height=120,
    placeholder="Enter a customer inquiry...",
)


# ------------------------------------------------------------------------------
# 13. Prompt Templates
# ------------------------------------------------------------------------------
st.markdown("### 🔍 Prompt Engineering Templates")

tab1, tab2, tab3 = st.tabs(
    [
        "Prompt V1 (Basic)",
        "Prompt V2 (Structured)",
        "Prompt V3 (Advanced)",
    ]
)

with tab1:
    st.caption("Zero-shot prompting without additional role, context, or constraints.")
    st.code(prompt_v1_raw, language="text")

with tab2:
    st.caption("Role prompting with context and explicit response structure.")
    st.code(prompt_v2_raw, language="text")

with tab3:
    st.caption(
        "Role prompting, constraints, guardrails, and few-shot examples."
    )
    st.code(prompt_v3_raw, language="text")


# ------------------------------------------------------------------------------
# 14. Prompt Techniques Matrix
# ------------------------------------------------------------------------------
st.markdown("### 🧠 Prompt Techniques Matrix")

st.markdown(
    """
| Prompt Engineering Technique | V1 (Basic) | V2 (Structured) | V3 (Advanced) |
|:---|:---:|:---:|:---:|
| **Zero-shot prompting** | ✅ | ✅ | ✅ |
| **Role prompting** | ❌ | ✅ | ✅ |
| **Context & Domain Scope** | ❌ | ✅ | ✅ |
| **Structured instructions** | ❌ | ✅ | ✅ |
| **Constraints & Guardrails** | ❌ | ❌ | ✅ |
| **Few-shot examples** | ❌ | ❌ | ✅ |
| **Hallucination controls** | ❌ | ❌ | ✅ |
"""
)


# ------------------------------------------------------------------------------
# 15. Compare Button & Execution
# ------------------------------------------------------------------------------
if st.button(
    "🚀 Compare Prompts",
    type="primary",
    use_container_width=True,
):

    if not customer_message.strip():
        st.error("Please enter a customer message first.")

    elif not client:
        st.error(
            "Groq API is not configured. Add GROQ_API_KEY to your .env file "
            "and restart Streamlit."
        )

    else:
        p1 = inject_customer_message(prompt_v1_raw, customer_message)
        p2 = inject_customer_message(prompt_v2_raw, customer_message)
        p3 = inject_customer_message(prompt_v3_raw, customer_message)

        st.markdown("---")
        st.markdown(
            "<h3 style='text-align:center;'>GENERATED RESPONSES</h3>",
            unsafe_allow_html=True,
        )

        with st.spinner("Generating V1, V2, and V3 responses via Groq..."):
            r1 = generate_response(p1)
            r2 = generate_response(p2)
            r3 = generate_response(p3)

        response_columns = st.columns(3)

        cards = [
            ("V1 — Zero-Shot", r1, "v1-card"),
            ("V2 — Role + Structure", r2, "v2-card"),
            ("V3 — Constraints + Few-Shot", r3, "v3-card"),
        ]

        for column, (title, response_text, card_class) in zip(
            response_columns, cards
        ):
            with column:
                actual_class = "error-card" if is_error_response(response_text) else card_class

                safe_text = (
                    response_text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )

                st.markdown(
                    f"""
                    <div class="response-card {actual_class}">
                        <h3>{title}</h3>
                        <p>{safe_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if any(is_error_response(item) for item in [r1, r2, r3]):
            st.error(
                "At least one Groq request failed. Fix the API key/model configuration."
            )
            st.stop()

        st.markdown("---")
        st.markdown(
            "<h3 style='text-align:center;'>PROMPT EVALUATION</h3>",
            unsafe_allow_html=True,
        )

        with st.spinner("Evaluating responses against the QA rubric..."):
            e1 = evaluate_response("Prompt V1", prompt_v1_raw, customer_message, r1)
            e2 = evaluate_response("Prompt V2", prompt_v2_raw, customer_message, r2)
            e3 = evaluate_response("Prompt V3", prompt_v3_raw, customer_message, r3)

        s1 = compute_scorecard(e1)
        s2 = compute_scorecard(e2)
        s3 = compute_scorecard(e3)

        scenario_name = (
            selected_preset
            if selected_preset != "Custom Input"
            else "Custom Test"
        )

        st.session_state.history.append(
            {
                "scenario": scenario_name,
                "s1": s1,
                "s2": s2,
                "s3": s3,
            }
        )

        sc1, sc2, sc3 = st.columns(3)

        for column, version, score in [
            (sc1, "V1 Score", s1),
            (sc2, "V2 Score", s2),
            (sc3, "V3 Score", s3),
        ]:
            with column:
                st.markdown(
                    f"""
                    <div class="score-card">
                        <div class="score-title">{version}</div>
                        <div class="score-val">{score} / 5</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### 📊 Detailed Evaluation")

        evals = [
            ("Prompt V1", e1),
            ("Prompt V2", e2),
            ("Prompt V3", e3),
        ]

        criteria_keys = [
            ("Relevance", "relevance"),
            ("Instruction Following", "instruction_following"),
            ("Professional Tone", "professional_tone"),
            ("Completeness", "completeness"),
            ("Format Compliance", "format_compliance"),
            ("Hallucination Risk", "hallucination_risk"),
        ]

        cols = st.columns(3)

        badge_colors = {
            "Excellent": "🟢",
            "Good": "🟡",
            "Needs Improvement": "🔴",
        }

        for index, (version_name, evaluation) in enumerate(evals):
            with cols[index]:
                st.markdown(f"### {version_name}")

                for criterion_name, field_name in criteria_keys:
                    criterion = getattr(evaluation, field_name)
                    badge = badge_colors.get(criterion.label, "⚪")

                    st.markdown(
                        f"**{criterion_name}:** {badge} {criterion.label}"
                    )
                    st.caption(criterion.explanation)
                    st.write("")

        st.markdown("---")
        st.markdown(
            "<h3 style='text-align:center;'>EXPORT RESULTS</h3>",
            unsafe_allow_html=True,
        )

        export_data = {
            "scenario": scenario_name,
            "customer_message": customer_message,
            "model": MODEL_NAME,
            "versions": {
                "V1": {
                    "score": s1,
                    "response": r1,
                    "evaluation": e1.model_dump(),
                },
                "V2": {
                    "score": s2,
                    "response": r2,
                    "evaluation": e2.model_dump(),
                },
                "V3": {
                    "score": s3,
                    "response": r3,
                    "evaluation": e3.model_dump(),
                },
            },
        }

        json_str = create_json_export(export_data)
        csv_str = create_csv_export(export_data)

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.download_button(
                label="📥 Download Results as JSON",
                data=json_str,
                file_name=f"evaluation_{scenario_name.lower().replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with exp_col2:
            st.download_button(
                label="📊 Download Results as CSV",
                data=csv_str,
                file_name=f"evaluation_{scenario_name.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown(
            "<h3 style='text-align:center;'>WHAT CHANGED?</h3>",
            unsafe_allow_html=True,
        )

        what_changed_col1, what_changed_col2 = st.columns(2)

        with what_changed_col1:
            st.markdown(
                """
                <div class="diff-card">
                    <h4>V1 → V2</h4>
                    <ul>
                        <li>➕ <b>Role prompting</b> — defines a customer-support role.</li>
                        <li>➕ <b>Context</b> — identifies the business/domain.</li>
                        <li>➕ <b>Structured instructions</b> — introduces a four-step reply framework.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with what_changed_col2:
            st.markdown(
                """
                <div class="diff-card">
                    <h4>V2 → V3</h4>
                    <ul>
                        <li>➕ <b>Constraints</b> — prevents unsupported promises.</li>
                        <li>➕ <b>Guardrails</b> — limits response length and unsupported claims.</li>
                        <li>➕ <b>Few-shot examples</b> — demonstrates the desired response behavior.</li>
                        <li>➕ <b>Hallucination controls</b> — discourages invented IDs, URLs, actions, or access.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------------------------
# 16. About
# ------------------------------------------------------------------------------
st.markdown("---")

with st.expander("ℹ️ About This Lab", expanded=False):
    st.markdown(
        f"""
**Problem:**  
LLM outputs can vary depending on how instructions are written. Unstructured
prompts can lead to inconsistent tone, formatting, and unsupported claims.

**Objective:**  
Compare three prompt versions and evaluate their effect on customer-support
responses using six observable criteria.

**Technologies:**  
`Python` • `Streamlit` • `Groq API` • `Pydantic`

**Model:**  
`{MODEL_NAME}`

**API key:**  
Loaded server-side from `.env`; it is never requested from the browser.
"""
    )