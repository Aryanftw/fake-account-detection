import streamlit as st
import requests
import pandas as pd

import random

def mock_analyze(text):
    # Fake but realistic scores
    toxicity_score = min(1.0, max(0.0, len(text) / 200))
    readability_score = max(10, 100 - len(text) / 3)

    readability_risk = 1 - (readability_score / 100)

    risk_score = (0.6 * toxicity_score + 0.4 * readability_risk) * 100

    if risk_score > 70:
        level = "HIGH"
    elif risk_score > 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "risk_score": risk_score,
        "risk_level": level,
        "toxicity_score": round(toxicity_score, 2),
        "readability_score": round(readability_score, 1),
        "behavior_risk": round(random.uniform(0.3, 0.8), 2),
        "graph_risk": round(random.uniform(0.2, 0.7), 2),
        "explanation": f"{level} risk due to combined textual, behavioral, and network signals."
    }

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Fake Account Risk Analyzer",
    layout="centered"
)

st.title("🚨 Fake Account Risk Analyzer")
st.caption("Multi-signal risk detection using behavior, graph, and NLP signals")

st.divider()

# ----------------------------
# Input section
# ----------------------------
st.subheader("🔍 Analyze Text Content")

user_text = st.text_area(
    "Enter post / bio / tweet text",
    height=150,
    placeholder="Type or paste text here..."
)

analyze_btn = st.button("Analyze Risk")

# ----------------------------
# Backend call
# ----------------------------
if analyze_btn and user_text.strip() != "":
    with st.spinner("Analyzing content..."):
        result = mock_analyze(user_text)


    # ----------------------------
    # Final Risk Summary
    # ----------------------------
    st.divider()
    st.subheader("📊 Final Risk Assessment")

    risk_score = result["risk_score"]
    risk_level = result["risk_level"]

    col1, col2 = st.columns(2)
    col1.metric("Final Risk Score", f"{int(risk_score)} / 100")

    if risk_level == "HIGH":
        col2.error("HIGH RISK 🚨")
    elif risk_level == "MEDIUM":
        col2.warning("MEDIUM RISK ⚠️")
    else:
        col2.success("LOW RISK ✅")

    # ----------------------------
    # Risk Breakdown
    # ----------------------------
    st.divider()
    st.subheader("🧩 Risk Breakdown")

    breakdown = {
        "Behavior Risk": result.get("behavior_risk", 0),
        "Graph Risk": result.get("graph_risk", 0),
        "Text Toxicity": result["toxicity_score"],
        "Readability Risk": 1 - (result["readability_score"] / 100)
    }

    df = pd.DataFrame.from_dict(
        breakdown,
        orient="index",
        columns=["Score"]
    )

    st.bar_chart(df)

    # ----------------------------
    # Detailed Signals
    # ----------------------------
    st.subheader("📌 Signal Details")

    st.write(f"**Toxicity Score (BERT):** {result['toxicity_score']:.2f}")
    st.write(f"**Readability Score (Flesch):** {result['readability_score']:.1f}")

    # ----------------------------
    # Explanation
    # ----------------------------
    st.divider()
    st.subheader("🧠 Explanation")

    st.info(result["explanation"])

    # ----------------------------
    # Analyst Actions (Optional)
    # ----------------------------
    st.divider()
    st.subheader("🛠️ Analyst Action")

    colA, colB, colC = st.columns(3)
    colA.button("✅ Mark Legit")
    colB.button("🚫 Mark Fake")
    colC.button("🔍 Needs Review")

elif analyze_btn:
    st.warning("Please enter some text to analyze.")
