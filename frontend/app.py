import streamlit as st
import pandas as pd
import random
import math

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Fake Account Risk Analyzer",
    layout="centered"
)

st.title("🚨 Fake Account Risk Analyzer")
st.caption("Behavioral + Textual + Network Risk Scoring")

st.divider()

# =====================================================
# MOCK BACKEND (REPLACE LATER)
# =====================================================
def mock_analyze(text, rf):
    # --- NLP ---
    toxicity_score = min(1.0, len(text) / 200)
    readability_score = max(10, 100 - len(text) / 3)
    readability_risk = 1 - (readability_score / 100)

    # --- Behavioral risk (RF-like heuristic) ---
    behavior_risk = min(
        1.0,
        0.30 * (rf["statuses_per_day"] / 150) +
        0.25 * (1 / (rf["ff_ratio"] + 0.1)) +
        0.20 * (1 / (rf["log_followers_count"] + 1)) +
        0.15 * (1 if not rf["has_profile_pic"] else 0) +
        0.10 * random.uniform(0.3, 0.7)
    )

    # --- Graph risk (mocked) ---
    graph_risk = random.uniform(0.3, 0.7)

    # --- Final fusion ---
    final_risk = (
        0.35 * behavior_risk +
        0.30 * toxicity_score +
        0.20 * graph_risk +
        0.15 * readability_risk
    ) * 100

    if final_risk >= 70:
        level = "HIGH"
    elif final_risk >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "final_risk_score": round(final_risk, 1),
        "risk_level": level,
        "behavior_risk": round(behavior_risk, 2),
        "graph_risk": round(graph_risk, 2),
        "toxicity_score": round(toxicity_score, 2),
        "readability_score": round(readability_score, 1),
        "explanation": f"{level} risk due to abnormal behavior, toxic language, and network signals."
    }

# =====================================================
# INPUTS
# =====================================================

st.subheader("📝 Text Content")

text_type = st.selectbox(
    "Select text type",
    ["Post", "Bio", "Username"]
)

text_input = st.text_area(
    f"Enter {text_type.lower()} text",
    height=120,
    placeholder=f"Paste {text_type.lower()} here..."
)

st.divider()
st.subheader("🎛️ Behavioral Features (Random Forest Inputs)")

col1, col2 = st.columns(2)

with col1:
    log_followers_count = st.slider(
        "Log10 Followers Count",
        0.0, 7.0, 4.0, 0.1,
        help="log10(followers + 1)"
    )

    log_friends_count = st.slider(
        "Log10 Following Count",
        0.0, 7.0, 4.2, 0.1,
        help="log10(following + 1)"
    )

    statuses_per_day = st.slider(
        "Statuses Per Day",
        0.0, 150.0, 8.0, 1.0
    )

with col2:
    favourites_count = st.slider(
        "Favourites Count",
        0, 10000, 300, 50
    )

    listed_count = st.slider(
        "Listed Count",
        0, 500, 15, 5
    )

    verified = st.checkbox("Verified Account", value=False)
    has_profile_pic = st.checkbox("Has Profile Picture", value=True)

# =====================================================
# DERIVED FEATURE (ff_ratio)
# =====================================================
followers = math.pow(10, log_followers_count)
friends = math.pow(10, log_friends_count)

ff_ratio = followers / (friends + 1)

st.metric(
    "Follower / Following Ratio (Derived)",
    round(ff_ratio, 3)
)

rf_features = {
    "log_followers_count": log_followers_count,
    "log_friends_count": log_friends_count,
    "ff_ratio": ff_ratio,
    "statuses_per_day": statuses_per_day,
    "favourites_count": favourites_count,
    "listed_count": listed_count,
    "verified": int(verified),
    "has_profile_pic": int(has_profile_pic)
}

with st.expander("🔍 RF Features Sent to Model"):
    st.json(rf_features)

st.divider()
analyze_btn = st.button("🔍 Analyze Risk")

# =====================================================
# OUTPUT
# =====================================================
if analyze_btn:
    if text_input.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Analyzing risk..."):
        result = mock_analyze(text_input, rf_features)

    st.subheader("📊 Final Risk Assessment")

    c1, c2 = st.columns(2)
    c1.metric("Final Risk Score", f"{result['final_risk_score']} / 100")

    if result["risk_level"] == "HIGH":
        c2.error("HIGH RISK 🚨")
    elif result["risk_level"] == "MEDIUM":
        c2.warning("MEDIUM RISK ⚠️")
    else:
        c2.success("LOW RISK ✅")

    st.divider()
    st.subheader("🧩 Risk Breakdown")

    df = pd.DataFrame({
        "Signal": ["Behavior", "Text Toxicity", "Graph", "Readability"],
        "Score": [
            result["behavior_risk"],
            result["toxicity_score"],
            result["graph_risk"],
            1 - (result["readability_score"] / 100)
        ]
    }).set_index("Signal")

    st.bar_chart(df)

    st.divider()
    st.subheader("🧠 Explanation")
    st.info(result["explanation"])
