import numpy as np
import pandas as pd
import torch
import joblib
import shap
import textstat
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from typing import List
from pydantic import Field
# =========================================================
# 1. SETUP & MODEL LOADING
# =========================================================
app = FastAPI(
    title="Omni-Fusion Bot Detection API",
    description="Fuses BERTweet, Random Forest, LightGBM, and Isolation Forest for MGTAB/Cresci analysis."
)

print("⏳ [INIT] Loading Intelligence Engines...")
text_encoder = SentenceTransformer("all-distilroberta-v1")
# --- A. LINGUISTIC ENGINE (BERTweet) ---
# Ensure this path matches where you unzipped your Colab model
BERT_PATH = "../model.safetensors"
try:
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
    bert_model.eval()
    print("✅ [LOADED] BERTweet Linguistic Model")
except Exception as e:
    print(f"⚠️ [WARNING] BERTweet not found at {BERT_PATH}. Using mock logic for testing.")
    bert_model = None
def bertweet_tweet_level_risk(tweets):
    if not bert_model or not tweets:
        return 0.0, {}

    risks = []
    for t in tweets:
        norm = normalize_tweet(t)
        inputs = bert_tokenizer(norm, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = bert_model(**inputs).logits
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            risks.append(prob)

    return float(np.mean(risks)), {
        "mean": round(np.mean(risks), 3),
        "max": round(np.max(risks), 3),
        "variance": round(np.var(risks), 4)
    }
def analyze_textstat_tweets(tweets):
    if not tweets:
        return {"linguistic_risk": "No Tweets"}

    eases, grades, lengths = [], [], []

    for t in tweets:
        if len(t.strip()) < 5:
            continue
        eases.append(textstat.flesch_reading_ease(t))
        grades.append(textstat.flesch_kincaid_grade(t))
        lengths.append(len(t.split()))

    if not eases:
        return {"linguistic_risk": "Low Content"}

    avg_len = np.mean(lengths)

    if avg_len < 6 and np.mean(eases) > 85:
        risk = "High Bot Likelihood"
        reason = "Short, repetitive, simplistic tweets"
    elif np.var(grades) < 1.0:
        risk = "Moderate Bot Likelihood"
        reason = "Low linguistic diversity"
    else:
        risk = "Low Bot Likelihood"
        reason = "Human-like variation"

    return {
        "avg_tweet_length": round(avg_len, 2),
        "reading_ease_mean": round(np.mean(eases), 2),
        "grade_variance": round(np.var(grades), 2),
        "linguistic_risk": risk,
        "interpretation": reason
    }

# --- B. BEHAVIORAL ENGINE (Random Forest) ---
try:
    rf_model = joblib.load("../random_forest_model.pkl")
    rf_explainer = shap.TreeExplainer(rf_model)
    print("✅ [LOADED] Random Forest (Cresci Metadata)")
except:
    print("⚠️ [WARNING] Random Forest .pkl not found.")
    rf_model = None

# --- C. NETWORK & ANOMALY ENGINE (LightGBM + IsoForest) ---
# Assuming these were saved from your MGTAB notebook
try:
    lgb_model = joblib.load("../mgtab_lightgbm.pkl")
    iso_model = joblib.load("../mgtab_isolation_forest.pkl")
    print("✅ [LOADED] LightGBM & Isolation Forest")
except:
    print("⚠️ [WARNING] MGTAB models not found.")
    lgb_model = None
    iso_model = None
# =========================================================
# 2. DATA SCHEMAS
# =========================================================
class AccountRequest(BaseModel):
    username: str
    name: str = ""
    tweets: List[str] = Field(default_factory=list)

    followers_count: int
    friends_count: int
    statuses_count: int
    favourites_count: int
    listed_count: int

    verified: bool
    protected: bool = False
    default_profile: bool = False
    default_profile_image: bool = False
    is_geo_enabled: bool = False

    created_at: str
    account_age_days: int


def normalize_tweet(text):
    """BERTweet required normalization."""
    new_text = []
    for t in text.split():
        if t.startswith("@") and len(t) > 1: new_text.append("@USER")
        elif t.startswith("http"): new_text.append("HTTPURL")
        else: new_text.append(t)
    return " ".join(new_text)

def build_mgtab_embedding(data: AccountRequest) -> np.ndarray:
    """
    Returns 790-dim MGTAB embedding:
    [768 text | 22 metadata]
    """

    # ---- TEXT (768) ----
    joined_text = " ".join(data.tweets[:20]) if data.tweets else ""
    text_emb = text_encoder.encode(joined_text if joined_text else "empty")



    age = max(data.account_age_days, 1)

    # ---- METADATA (22 – EXACT ORDER) ----
    meta_feat = np.array([
        np.log1p(data.followers_count),
        np.log1p(data.friends_count),
        np.log1p(data.statuses_count),
        np.log1p(data.favourites_count),
        np.log1p(data.listed_count),

        float(len(data.username)),
        float(len(data.name)),
        float(len(joined_text)),

        float(data.verified),
        float(data.protected),
        float(data.default_profile),
        float(data.default_profile_image),

        np.log1p(data.followers_count / (data.friends_count + 1)),
        np.log1p(data.friends_count / (data.followers_count + 1)),

        float(data.account_age_days),
        data.statuses_count / age,
        data.favourites_count / age,

        float(data.is_geo_enabled),

        np.log1p(data.account_age_days),
        data.followers_count / age,
        data.friends_count / age,
        data.statuses_count / (data.followers_count + 1),
    ], dtype=np.float32)

    # Standardize metadata (same as training-time safe norm)
    meta_feat = (meta_feat - meta_feat.mean()) / (meta_feat.std() + 1e-7)

    # ---- CONCAT (790) ----
    final_vec = np.concatenate([text_emb, meta_feat]).astype(np.float32)

    return final_vec.reshape(1, -1)

def build_metadata_vector(data: AccountRequest):
    """
    Constructs vector strictly matching the 9 features in your notebook:
    [statuses_count, friends_count, favourites_count, listed_count, 
     utc_offset, log_followers_count, ff_ratio, statuses_per_day, listed_count]
    """
    
    # 1. Compute Account Age (Days)
    # Using 'coerce' logic to handle format errors gracefully
    try:
        # Try standardized UTC parsing
        created_dt = pd.to_datetime(data.created_at, utc=True)
    except:
        # Fallback to current time if parsing fails (prevents API crash)
        created_dt = pd.Timestamp.now(tz="UTC")
    
    now_utc = pd.Timestamp.now(tz="UTC")
    account_age_days = (now_utc - created_dt).days
    
    # Avoid division by zero
    safe_age = max(account_age_days, 1)

    # 2. Compute Ratios & Logs (Matching your notebook logic)
    # users_df["ff_ratio"] = ...
    ff_ratio = data.followers_count / (data.friends_count + 1)
    
    # users_df["statuses_per_day"] = ...
    statuses_per_day = data.statuses_count / (safe_age + 1)
    
    # users_df[f"log_{col}"] = np.log1p(...)
    log_followers = np.log1p(data.followers_count)

    # 3. Construct 9-Feature Vector
    # NOTE: 'utc_offset' is usually null in new Twitter API, we pass 0 as placeholder
    # NOTE: 'listed_count' appears twice because it was twice in your provided list
    features = [
        data.statuses_count,       # 1
        data.friends_count,        # 2
        data.favourites_count,     # 3
        data.listed_count,         # 4
        0,                         # 5 (utc_offset)
        log_followers,             # 6 (log_followers_count)
        ff_ratio,                  # 7
        statuses_per_day,          # 8
        data.listed_count          # 9 (Repeated as per your list)
    ]
    
    return np.array(features).reshape(1, -1), ff_ratio, statuses_per_day, account_age_days
# =========================================================
# 4. THE FUSION ENDPOINT
# =========================================================
@app.post("/analyze/fusion")
async def analyze_fusion(data: AccountRequest):
    # --- STEP 1: METADATA & BEHAVIOR (Random Forest) ---
    meta_vec, ff_ratio, spd, age_days = build_metadata_vector(data)
    
    rf_risk = 0.0
    shap_explanation = [] # Initialize as empty list
    
    if rf_model:
        # 1. Prediction
        rf_risk = float(rf_model.predict_proba(meta_vec)[0, 1])
        
        # 2. SHAP Explainability (Robust Version)
        try:
            shap_values = rf_explainer.shap_values(meta_vec)
            if isinstance(shap_values, list):
                vals = shap_values[1] 
            else:
                vals = shap_values

            vals = np.array(vals).reshape(-1)
            
            # The exact 9 features from your list
            known_features = [
               "statuses_count",
               "friends_count",
               "favourites_count",
               "listed_count",
               "utc_offset",
               "log_followers_count",
               "ff_ratio",
               "statuses_per_day",
               "listed_count_dup"
             ]
            
            # Pad if model has extra hidden features
            if len(vals) > len(known_features):
                for k in range(len(known_features), len(vals)):
                    known_features.append(f"Extra_Feature_{k}")
            
            # 4. Get Top 3 Contributors
            top_indices = np.argsort(np.abs(vals))[::-1][:3]

            for i in top_indices:
                # Calculate Impact Direction
                direction = "Increases Risk" if vals[i] > 0 else "Decreases Risk"
                
                # Human-Readable Reasoning
                reason_text = "Abnormal pattern"
                if vals[i] > 0:
                    if "Ratio" in known_features[i]: reason_text = "Suspicious ratio balance"
                    elif "Tweet" in known_features[i]: reason_text = "Activity volume anomaly"
                    elif "Log" in known_features[i]: reason_text = "Follower count mismatch"
                else:
                    reason_text = "Normal behavior observed"

                shap_explanation.append({
                    "feature": known_features[i],
                    "impact": round(float(vals[i]), 4),
                    "direction": direction,
                    "reason": reason_text
                })

        except Exception as e:
            print(f"SHAP Error: {e}")
            shap_explanation = [{
                "feature": "SHAP Calculation",
                "impact": 0,
                "direction": "Error",
                "reason": "Feature dimension mismatch in model"
            }]

    # --- STEP 2: LINGUISTIC ANALYSIS (BERTweet + Textstat) ---
    bert_risk, bert_stats = bertweet_tweet_level_risk(data.tweets)
    text_metrics = analyze_textstat_tweets(data.tweets)



    # --- STEP 3: NETWORK ANOMALY (LightGBM + IsoForest) ---
    # --- STEP 3: NETWORK ANOMALY (MGTAB LightGBM + IsoForest) ---

    lgb_risk = 0.0
    anomaly_score = 0.0

    if lgb_model and iso_model:
      mgtab_vec = build_mgtab_embedding(data)
      lgb_risk = float(lgb_model.predict_proba(mgtab_vec)[0, 1])
      raw_iso = iso_model.score_samples(mgtab_vec)[0]
      anomaly_score = 1.0 / (1.0 + np.exp(raw_iso)) 


    # --- STEP 4: FINAL FUSION LOGIC ---
    # Weighted Ensemble: 40% RF (Behavior), 40% BERT (Content), 20% Network (LGBM)
    # + Bonus penalty for Anomaly Score
    
    base_risk = (rf_risk * 0.4) + (bert_risk * 0.4) + (lgb_risk * 0.2)
    final_risk = min(base_risk + (anomaly_score * 0.15), 1.0) # Cap at 1.0

    # --- STEP 5: DASHBOARD JSON RESPONSE ---
    return {
        "identity": {
            "username": data.username,
            "final_verdict": "BOT" if final_risk > 0.65 else "HUMAN",
            "risk_score": round(final_risk * 100, 2)
        },
        "tweets": data.tweets,
        "model_breakdown": {
            "behavioral_risk_rf": round(rf_risk, 2),
            "linguistic_risk_bert": round(bert_risk, 2),
            "network_risk_lgbm": round(lgb_risk, 2),
            "anomaly_detector": "CRITICAL" if anomaly_score > 0.5 else "NORMAL"
        },
        "explainability": {
            "top_contributing_features_shap": shap_explanation,
            "linguistic_analysis": {
               "bot_language_probability": f"{round(bert_risk*100)}%",
               "avg_tweet_length": text_metrics.get("avg_tweet_length"),
               "reading_ease_mean": text_metrics.get("reading_ease_mean"),
               "grade_variance": text_metrics.get("grade_variance"),

               "bertweet_stats": bert_stats,
               "linguistic_risk": text_metrics.get("linguistic_risk"),
               "interpretation": text_metrics.get("interpretation"),

            },"behavioral_flags": [
                "High Posting Frequency" if spd > 50 else None,
                "Follower/Friend Imbalance" if ff_ratio < 0.1 else None,
                "Default Profile Profile" if data.default_profile else None
            ]
        },
        "visual_data": {
            "timeline_data": { "avg_daily_posts": spd },
            "network_node": {
                "id": data.username,
                "cluster_group": "Botnet_A" if lgb_risk > 0.8 else "Organic",
                "connection_risk": round(lgb_risk, 2)
            }
        }
    }
# class UserIDRequest(BaseModel):
#     user_id: str

# import random
# from datetime import timedelta
# def generate_mock_account(user_id: str) -> AccountRequest:
#     age_days = random.randint(10, 4000)
#     created_at = (datetime.now() - timedelta(days=age_days)).strftime("%Y-%m-%d")

#     return AccountRequest(
#         username=user_id,
#         name=random.choice(["Alex", "Sam", "John", "BotX", "User123"]),
#         description=random.choice([
#             "Love tech and AI",
#             "Crypto | Web3 | NFT",
#             "Just living life",
#             "",
#             "Follow for follow"
#         ]),
#         followers_count=random.randint(0, 50000),
#         friends_count=random.randint(0, 50000),
#         statuses_count=random.randint(0, 200000),
#         favourites_count=random.randint(0, 10000),
#         listed_count=random.randint(0, 500),

#         verified=random.random() < 0.05,
#         protected=random.random() < 0.1,
#         default_profile=random.random() < 0.3,
#         default_profile_image=random.random() < 0.4,
#         is_geo_enabled=random.random() < 0.5,

#         created_at=created_at,
#         account_age_days=age_days
#     )

# @app.post("/analyze/by-userid")
# async def analyze_by_userid(req: UserIDRequest):
#     """
#     Frontend sends only user_id.
#     Backend generates (or later fetches) account data,
#     then forwards it to /analyze/fusion.
#     """

#     # 🔹 Step 1: fetch or mock user data
#     account_data = generate_mock_account(req.user_id)

#     # 🔹 Step 2: call existing fusion logic directly
#     response = await analyze_fusion(account_data)

#     return {
#         "source": "mock-data",
#         "user_id": req.user_id,
#         "analysis": response
#     }

import x
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)