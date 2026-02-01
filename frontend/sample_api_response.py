{
  "identity": {
    "username": "CarloMichettoni",
    "final_verdict": "HUMAN",
    "risk_score": 49.11
  },
  "model_breakdown": {
    "behavioral_risk_rf": 1.0,
    "linguistic_risk_bert": 0.0,
    "network_risk_lgbm": 0.02,
    "anomaly_detector": "CRITICAL"
  },
  "explainability": {
    "top_contributing_features_shap": [
      {
        "feature": "log_followers_count",
        "impact": 0.2058,
        "direction": "Increases Risk",
        "reason": "Abnormal follower count pattern detected"
      },
      {
        "feature": "utc_offset",
        "impact": -0.2058,
        "direction": "Decreases Risk",
        "reason": "Normal timezone behavior observed"
      },
      {
        "feature": "Extra_Feature_12",
        "impact": -0.1542,
        "direction": "Decreases Risk",
        "reason": "Normal behavior pattern detected"
      },
      {
        "feature": "posting_frequency",
        "impact": 0.1234,
        "direction": "Increases Risk",
        "reason": "Unusually high posting frequency"
      },
      {
        "feature": "account_age",
        "impact": -0.0987,
        "direction": "Decreases Risk",
        "reason": "Established account with normal age"
      }
    ],
    "linguistic_analysis": {
      "bot_language_probability": "5%",
      "avg_tweet_length": 124.5,
      "reading_ease_mean": 67.71,
      "grade_variance": 8.32,
      "bertweet_stats": {},
      "linguistic_risk": "Low Bot Likelihood",
      "interpretation": "Writing shows natural human variation with diverse vocabulary and sentence structure"
    },
    "behavioral_flags": [
      null,
      "Follower/Friend Imbalance",
      "Default Profile Profile"
    ]
  },
  "visual_data": {
    "timeline_data": {
      "avg_daily_posts": 0.019699629412912034
    },
    "network_node": {
      "id": "CarloMichettoni",
      "cluster_group": "Organic",
      "connection_risk": 0.02
    }
  }
}