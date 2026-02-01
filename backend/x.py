import pandas as pd
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel
from main import app, analyze_fusion, AccountRequest

USERS_DF = pd.read_csv("../user_hidden.csv")
TWEETS_DF = pd.read_csv("../tweet_hidden.csv")
def parse_twitter_date(val):
    try:
        if isinstance(val, str) and val.endswith("L"):
            return datetime.fromtimestamp(int(val[:-1]) / 1000)
        return pd.to_datetime(val, utc=True)
    except:
        return None
def get_user_by_username(username: str):
    row = USERS_DF[USERS_DF["screen_name"] == username]
    if row.empty:
        raise HTTPException(404, "User not found")
    return row.iloc[0]
def get_user_tweets(user_id, limit=10):
    tweets = TWEETS_DF[TWEETS_DF["user_id"] == user_id] \
                .sort_values("created_at", ascending=False) \
                .head(limit)

    lines = []
    for _, t in tweets.iterrows():
        lines.append(f"[{t['created_at']}] {t['text']}")

    return "\n".join(lines)
def build_account_request(username: str) -> AccountRequest:
    user = get_user_by_username(username)

    created_dt = parse_twitter_date(user["created_at"])

# 🔥 FORCE timezone removal (THIS IS THE KEY)
    if isinstance(created_dt, pd.Timestamp):
       created_dt = created_dt.tz_convert(None)

    age_days = max((datetime.now() - created_dt).days, 1)

    raw = get_user_tweets(user["id"])
    tweets_text = raw.split("\n") if raw else []

    return AccountRequest(
        username=user["screen_name"],
        name=user.get("name", ""),
        tweets=tweets_text, 
        followers_count=int(user["followers_count"]),
        friends_count=int(user["friends_count"]),
        statuses_count=int(user["statuses_count"]),
        favourites_count=int(user["favourites_count"]),
        listed_count=int(user["listed_count"]),

        verified=bool(user["verified"]),
        protected=bool(user["protected"]),
        default_profile=bool(user["default_profile"]),
        default_profile_image=bool(user["default_profile_image"]),
        is_geo_enabled=bool(user["geo_enabled"]),

        created_at=created_dt.isoformat(),
        account_age_days=age_days
    )
class UsernameRequest(BaseModel):
    username: str

@app.post("/analyze/by-username")
async def analyze_by_username(req: UsernameRequest):
    account_data = build_account_request(req.username)
    result = await analyze_fusion(account_data)
    return result
