from pydantic import BaseModel

class UsernameRequest(BaseModel):
    username: str

class UserMetadataResponse(BaseModel):
    username: str
    followers_count: int
    following_count: int
    tweet_count: int
    listed_count: int
    verified: bool
    has_profile_pic: bool
    account_age_days: int
