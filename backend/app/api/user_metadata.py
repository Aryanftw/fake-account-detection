from fastapi import APIRouter
from random import randint, choice
from app.schemas.user import UsernameRequest, UserMetadataResponse

router = APIRouter()

@router.post("/user/metadata", response_model=UserMetadataResponse)
def fetch_user_metadata(req: UsernameRequest):
    """
    Mock Twitter/X user metadata service.
    Can be replaced with real API later.
    """

    username = req.username.lower()

    # ---- Simulated but realistic values ----
    followers = randint(10, 5000)
    following = randint(50, 3000)
    tweets = randint(50, 20000)
    listed = randint(0, 50)

    metadata = {
        "username": username,
        "followers_count": followers,
        "following_count": following,
        "tweet_count": tweets,
        "listed_count": listed,
        "verified": choice([False, False, False, True]),
        "has_profile_pic": choice([True, True, False]),
        "account_age_days": randint(30, 4000)
    }

    return metadata
