from fastapi import FastAPI
from app.api.user_metadata import router as user_router

app = FastAPI(title="User Metadata Service")

app.include_router(user_router)
