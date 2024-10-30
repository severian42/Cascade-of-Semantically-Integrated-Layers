from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .api_config import Settings

settings = Settings()
security = HTTPBearer()

async def verify_api_key(request: Request):
    try:
        auth = await security(request)
        if auth.credentials != settings.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )