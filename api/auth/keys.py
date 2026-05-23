"""
AERO.AI MVP1 — auth/keys.py
API key validation via X-API-Key header.
Keys are loaded from .env via config.py — never hardcoded here.
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config import settings

# ------------------------------------------------------------------ #
#  Key registry — built at startup from .env values                   #
#  tier: "unlimited" = no rate limit | "starter" = future rate limit  #
# ------------------------------------------------------------------ #
VALID_KEYS: dict[str, dict] = {
    settings.api_key_admin:   {"name": "You (Admin)",  "tier": "unlimited"},
    settings.api_key_friend1: {"name": "Friend 1",     "tier": "starter"},
    settings.api_key_friend2: {"name": "Friend 2",     "tier": "starter"},
}

# Header scheme — client sends:  X-API-Key: <key>
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str = Security(api_key_header)) -> dict:
    """
    FastAPI dependency. Inject into any route that requires auth.

    Keys are loaded from .env at startup via config.py (pydantic-settings).
    To add or rotate keys: update .env and restart the server.

    Returns the key metadata dict {"name": ..., "tier": ...} on success.
    Raises 401 if key is missing.
    Raises 403 if key is present but not recognised.

    Usage in router:
        @router.post("/predict")
        def predict(req: PredictRequest, key_meta: dict = Depends(require_api_key)):
            user_name = key_meta["name"]
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Add header:  X-API-Key: <your-key>",
        )
    if api_key not in VALID_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return VALID_KEYS[api_key]
