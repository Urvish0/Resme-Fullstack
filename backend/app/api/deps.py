import json
import urllib.request
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

# Cache for JWKS keys
_jwks_cache: Optional[dict] = None


def get_supabase_jwks() -> Optional[dict]:
    """Fetch JSON Web Key Set from Supabase to verify ES256 signatures."""
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache

    if not settings.supabase_url:
        return None

    try:
        # Supabase v2+ exposes JWKS at this endpoint
        # Format: https://<project>.supabase.co/auth/v1/.well-known/jwks.json
        url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
        logger.info(f"Fetching JWKS from {url}")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            _jwks_cache = json.loads(response.read().decode())
            logger.info(f"JWKS fetched successfully. Keys: {len(_jwks_cache.get('keys', []))}")
            return _jwks_cache
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        return None


def _get_signing_key(token: str) -> Optional[dict]:
    """Find the correct signing key from JWKS based on the token's kid header."""
    jwks = get_supabase_jwks()
    if not jwks:
        return None

    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")

        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key

        logger.warning(f"No matching kid found in JWKS. Token kid: {kid}")
        return None
    except Exception as e:
        logger.error(f"Error parsing token header: {e}")
        return None


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Optional[dict]:
    """Verify Supabase JWT Token. Supports both ES256 (JWKS) and HS256 (secret)."""
    if not credentials:
        return None

    token = credentials.credentials

    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "")
    except Exception:
        logger.error("Failed to read JWT header")
        raise HTTPException(status_code=401, detail="Malformed token")

    # --- ES256 path (asymmetric, uses JWKS) ---
    if alg.startswith("ES"):
        signing_key = _get_signing_key(token)
        if not signing_key:
            logger.warning("Could not find signing key in JWKS. Falling back to unauthenticated.")
            return None

        try:
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=["ES256", "ES384", "ES512"],
                options={"verify_aud": False},
            )
            return payload
        except jwt.JWTError as e:
            logger.error(f"JWT ES256 verification failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    # --- HS256 path (symmetric, uses JWT secret) ---
    supabase_jwt_secret = getattr(settings, "supabase_jwt_secret", None)
    if not supabase_jwt_secret:
        logger.warning("No SUPABASE_JWT_SECRET found. Assuming unauthenticated.")
        return None

    try:
        payload = jwt.decode(
            token,
            supabase_jwt_secret,
            algorithms=["HS256", "HS384", "HS512"],
            options={"verify_aud": False},
        )
        return payload
    except jwt.JWTError as e:
        logger.error(f"JWT HS256 verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


def get_current_user(payload: dict = Security(verify_token)) -> str:
    """Dependency that returns the user ID from the verified token."""
    if payload and "sub" in payload:
        return payload["sub"]

    # Fallback to anonymous for now
    return "anonymous_user"
