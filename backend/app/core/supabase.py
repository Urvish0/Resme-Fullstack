import logging
from supabase import create_client, Client
from .config import settings

logger = logging.getLogger(__name__)

# Supabase configuration — loaded from .env via pydantic-settings
SUPABASE_URL = settings.supabase_url
SUPABASE_KEY = settings.supabase_service_role_key

supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("Supabase credentials missing. Long-term memory will be disabled.")

def get_supabase() -> Client:
    """Returns the initialized Supabase client."""
    return supabase

class SupabaseService:
    @staticmethod
    def save_resume_version(user_id: str, content: str, score: int = None, keywords: list = None):
        """Saves a new version of the resume to Supabase."""
        if not supabase:
            return None
        try:
            data = {
                "user_id": user_id,
                "content": content,
                "ats_score": score,
                "keywords": keywords
            }
            result = supabase.table("resumes").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Supabase Error (save_resume): {e}")
            return None

    @staticmethod
    def get_latest_resume(user_id: str):
        """Fetches the latest resume for a user."""
        if not supabase:
            return None
        try:
            result = supabase.table("resumes") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Supabase Error (get_latest): {e}")
            return None

    @staticmethod
    def get_score_history(user_id: str):
        """Fetches ATS score history for a user."""
        if not supabase:
            return []
        try:
            result = supabase.table("resumes") \
                .select("ats_score, created_at") \
                .eq("user_id", user_id) \
                .order("created_at", desc=False) \
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Supabase Error (get_history): {e}")
            return []
