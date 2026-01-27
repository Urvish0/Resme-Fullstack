# from tavily import TavilyClient
# from ..core.config import settings
# import logging

# logger = logging.getLogger(__name__)


# # Initialize Tavily client lazily and safely. Avoid raising at import-time if the
# # API key is missing so importing this module doesn't break the whole app.
# tavily_client = None
# try:
#     api_key = getattr(settings, "tavily_api_key", None)
#     if api_key:
#         tavily_client = TavilyClient(api_key=api_key)
#     else:
#         tavily_client = None
# except Exception:
#     tavily_client = None

# # def get_url_content_from_tavily(url: str) -> str:
# #     """
# #     Uses Tavily Extract API to get clean page content
# #     """
# #     if tavily_client is None:
# #         return "Error: Tavily client not configured. Ensure TAVILY_API_KEY is set."

# #     try:
# #         response = tavily_client.extract(
# #             urls=[url],
# #             # include_raw_content=False
# #         )

# #         if not response or "results" not in response:
# #             return ""

# #         results = response.get("results", [])
# #         if not results:
# #             return ""

# #         # Tavily returns structured content
# #         return results[0].get("content", "")

# #     except Exception as e:
# #         return f"Error using Tavily extract: {e}"

# def get_url_content_from_tavily(url: str) -> str:
#     """
#     Uses Tavily Extract API to get clean page content
#     """
#     if tavily_client is None:
#         return ""

#     try:
#         response = tavily_client.extract(
#             urls=[url],
#             include_raw_content=True,
#         )

#         # Defensive checks
#         if not isinstance(response, dict):
#             return ""

#         results = response.get("results")
#         if not results or not isinstance(results, list):
#             return ""

#         first = results[0]

#         # Prefer cleaned content, fallback to raw
#         content = first.get("content") or first.get("raw_content") or ""

#         return content.strip()

#     except Exception as e:
#         logger.exception("[TAVILY] Extract failed")
#         return ""


# # Backwards-compatibility: allow callers to use `.invoke(...)` on this function.
# try:
#     get_url_content_from_tavily.invoke = get_url_content_from_tavily
# except Exception:
#     pass

from tavily import TavilyClient
from ..core.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Tavily client with better error handling
tavily_client = None


def init_tavily_client():
    """Initialize Tavily client with proper error logging"""
    global tavily_client
    try:
        api_key = getattr(settings, "tavily_api_key", None)
        if not api_key:
            logger.error("[TAVILY] API key not found in settings")
            return False

        tavily_client = TavilyClient(api_key=api_key)
        logger.info("[TAVILY] Client initialized successfully")
        return True

    except Exception as e:
        logger.exception(f"[TAVILY] Failed to initialize client: {e}")
        tavily_client = None
        return False


# Initialize on module import
init_tavily_client()


def get_url_content_from_tavily(url: str) -> str:
    """
    Uses Tavily Extract API to get clean page content
    """
    if tavily_client is None:
        logger.error("[TAVILY] Client not initialized")
        if not init_tavily_client():
            return ""

    try:
        logger.info(f"[TAVILY] Extracting content from: {url}")

        # Try with default parameters first
        response = tavily_client.extract(
            urls=[url],
            # Try without include_raw_content first
        )

        logger.info(f"[TAVILY] Response type: {type(response)}")
        logger.info(
            f"[TAVILY] Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}"
        )

        # Defensive checks
        if not isinstance(response, dict):
            logger.error(f"[TAVILY] Response is not a dict: {type(response)}")
            return ""

        results = response.get("results")
        if not results or not isinstance(results, list):
            logger.error(f"[TAVILY] No results in response or not a list: {results}")
            return ""

        first = results[0]
        logger.info(f"[TAVILY] First result keys: {list(first.keys())}")

        # Try to get content in multiple ways
        content = ""
        if "content" in first:
            content = first.get("content", "")
            logger.info(f"[TAVILY] Got 'content' field: {len(content)} chars")
        elif "raw_content" in first:
            content = first.get("raw_content", "")
            logger.info(f"[TAVILY] Got 'raw_content' field: {len(content)} chars")
        elif "text" in first:
            content = first.get("text", "")
            logger.info(f"[TAVILY] Got 'text' field: {len(content)} chars")

        if content:
            content = content.strip()
            logger.info(f"[TAVILY] Final content length: {len(content)}")
            return content
        else:
            logger.error("[TAVILY] No content found in response")
            # Log the entire first result for debugging
            logger.debug(f"[TAVILY] First result: {first}")
            return ""

    except Exception as e:
        logger.exception(f"[TAVILY] Extract failed for URL {url}: {e}")
        return ""


# Backwards-compatibility: allow callers to use `.invoke(...)` on this function.
try:
    get_url_content_from_tavily.invoke = get_url_content_from_tavily
except Exception:
    pass
