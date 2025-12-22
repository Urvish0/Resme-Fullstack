from tavily import TavilyClient
from ..core.config import settings

# Initialize Tavily client lazily and safely. Avoid raising at import-time if the
# API key is missing so importing this module doesn't break the whole app.
tavily_client = None
try:
    api_key = getattr(settings, "tavily_api_key", None)
    if api_key:
        tavily_client = TavilyClient(api_key=api_key)
    else:
        tavily_client = None
except Exception:
    tavily_client = None


def get_url_content_from_tavily(url: str) -> str:
    """
    Uses Tavily Search to get the content of a specific URL.
    Returns a human-readable error message when the client isn't configured.
    """
    if tavily_client is None:
        return "Error: Tavily client not configured. Ensure TAVILY_API_KEY is set in .env or the environment."

    try:
        response = tavily_client.get_content(urls=[url])
        if response and response[0]:
            return response[0]
        return f"No content found for URL: {url}"
    except Exception as e:
        return f"Error using Tavily to get URL content: {e}"

# Backwards-compatibility: allow callers to use `.invoke(...)` on this function.
try:
    get_url_content_from_tavily.invoke = get_url_content_from_tavily
except Exception:
    pass