from langchain_groq import ChatGroq
from ..core.config import settings

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini (Google AI) — used for Analyst, Auditor, and Arbitrator roles.
# Falls back to Groq Llama-70B if no GOOGLE_API_KEY is configured.
# ---------------------------------------------------------------------------

_gemini_available = False

try:
    if settings.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _gemini_available = True
        logger.info(
            "[LLM] Gemini Pro available — using %s for analyst/arbitrator roles.",
            settings.model_gemini,
        )
    else:
        logger.warning(
            "[LLM] GOOGLE_API_KEY not set — falling back to Groq for analyst."
        )
except ImportError:
    logger.warning(
        "[LLM] langchain-google-genai not installed — falling back to Groq for analyst."
    )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def get_llm(
    model: str | None = None,
    temperature: float = 0.0,
):
    """
    Factory function to create a Groq LLM instance.
    Centralized so UI / workflows never create models directly.
    """
    if model is None:
        model = settings.model_editor

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=settings.groq_api_key,
        max_retries=6,  # Higher retries for free tier stability
    )


def get_gemini_llm(temperature: float = 0.0):
    """Return a Gemini Pro LLM instance for high-reasoning tasks."""
    if not _gemini_available:
        logger.warning("[LLM] Gemini not available, falling back to Groq analyst.")
        return get_llm(model=settings.model_analyst, temperature=temperature)

    return ChatGoogleGenerativeAI(
        model=settings.model_gemini,
        temperature=temperature,
        google_api_key=settings.google_api_key,
        max_retries=3,
    )


def get_analyst_llm(temperature: float = 0.0):
    """
    Returns the best available model for analysis/reasoning.
    Gemini Pro if available, otherwise Groq Llama-70B.
    """
    if _gemini_available:
        return get_gemini_llm(temperature=temperature)
    return get_llm(model=settings.model_analyst, temperature=temperature)


def get_editor_llm(temperature: float = 0.0):
    """Returns Groq Llama-8B — fast model for instruction-following editing."""
    return get_llm(model=settings.model_editor, temperature=temperature)


def get_arbitrator_llm(temperature: float = 0.0):
    """
    Returns the Arbitrator model — Gemini Pro at temp=0 for deterministic scoring.
    Falls back to Groq Llama-70B if Gemini is unavailable.
    """
    return get_gemini_llm(temperature=temperature)
