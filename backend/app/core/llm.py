from langchain_groq import ChatGroq
from ..core.config import settings

import logging

logger = logging.getLogger(__name__)

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
    """Return the analyst model (Groq) instead of Gemini."""
    logger.info("[LLM] Gemini disabled — using Groq %s for high-reasoning task.", settings.model_analyst)
    return get_llm(model=settings.model_analyst, temperature=temperature)


def get_analyst_llm(temperature: float = 0.0):
    """Returns the analyst model (Groq Llama-70B)."""
    return get_llm(model=settings.model_analyst, temperature=temperature)


def get_editor_llm(temperature: float = 0.0):
    """Returns Groq Llama-8B — fast model for instruction-following editing."""
    return get_llm(model=settings.model_editor, temperature=temperature)


def get_arbitrator_llm(temperature: float = 0.0):
    """Returns the Arbitrator model (Groq Llama-70B)."""
    return get_llm(model=settings.model_analyst, temperature=temperature)
