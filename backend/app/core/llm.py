from langchain_groq import ChatGroq
from ..core.config import settings


def get_llm(
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
):
    """
    Factory function to create an LLM instance.
    Centralized so UI / workflows never create models directly.
    """
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=settings.groq_api_key,
    )
