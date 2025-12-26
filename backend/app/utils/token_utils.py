import math

AVG_CHARS_PER_TOKEN = 4

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / AVG_CHARS_PER_TOKEN)


def enforce_token_limit(
    text: str,
    max_tokens: int,
    hard_truncate: bool = True
) -> str:
    """
    Ensures text stays within max token limit.
    Truncates safely if needed.
    """
    if not text:
        return text

    tokens = estimate_tokens(text)

    if tokens <= max_tokens:
        return text

    # Hard truncate by character count
    max_chars = max_tokens * AVG_CHARS_PER_TOKEN
    truncated = text[:max_chars]

    if hard_truncate:
        return truncated

    # Optional: later we’ll replace this with summarization
    return truncated
