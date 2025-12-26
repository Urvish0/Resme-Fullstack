MAX_CHARS = 12000

def enforce_payload_limit(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text

    return (
        text[:MAX_CHARS]
        + "\n\n[Truncated to avoid token overflow]"
    )
