import hashlib
import json

def make_request_fingerprint(payload: dict) -> str:
    """
    Creates a stable hash for identical requests.
    Order-independent and deterministic.
    """
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
