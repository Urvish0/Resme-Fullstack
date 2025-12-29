from enum import Enum

class ErrorType(str, Enum):
    USER_ERROR = "user_error"
    SYSTEM_ERROR = "system_error"
    RETRYABLE_ERROR = "retryable_error"