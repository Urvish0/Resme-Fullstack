from fastapi import HTTPException
from .errors import ErrorType

class APIError(HTTPException):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: ErrorType,
        details: dict | None = None,
    ):
        super().__init__(
            status_code=status_code,
            detail={
                "error": message,
                "type": error_type,
                "details": details,
            },
        )

class UserInputError(APIError):
    def __init__(self, message:str, details: dict | None = None):
        super().__init__(
            status_code=400,
            message=message,
            error_type=ErrorType.USER_ERROR,
            details=details,
        )
        
class SystemFailure(APIError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=500,
            message=message,
            error_type=ErrorType.SYSTEM_ERROR,
            details=details,
        )      
        
class RetryableFailure(APIError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=503,
            message=message,
            error_type=ErrorType.RETRYABLE_ERROR,
            details=details,
        )