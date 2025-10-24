"""
Custom exceptions and error handlers for ChefGenius application.
"""

from typing import Any, Dict, Optional, Union
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from loguru import logger


class ChefGeniusException(Exception):
    """Base exception class for ChefGenius application."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ):
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        super().__init__(self.message)


class ModelNotFoundError(ChefGeniusException):
    """Raised when a required AI model is not found."""
    pass


class ModelLoadError(ChefGeniusException):
    """Raised when an AI model fails to load."""
    pass


class InvalidRecipeError(ChefGeniusException):
    """Raised when recipe data is invalid."""
    pass


class IngredientNotFoundError(ChefGeniusException):
    """Raised when a requested ingredient is not found."""
    pass


class SubstitutionError(ChefGeniusException):
    """Raised when ingredient substitution fails."""
    pass


class VisionProcessingError(ChefGeniusException):
    """Raised when image processing fails."""
    pass


class DatabaseConnectionError(ChefGeniusException):
    """Raised when database connection fails."""
    pass


class ExternalAPIError(ChefGeniusException):
    """Raised when external API calls fail."""
    pass


class RateLimitExceededError(ChefGeniusException):
    """Raised when rate limit is exceeded."""
    pass


class FileUploadError(ChefGeniusException):
    """Raised when file upload fails."""
    pass


async def chef_genius_exception_handler(
    request: Request, exc: ChefGeniusException
) -> JSONResponse:
    """Handle custom ChefGenius exceptions."""
    
    logger.error(
        f"ChefGenius exception: {exc.message}",
        error_code=exc.error_code,
        details=exc.details,
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "message": exc.message,
                "code": exc.error_code,
                "details": exc.details,
                "type": type(exc).__name__,
            }
        },
    )


async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    
    logger.warning(
        f"HTTP exception: {exc.detail}",
        status_code=exc.status_code,
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "code": f"HTTP_{exc.status_code}",
                "type": "HTTPException",
            }
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    
    logger.warning(
        f"Validation error: {exc.errors()}",
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Validation failed",
                "code": "VALIDATION_ERROR",
                "details": exc.errors(),
                "type": "RequestValidationError",
            }
        },
    )


async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    
    logger.error(
        f"Unexpected error: {str(exc)}",
        exc_info=exc,
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "Internal server error",
                "code": "INTERNAL_ERROR",
                "type": "InternalServerError",
            }
        },
    )


def create_http_error(
    status_code: int,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None,
) -> HTTPException:
    """Create a standardized HTTP error."""
    
    return HTTPException(
        status_code=status_code,
        detail={
            "message": message,
            "code": error_code or f"HTTP_{status_code}",
            "details": details or {},
        },
    )


# Common HTTP errors
def not_found_error(message: str = "Resource not found") -> HTTPException:
    return create_http_error(
        status.HTTP_404_NOT_FOUND,
        message,
        error_code="NOT_FOUND"
    )


def unauthorized_error(message: str = "Authentication required") -> HTTPException:
    return create_http_error(
        status.HTTP_401_UNAUTHORIZED,
        message,
        error_code="UNAUTHORIZED"
    )


def forbidden_error(message: str = "Access forbidden") -> HTTPException:
    return create_http_error(
        status.HTTP_403_FORBIDDEN,
        message,
        error_code="FORBIDDEN"
    )


def bad_request_error(message: str = "Bad request") -> HTTPException:
    return create_http_error(
        status.HTTP_400_BAD_REQUEST,
        message,
        error_code="BAD_REQUEST"
    )


def server_error(message: str = "Internal server error") -> HTTPException:
    return create_http_error(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        message,
        error_code="SERVER_ERROR"
    )