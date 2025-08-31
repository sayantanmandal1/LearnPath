"""
Global error handling middleware
"""
import traceback
from typing import Any, Dict

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.exceptions import APIException

logger = structlog.get_logger()


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle requests and catch exceptions"""
        try:
            response = await call_next(request)
            return response
        except APIException as exc:
            # Handle custom API exceptions
            logger.warning(
                "API exception occurred",
                path=request.url.path,
                method=request.method,
                status_code=exc.status_code,
                detail=exc.detail,
                error_code=exc.error_code,
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.error_code or "API_ERROR",
                        "message": exc.detail,
                        "details": exc.details if hasattr(exc, 'details') else None,
                    }
                }
            )
        except ValueError as exc:
            # Handle validation errors
            logger.warning(
                "Validation error occurred",
                path=request.url.path,
                method=request.method,
                error=str(exc),
            )
            
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": str(exc),
                    }
                }
            )
        except Exception as exc:
            # Handle unexpected errors
            logger.error(
                "Unexpected error occurred",
                path=request.url.path,
                method=request.method,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected error occurred. Please try again later.",
                    }
                }
            )