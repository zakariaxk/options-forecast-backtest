from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException, status


class ApiError(HTTPException):
    def __init__(self, error_code: str, message: str, status_code: int = status.HTTP_400_BAD_REQUEST, details: Dict[str, Any] | None = None):
        super().__init__(status_code=status_code, detail={"error_code": error_code, "message": message, "details": details or {}})


def not_found(message: str) -> ApiError:
    return ApiError("NOT_FOUND", message, status.HTTP_404_NOT_FOUND)


def bad_request(message: str) -> ApiError:
    return ApiError("BAD_REQUEST", message, status.HTTP_400_BAD_REQUEST)
