"""Options data API router."""
from __future__ import annotations

from fastapi import APIRouter

from api.core.errors import ApiError
from api.core.logging import get_logger
from api.schemas.io import OptionsChainRequest, OptionsChainResponse
from common.options_provider import fetch_options_chain

router = APIRouter(prefix="/options", tags=["options"])
logger = get_logger("options")


@router.post("/chain", response_model=OptionsChainResponse)
def get_options_chain(request: OptionsChainRequest) -> OptionsChainResponse:
    """Fetch options chain for a symbol."""

    try:
        result = fetch_options_chain(
            symbol=request.symbol.upper(),
            expiry=request.expiry,
        )
    except ValueError as exc:
        raise ApiError(
            error_code="OPTIONS_FETCH_FAILED",
            message=str(exc),
            status_code=422,
        )
    except Exception as exc:
        raise ApiError(
            error_code="OPTIONS_FETCH_FAILED",
            message=f"Failed to fetch options data: {exc}",
            status_code=500,
        )

    logger.info(
        "options_chain_fetched",
        symbol=result.symbol,
        expiry=result.expiry,
        n_calls=len(result.calls),
        n_puts=len(result.puts),
    )
    return OptionsChainResponse(**result.to_dict())
