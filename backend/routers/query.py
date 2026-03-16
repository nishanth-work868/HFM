import logging

from fastapi import APIRouter, HTTPException, status
from models.schemas import QueryRequest, ConversationResponse
from services.rag_service import handle_query

router = APIRouter()
logger = logging.getLogger("query_router")

@router.post("/query", response_model=ConversationResponse)
def query_rag(request: QueryRequest):
    try:
        return handle_query(request)
    except ValueError as exc:
        logger.warning("Query rejected: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Upstream query dependency failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected query failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while processing query.",
        ) from exc
