from fastapi import APIRouter, UploadFile, File, HTTPException, status
from services.rag_service import upload_document, clear_index
from config import MAX_FILE_SIZE
import logging

logger = logging.getLogger("upload_router")

router = APIRouter()

@router.post("/upload")
def upload(file: UploadFile = File(...)):
    # Ensure the file isn't larger than the maximum allowed size before processing
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        logger.warning(f"Upload rejected: File size {file_size} exceeds maximum {MAX_FILE_SIZE}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB."
        )
        
    try:
        return upload_document(file)
    except ValueError as exc:
        logger.warning("Upload rejected: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Upstream upload dependency failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected upload failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while processing upload.",
        ) from exc

@router.delete("/clear-index")
def clear():
    """Clear all stored vectors from the backing store. Use before re-uploading documents."""
    return clear_index()
