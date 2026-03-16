from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    use_rag: bool = True

class Source(BaseModel):
    content: str
    metadata: dict

class ConversationResponse(BaseModel):
    conversation_id: str
    response: str
    sources: List[Source] = []