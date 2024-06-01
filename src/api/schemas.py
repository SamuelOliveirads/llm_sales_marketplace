from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    stage: str = "main"


class QueryResponse(BaseModel):
    message: str
    rag_content: Optional[str] = None
