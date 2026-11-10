
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List

from app.rag.generation import generate_answer


router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=1000,
        description="Question about ZyVoxa"
    )


class SourceResponse(BaseModel):
    source: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest):

    try:
        answer, results = generate_answer(
            question=body.question,
            llm=request.app.state.llm,
            embeddings=request.app.state.embeddings,
            client=request.app.state.qdrant
        )

        sources = []

        for point in results:
            metadata = point.payload.get("metadata", {})

            sources.append(
                SourceResponse(
                    source=metadata.get("source", "Unknown"),
                    page=metadata.get("page", 0) + 1,
                    score=round(point.score, 4)
                )
            )

        return ChatResponse(
            answer=answer,
            sources=sources
        )

    except Exception:
        # Log the actual exception server-side in a later hardening step.
        raise HTTPException(
            status_code=500,
            detail="The assistant could not process your question."
        )
