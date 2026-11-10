
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.rag.retrieval import (
    get_embedding_model,
    get_qdrant_client
)
from app.rag.generation import get_llm


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting ZyVoxa RAG API...")

    # Initialize shared resources once
    app.state.embeddings = get_embedding_model()
    app.state.qdrant = get_qdrant_client()
    app.state.llm = get_llm()

    # Verify the Qdrant connection
    app.state.qdrant.get_collection(
        "zyvoxa_knowledge_base"
    )

    print("Embedding model initialized.")
    print("Qdrant connected.")
    print("Groq initialized.")
    print("ZyVoxa API is ready.")

    yield

    print("Shutting down ZyVoxa RAG API...")


app = FastAPI(
    title="ZyVoxa Intelligence API",
    description="Company knowledge assistant powered by RAG.",
    version="1.0.0",
    lifespan=lifespan
)


app.include_router(
    router,
    prefix="/api",
    tags=["Chat"]
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "ZyVoxa Intelligence API"
    }
