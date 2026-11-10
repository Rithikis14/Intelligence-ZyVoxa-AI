import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "zyvoxa_knowledge_base"

TOP_K = 5

MIN_RELEVANCE_SCORE = float(
    os.getenv("MIN_RELEVANCE_SCORE", "0.60")
)

def get_embedding_model():
    """Initialize the embedding model."""

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def get_qdrant_client():
    """Connect to Qdrant."""

    return QdrantClient(url=QDRANT_URL)


def retrieve_documents(
    question,
    embeddings,
    client,
    top_k=TOP_K
):
    query_vector = embeddings.embed_query(question)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    relevant_results = [
        point
        for point in response.points
        if point.score >= MIN_RELEVANCE_SCORE
    ]

    return relevant_results


def display_results(results):

    print("\n========== RETRIEVAL RESULTS ==========")

    for index, point in enumerate(results, start=1):

        payload = point.payload
        metadata = payload["metadata"]

        print(f"\nResult {index}")
        print(f"Similarity score: {point.score:.4f}")
        print(f"Source page: {metadata.get('page')}")
        print(f"Content:\n{payload['page_content']}")
        print("-" * 60)


if __name__ == "__main__":

    embeddings = get_embedding_model()
    client = get_qdrant_client()

    question = input("\nAsk something about ZyVoxa: ")

    results = retrieve_documents(
        question,
        embeddings,
        client
    )

    display_results(results)
