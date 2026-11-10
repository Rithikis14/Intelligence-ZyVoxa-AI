
from app.rag.retrieval import (
    get_embedding_model,
    get_qdrant_client,
    COLLECTION_NAME
)

TEST_QUESTIONS = [
    # Relevant questions
    ("What services does ZyVoxa offer?", "IN-DOMAIN"),
    ("What is ZyVoxa LaunchPad?", "IN-DOMAIN"),
    ("Tell me about VoxaVault.", "IN-DOMAIN"),
    ("What technologies does ZyVoxa use?", "IN-DOMAIN"),

    # Company-related but potentially unanswered
    ("What is ZyVoxa's annual revenue?", "UNKNOWN-ANSWER"),

    # Unrelated questions
    ("What is the weather today?", "OUT-OF-DOMAIN"),
    ("Teach me quantum physics.", "OUT-OF-DOMAIN"),
    ("How do I learn React from scratch?", "OUT-OF-DOMAIN"),
]


def evaluate_retrieval():

    print("Initializing retrieval evaluation...")

    embeddings = get_embedding_model()
    client = get_qdrant_client()

    for question, category in TEST_QUESTIONS:

        query_vector = embeddings.embed_query(question)

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5,
            with_payload=True
        )

        print("\n" + "=" * 70)
        print(f"Category: {category}")
        print(f"Question: {question}")
        print("-" * 70)

        if not response.points:
            print("No results found.")
            continue

        for index, point in enumerate(response.points, start=1):

            metadata = point.payload.get("metadata", {})

            source = metadata.get("source", "Unknown")
            page = metadata.get("page", 0) + 1

            print(
                f"Result {index} | "
                f"Score: {point.score:.4f} | "
                f"Page: {page} | "
                f"Source: {source}"
            )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    evaluate_retrieval()
