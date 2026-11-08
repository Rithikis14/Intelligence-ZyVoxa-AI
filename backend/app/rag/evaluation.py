
from app.rag.retrieval import (
    get_embedding_model,
    get_qdrant_client,
    retrieve_documents
)


TEST_QUESTIONS = [
    {
        "question": "What services does ZyVoxa provide?",
        "category": "answerable"
    },
    {
        "question": "What is ZyVoxa LaunchPad?",
        "category": "answerable"
    },
    {
        "question": "Explain the VoxaVault project.",
        "category": "answerable"
    },
    {
        "question": "What is the weather in Chennai today?",
        "category": "out_of_domain"
    },
    {
        "question": "Explain quantum physics.",
        "category": "out_of_domain"
    },
    {
        "question": "What is ZyVoxa's annual revenue?",
        "category": "potentially_unanswerable"
    }
]


def evaluate_retrieval():

    embeddings = get_embedding_model()
    client = get_qdrant_client()

    for test in TEST_QUESTIONS:

        question = test["question"]
        category = test["category"]

        results = retrieve_documents(
            question,
            embeddings,
            client,
            top_k=5
        )

        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print(f"Category: {category}")

        if not results:
            print("No results returned.")
            continue

        print("\nTop retrieved results:")

        for index, point in enumerate(results[:3], start=1):

            metadata = point.payload["metadata"]

            print(f"\nResult {index}")
            print(f"Score: {point.score:.4f}")
            print(f"Page: {metadata.get('page', 0) + 1}")
            print(
                f"Text: "
                f"{point.payload['page_content'][:250]}"
            )

        print(
            f"\nHighest similarity score: "
            f"{results[0].score:.4f}"
        )


if __name__ == "__main__":
    evaluate_retrieval()
