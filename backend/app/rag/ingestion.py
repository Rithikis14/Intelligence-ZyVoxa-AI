from pathlib import Path
from uuid import uuid5, NAMESPACE_URL
from qdrant_client import QdrantClient, models
from langchain_community.document_loaders import PyPDFLoader
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "zyvoxa_knowledge_base"

VECTOR_SIZE = 384
BATCH_SIZE = 64
# Resolve the project root from this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[3]

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "ZyVoxa_Company_Knowledge_Base_v1.pdf"
)


def load_pdf():
    """Load the company knowledge base PDF."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base PDF not found: {PDF_PATH}"
        )

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    if not documents:
        raise ValueError("The PDF contains no extractable pages.")

    return documents




def clean_documents(documents):
    """Clean extracted PDF text while preserving metadata."""

    cleaned_documents = []

    for document in documents:
        text = document.page_content

        # Remove unwanted control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

        # Remove spaces before line breaks
        text = re.sub(r" +\n", "\n", text)

        # Reduce excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        text = text.strip()

        if text:
            cleaned_documents.append(
                Document(
                    page_content=text,
                    metadata=document.metadata.copy()
                )
            )

    return cleaned_documents

def split_documents(documents):
    """Split cleaned documents into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)

    return chunks

def get_embedding_model():
    """Initialize the local BGE embedding model."""

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


def generate_embeddings(chunks, embeddings):
    """Convert document chunks into numerical vectors."""

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embeddings.embed_documents(texts)

    return vectors


def get_qdrant_client():
    """Connect to the local Qdrant instance."""

    client = QdrantClient(
        url=QDRANT_URL
    )

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

        print("Qdrant collection created.")

    else:
        print("Qdrant collection already exists.")

    collection_info = client.get_collection(COLLECTION_NAME)

    vector_config = collection_info.config.params.vectors

    if vector_config.size != VECTOR_SIZE:
        raise ValueError(
            f"Collection expects {vector_config.size} dimensions, "
            f"but our model generates {VECTOR_SIZE}."
        )

    return client

def store_documents(chunks, vectors, client):
    """Store document vectors and their payloads in Qdrant."""

    if len(chunks) != len(vectors):
        raise ValueError(
            "Number of chunks and vectors must match."
        )

    points = []

    for chunk, vector in zip(chunks, vectors):

        identity = (
            f"{chunk.metadata.get('source', '')}:"
            f"{chunk.metadata.get('page', '')}:"
            f"{chunk.metadata.get('start_index', '')}:"
            f"{chunk.page_content}"
        )

        point_id = str(
            uuid5(NAMESPACE_URL, identity)
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
            )
        )

    for start in range(0, len(points), BATCH_SIZE):

        batch = points[start:start + BATCH_SIZE]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True
        )

        print(
            f"Uploaded {min(start + BATCH_SIZE, len(points))}"
            f"/{len(points)} points"
        )

    return len(points)



if __name__ == "__main__":

    # Stage 1: Load PDF
    documents = load_pdf()

    # Stage 2: Clean and split
    cleaned_documents = clean_documents(documents)
    chunks = split_documents(cleaned_documents)

    # Stage 3: Generate embeddings
    embeddings = get_embedding_model()
    vectors = generate_embeddings(chunks, embeddings)

    # Stage 4: Store in Qdrant
    client = get_qdrant_client()

    stored_count = store_documents(
        chunks,
        vectors,
        client
    )

    print("\n========== INGESTION COMPLETE ==========")

    print(f"PDF pages: {len(documents)}")
    print(f"Cleaned pages: {len(cleaned_documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total vectors: {len(vectors)}")
    print(f"Stored points: {stored_count}")

    print("\nQdrant collection:")
    print(COLLECTION_NAME)