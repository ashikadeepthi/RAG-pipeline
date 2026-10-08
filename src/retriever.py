from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "chunks.json"
)

EMBEDDINGS_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "embeddings.npy"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Load chunks
# --------------------------------------------------

print("Loading chunks...")

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------

print("Loading embeddings...")

embeddings = np.load(
    EMBEDDINGS_FILE
)

print(f"Embedding shape: {embeddings.shape}")


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    str(MODEL_PATH)
)


# --------------------------------------------------
# Retrieval function
# --------------------------------------------------

def retrieve(query, top_k=5):

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    # Normalize query
    query_embedding = (
        query_embedding
        / np.linalg.norm(query_embedding)
    )

    # Normalize document embeddings
    normalized_embeddings = (
        embeddings
        / np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )
    )

    # Cosine similarity
    similarities = (
        normalized_embeddings
        @ query_embedding
    )

    # Get highest similarity scores
    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "chunk_id": chunks[index]["chunk_id"],
            "source": chunks[index]["source"],
            "score": float(similarities[index]),
            "text": chunks[index]["text"]
        })

    return results


# --------------------------------------------------
# Test retrieval
# --------------------------------------------------

if __name__ == "__main__":

    query = input(
        "\nEnter your engineering question: "
    )

    results = retrieve(
        query,
        top_k=5
    )

    print("\n" + "=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {i}"
        )

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print(
            f"Similarity: {result['score']:.4f}"
        )

        print(
            f"Source: {result['source']}"
        )

        print("\nText:")

        print(
            result["text"][:1000]
        )

        print("-" * 80)