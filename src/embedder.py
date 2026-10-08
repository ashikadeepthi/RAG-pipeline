from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer


if __name__ == "__main__":

    input_path = Path(
        r"D:\RAG_ASHIKA\engineering-rag\data\processed\chunks.json"
    )

    output_path = Path(
        r"D:\RAG_ASHIKA\engineering-rag\data\processed\embeddings.npy"
    )

    print("Loading chunks...")

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(f"Number of chunks: {len(texts)}")

    print("Loading embedding model...")

    model = SentenceTransformer(
        r"D:\RAG_ASHIKA\engineering-rag\models\all-MiniLM-L6-v2"
    )

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings)

    np.save(
        output_path,
        embeddings
    )

    print("Embedding generation completed.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {output_path}")