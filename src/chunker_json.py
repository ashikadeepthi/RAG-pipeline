from pathlib import Path
import json


def create_chunks(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


if __name__ == "__main__":

    input_path = Path(
        r"D:\RAG_ASHIKA\engineering-rag\data\processed\Review-Aerodynamics of insect flight.txt"
    )

    output_path = Path(
        r"D:\RAG_ASHIKA\engineering-rag\data\processed\chunks.json"
    )

    text = input_path.read_text(
        encoding="utf-8"
    )

    chunks = create_chunks(text)

    data = []

    for i, chunk in enumerate(chunks):

        data.append({
            "chunk_id": i,
            "source": "Review-Aerodynamics of insect flight.pdf",
            "text": chunk
        })

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("Chunking completed.")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved to: {output_path}")