from pathlib import Path


def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


if __name__ == "__main__":

    input_path = Path(
        r"D:\RAG_ASHIKA\engineering-rag\data\processed\Review-Aerodynamics of insect flight.txt"
    )

    output_path = Path(
        r"D:\RAG_ASHIKA\engineering-rag\data\processed\chunks.txt"
    )

    text = input_path.read_text(
        encoding="utf-8"
    )

    chunks = create_chunks(text)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for i, chunk in enumerate(chunks):

            file.write(f"\n--- CHUNK {i + 1} ---\n")
            file.write(chunk)
            file.write("\n")

    print("Chunking completed.")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved to: {output_path}")