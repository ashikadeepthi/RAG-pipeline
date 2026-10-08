import fitz
from pathlib import Path


def load_pdf(pdf_path):
    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


if __name__ == "__main__":

    pdf_path = Path(r"D:\RAG_ASHIKA\engineering-rag\data\documents\Review-Aerodynamics of insect flight.pdf")
    output_path = Path(r"D:\RAG_ASHIKA\engineering-rag\data\processed\Review-Aerodynamics of insect flight.txt")

    text = load_pdf(pdf_path)

    output_path.write_text(text, encoding="utf-8")
    print("PDF loaded successfully.")
    print(f"Characters extracted: {len(text)}")
    print(f"Saved to: {output_path}")