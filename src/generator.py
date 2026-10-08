import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_DIR / ".env"

load_dotenv(ENV_FILE)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        f"OPENAI_API_KEY not found in {ENV_FILE}"
    )

client = OpenAI(api_key=api_key)


# --------------------------------------------------
# Generate answer
# --------------------------------------------------

def generate_answer(query, retrieved_chunks):

    context = "\n\n".join(
        [
            f"""
SOURCE: {chunk['source']}
CHUNK ID: {chunk['chunk_id']}

{chunk['text']}
"""
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are an engineering question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough information,
   say that the information is not available in the
   provided documents.
4. Give a clear and technically accurate answer.

USER QUESTION:
{query}

CONTEXT:
{context}
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text


# --------------------------------------------------
# Test the generator independently
# --------------------------------------------------

if __name__ == "__main__":

    test_query = (
        "What are the major aerodynamic mechanisms "
        "involved in insect flight?"
    )

    test_chunks = [
        {
            "chunk_id": 1,
            "source": "Review-Aerodynamics of insect flight.pdf",
            "text": (
                "Insect flight involves several unsteady "
                "aerodynamic mechanisms including leading-edge "
                "vortex formation and rotational effects."
            )
        }
    ]

    print("Testing generator...")

    answer = generate_answer(
        test_query,
        test_chunks
    )

    print("\n" + "=" * 80)
    print("GENERATED ANSWER")
    print("=" * 80)
    print(answer)