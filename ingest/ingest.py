# ingest.py — Updated Day 2
# Chunking strategy: sentence boundary (winner from benchmark)

import PyPDF2
import json
import os
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

PDF_PATHS = [
    "data/fia_technical_regulations_2024.pdf",
    "data/fia_sporting_regulations_2024.pdf",
]

SENTENCES_PER_CHUNK = 5   # tweak this if needed


def extract_text_from_pdf(pdf_paths: list) -> str:
    text = ""
    for path in pdf_paths:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            print(f"📄 {path} — {len(reader.pages)} pages")
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    print(f"✅ {len(text):,} characters extracted")
    return text


def chunk_text(text: str) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
        group = sentences[i:i + SENTENCES_PER_CHUNK]
        chunk = " ".join(group).strip()
        if chunk:
            chunks.append(chunk)
    print(f"✅ {len(chunks)} chunks created (sentence boundary, {SENTENCES_PER_CHUNK} sentences each)") 
    return chunks


def save_chunks(chunks: list[str], output_path="data/chunks.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = [{"id": i, "text": c, "source": "multi"} for i, c in enumerate(chunks)]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved → {output_path}")


if __name__ == "__main__":
    text = extract_text_from_pdf(PDF_PATHS)
    chunks = chunk_text(text)
    save_chunks(chunks)

    print("\n── First 3 chunks preview ──")
    with open("data/chunks.json") as f:
        for c in json.load(f)[:3]:
            print(f"\n[Chunk {c['id']}]\n{c['text'][:200]}\n{'─'*40}")