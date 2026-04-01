"""
Ingest textbook PDFs into a ChromaDB vector store for RAG.

Usage:
    python -m rag.ingest

Reads all PDFs from the textbooks/ directory, extracts text, chunks by
section headings, embeds with nomic-embed-text via Ollama, and stores
in a persistent ChromaDB collection at rag/chroma_db/.
"""
import os
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image as PILImage
import chromadb
import ollama

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTBOOKS_DIR = os.path.join(PROJECT_ROOT, "textbooks")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "rag", "chroma_db")
COLLECTION_NAME = "cephalometric_textbooks"
EMBED_MODEL = "nomic-embed-text"

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF. Falls back to OCR for scanned pages."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        # Try embedded text first
        text = page.get_text("text").strip()
        if len(text) > 50:
            pages.append(text)
        else:
            # Scanned page — render to image and OCR
            pix = page.get_pixmap(dpi=300)
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang="eng")
            if ocr_text.strip():
                pages.append(ocr_text.strip())
            print(f"      [OCR] page {page_num + 1}: {len(ocr_text.strip())} chars")
    doc.close()
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Matches common textbook headings: lines that are short, title-cased or ALL CAPS,
# and don't end with typical sentence punctuation.
HEADING_RE = re.compile(
    r"^(?:"
    r"[A-Z][A-Za-z\s\-/()]{3,80}(?:analysis|angle|plane|position|length|height|"
    r"convexity|depth|relationship|measurement|incisor|molar|point|line|ratio|"
    r"interpretation|summary|introduction|overview|definition|method|procedure)"
    r"|[A-Z][A-Z\s\-/]{4,60}"  # ALL CAPS lines
    r")$",
    re.MULTILINE | re.IGNORECASE,
)

def chunk_text(text: str, source_name: str, max_chunk_chars: int = 1500, overlap_chars: int = 200) -> list[dict]:
    """
    Split extracted text into chunks.
    Strategy:
      1. Try splitting on section headings first.
      2. If a section is too long, split on paragraph boundaries.
      3. Each chunk gets metadata with source and heading.
    """
    # Clean up common PDF artifacts
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Split on headings
    parts = HEADING_RE.split(text)
    headings = HEADING_RE.findall(text)

    chunks = []
    current_heading = source_name

    sections = []
    if parts and parts[0].strip():
        sections.append((source_name, parts[0].strip()))

    for i, heading in enumerate(headings):
        body_idx = i + 1
        body = parts[body_idx].strip() if body_idx < len(parts) else ""
        if body:
            sections.append((heading.strip(), body))

    if not sections:
        sections = [(source_name, text)]

    for heading, body in sections:
        if len(body) <= max_chunk_chars:
            chunks.append({
                "text": f"{heading}\n\n{body}",
                "heading": heading,
                "source": source_name,
            })
        else:
            # Split long sections into overlapping paragraph-based chunks
            paragraphs = body.split("\n\n")
            current = ""
            for para in paragraphs:
                if len(current) + len(para) > max_chunk_chars and current:
                    chunks.append({
                        "text": f"{heading}\n\n{current.strip()}",
                        "heading": heading,
                        "source": source_name,
                    })
                    # Keep overlap from end of current chunk
                    current = current[-overlap_chars:] + "\n\n" + para
                else:
                    current = current + "\n\n" + para if current else para
            if current.strip():
                chunks.append({
                    "text": f"{heading}\n\n{current.strip()}",
                    "heading": heading,
                    "source": source_name,
                })

    return chunks


# ---------------------------------------------------------------------------
# Embedding & storage
# ---------------------------------------------------------------------------
def embed_text(text: str) -> list[float]:
    """Get embedding vector from Ollama nomic-embed-text."""
    resp = ollama.embed(model=EMBED_MODEL, input=text)
    return resp["embeddings"][0]


def ingest_all():
    """Main ingestion pipeline: extract → chunk → embed → store."""
    pdf_files = [f for f in os.listdir(TEXTBOOKS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {TEXTBOOKS_DIR}")
        return

    print(f"Found {len(pdf_files)} textbook(s): {pdf_files}")

    # Collect all chunks
    all_chunks = []
    for pdf_name in pdf_files:
        pdf_path = os.path.join(TEXTBOOKS_DIR, pdf_name)
        source = os.path.splitext(pdf_name)[0]
        print(f"  Extracting: {pdf_name} ...")
        text = extract_text_from_pdf(pdf_path)
        print(f"    -> {len(text)} chars extracted")
        chunks = chunk_text(text, source)
        print(f"    -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Setup ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Delete existing collection if re-ingesting
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed and insert
    print("Embedding chunks (this may take a minute) ...")
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(all_chunks):
        emb = embed_text(chunk["text"])
        ids.append(f"chunk_{i}")
        embeddings.append(emb)
        documents.append(chunk["text"])
        metadatas.append({"heading": chunk["heading"], "source": chunk["source"]})
        if (i + 1) % 10 == 0 or i == len(all_chunks) - 1:
            print(f"  [{i + 1}/{len(all_chunks)}]")

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"\nDone! {len(all_chunks)} chunks stored in {CHROMA_DIR}")


if __name__ == "__main__":
    ingest_all()
