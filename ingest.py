"""
ingest.py - Reads the return policy PDF, splits it into chunks,
converts each chunk into a vector (embedding), and stores it in ChromaDB.

Run this ONCE per PDF. If you add more PDFs later, run it again.
"""

import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# CONFIG - These are the "knobs" you'll tune later
# ------------------------------------------------------------------
PDF_PATH = "policies/amazon_return_policy.pdf"
CHROMA_PATH = "chroma_db"       # Chroma will save data here on disk
COLLECTION_NAME = "return_policies"
CHUNK_SIZE = 500                # characters per chunk (not words!)
CHUNK_OVERLAP = 100             # overlap between chunks so we don't lose context


# ------------------------------------------------------------------
# STEP 1: Extract raw text from the PDF
# ------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Reads a PDF and returns all its text as one long string.
    
    Why pypdf? It's lightweight and pure Python. For complex layouts
    (tables, columns) you'd need something like unstructured, but
    return policies are usually simple text.
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        full_text += page_text + "\n"
        print(f"  Extracted page {page_num + 1}: {len(page_text)} chars")
    return full_text


# ------------------------------------------------------------------
# STEP 2: Split text into overlapping chunks
# ------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits a long string into smaller overlapping chunks.
    
    WHY CHUNK?
    - LLMs have context limits - you can't send a 50-page PDF
    - Smaller chunks = more precise retrieval (a chunk about
      "refund timelines" won't be polluted by "non-returnable items")
    
    WHY OVERLAP?
    - Prevents cutting a sentence in half at chunk boundaries
    - If a key idea spans two chunks, the overlap keeps it together
      in at least one chunk
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:                       # skip empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap   # step forward, but back by overlap
    return chunks


# ------------------------------------------------------------------
# STEP 3: Convert each chunk into a vector (embedding)
# ------------------------------------------------------------------
def embed_chunks(chunks: list[str], model: SentenceTransformer):
    """
    Converts each text chunk into a list of 384 numbers.
    
    WHY EMBEDDINGS?
    - Computers can't compare "meaning" of raw text
    - An embedding model maps similar meanings to nearby points
      in 384-dimensional space
    - So "damaged phone refund" and "broken mobile return" end up
      close together, even though they share no words
    
    WHY all-MiniLM-L6-v2?
    - It's tiny (~80MB), fast, runs on CPU, and is good enough
      for retrieval. For production you might use a bigger model.
    - 384 is its output dimension.
    """
    print(f"\n  Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings.tolist()   # Chroma wants plain lists, not numpy arrays


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RETURN POLICY INGESTION")
    print("=" * 60)

    # --- Verify PDF exists ---
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("Place your return policy PDF in the 'policies/' folder.")
        return

    # --- Step 1: Extract ---
    print(f"\n[1/4] Extracting text from {PDF_PATH}")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"  Total text length: {len(text)} characters")

    # --- Step 2: Chunk ---
    print(f"\n[2/4] Chunking text (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  Created {len(chunks)} chunks")
    print(f"  Example chunk #0 (first 200 chars):")
    print(f"  \"{chunks[0][:200]}...\"")

    # --- Step 3: Embed ---
    print(f"\n[3/4] Loading embedding model (first run downloads ~80MB)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_chunks(chunks, model)

    # --- Step 4: Store in Chroma ---
    print(f"\n[4/4] Storing in ChromaDB at '{CHROMA_PATH}/'")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # If you re-run this script, delete the old collection first
    # so we don't get duplicate entries
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted old collection")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"  Stored {len(chunks)} chunks ✓")

    print("\n" + "=" * 60)
    print("DONE. You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()