import os
from pypdf import PdfReader
from src.vector_store.vector_store import VectorStore

PDF_PATH = "src/documents"

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        parts.append(page_text)
    return "\n".join(parts)

def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

def get_pdf_files(folder: str):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]

def main():
    pdf_files = get_pdf_files(PDF_PATH)
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {PDF_PATH}")

    print(f"Found {len(pdf_files)} PDFs")

    print("Initializing vector store...")
    vs = VectorStore(persist_directory="vectordb")

    for pdf_path in pdf_files:
        print(f"\nLoading: {pdf_path}")
        text = load_pdf(pdf_path)

        print("Splitting text...")
        chunks = split_text(text)

        chunks = [f"[SOURCE: {os.path.basename(pdf_path)}]\n{c}" for c in chunks]

        print(f"Adding {len(chunks)} chunks...")
        vs.add_documents(chunks)
        
    # vs.persist()
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
