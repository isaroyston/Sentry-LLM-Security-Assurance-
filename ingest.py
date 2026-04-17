import os
import sys
import json
from typing import List, Dict, Tuple, Optional
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db.supabase_client import SupabaseDB, SupabaseVectorStore

load_dotenv()

PDF_PATH = "src/documents"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
TEST_SEARCH_THRESHOLD = float(os.getenv("INGEST_TEST_SEARCH_THRESHOLD", "0.5"))
SEMANTIC_CHUNK_MAX_SIZE = int(os.getenv("SEMANTIC_CHUNK_MAX_SIZE", "1500"))  # chars per chunk target

'''
doc_id's for ref,
doc_id='sgbank_emergency_withdrawal_policy'
doc_id='sgbank_identity_verification_and_authentication_policy'
doc_id='sgbank_transaction_monitoring_and_fraud_detection_policy'
doc_id='sgbank_withdrawal_policy_and_procedures'
'''

def load_pdf(path: str) -> str:
    """Load text from PDF file"""
    reader = PdfReader(path)
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def validate_embedding_config(openai_client: OpenAI) -> bool:
    """Validate that embedding model and dimensions are consistent and working.
    
    Checks:
    1. Model name is not None
    2. Dimensions is a positive integer (384 for text-embedding-3-large)
    3. Test embedding succeeds and returns correct dimensions
    """
    print("\n[VALIDATION] Checking embedding configuration...")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Dimensions: {EMBEDDING_DIMENSIONS}")
    
    if not EMBEDDING_MODEL:
        print("  ❌ EMBEDDING_MODEL is not set")
        return False
    if EMBEDDING_DIMENSIONS <= 0 or EMBEDDING_DIMENSIONS > 4096:
        print(f"  ❌ EMBEDDING_DIMENSIONS must be 1-4096, got {EMBEDDING_DIMENSIONS}")
        return False
    
    # Test embedding
    try:
        test_input = "test"
        resp = openai_client.embeddings.create(
            input=test_input,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        test_embedding = resp.data[0].embedding
        actual_dim = len(test_embedding)
        
        if actual_dim != EMBEDDING_DIMENSIONS:
            print(f"  ❌ Expected {EMBEDDING_DIMENSIONS} dims but got {actual_dim}")
            return False
        
        print(f"  ✓ Test embedding successful ({actual_dim} dims)")
        return True
    except Exception as e:
        print(f"  ❌ Test embedding failed: {e}")
        return False


def semantic_chunk_with_llm(text: str, doc_id: str, openai_client: OpenAI, max_size: int = 1500) -> List[str]:
    """Split text into semantic chunks using LLM to identify logical boundaries.
    
    Instead of fixed character splits, this asks GPT to identify where logical
    sections break (e.g., between policy topics). Each returned section is one chunk.
    
    Args:
        text: Full document text
        doc_id: Document identifier (for logging)
        openai_client: OpenAI client
        max_size: Target max characters per chunk (soft limit)
    
    Returns:
        List of semantically meaningful chunks
    """
    if len(text) < 500:
        # Too small to chunk meaningfully; return as-is
        return [text]
    
    try:
        # Ask GPT to identify logical section boundaries
        prompt = f"""Analyze this policy document and identify logical section boundaries.
Return a JSON array of objects, each with 'start' (character index) and 'title' (section name).
Keep sections roughly {max_size}-2000 characters each, but prefer boundaries at natural breaks.
Return ONLY valid JSON, no markdown or explanation.

Document excerpt (first 5000 chars):
{text[:5000]}
"""
        
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        
        try:
            content = resp.choices[0].message.content or "{}"
            sections = json.loads(content)
            if not isinstance(sections, list):
                sections = []
        except json.JSONDecodeError:
            sections = []
        
        # If LLM parsing failed, fall back to sentence-based splits
        if not sections:
            print(f"    [semantic_chunk] LLM boundary detection failed, using fallback...")
            return _fallback_sentence_chunk(text, max_size)
        
        # Extract chunks based on identified boundaries
        chunks = []
        for i, section in enumerate(sections):
            start_idx = section.get("start", 0)
            if i + 1 < len(sections):
                end_idx = sections[i + 1].get("start", len(text))
            else:
                end_idx = len(text)
            
            chunk = text[start_idx:end_idx].strip()
            if chunk:
                chunks.append(chunk)
        
        if not chunks:
            chunks = [text]
        
        return chunks
    
    except Exception as e:
        print(f"    [semantic_chunk] Error during LLM chunking: {e}")
        return _fallback_sentence_chunk(text, max_size)


def _fallback_sentence_chunk(text: str, max_size: int = 1500) -> List[str]:
    """Fallback chunking: split by sentences, group until ~max_size chars."""
    # Split by sentence boundaries (. ! ?)
    sentences = []
    current_sent = []
    
    for char in text:
        current_sent.append(char)
        if char in ".!?":
            sent = "".join(current_sent).strip()
            if sent:
                sentences.append(sent)
            current_sent = []
    
    if current_sent:
        sent = "".join(current_sent).strip()
        if sent:
            sentences.append(sent)
    
    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > max_size and current_chunk:
            # Start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text]


def verify_ingested_embeddings(db: SupabaseDB, sample_size: int = 5) -> bool:
    """Verify that embeddings were stored correctly in Supabase.
    
    Checks:
    1. Sample N documents from DB
    2. Verify embedding field is not NULL
    3. Verify embedding is a valid list of floats
    4. Verify embedding length matches EMBEDDING_DIMENSIONS
    """
    print("\n[VALIDATION] Verifying ingested embeddings...")
    
    try:
        docs = db.get_all_documents(doc_type="policy")
        if not docs:
            print("  ❌ No documents found in database")
            return False
        
        print(f"  Total documents in DB: {len(docs)}")
        
        # Sample up to sample_size documents
        sample = docs[:min(sample_size, len(docs))]
        valid = 0
        invalid = 0
        
        for doc in sample:
            doc_id = doc.get("id", "unknown")
            embedding = doc.get("embedding")
            
            # Check if embedding exists
            if embedding is None:
                print(f"    ❌ {doc_id}: embedding is NULL")
                invalid += 1
                continue
            
            # Parse if string
            if isinstance(embedding, str):
                try:
                    import ast
                    embedding = ast.literal_eval(embedding)
                except:
                    print(f"    ❌ {doc_id}: embedding string is not parseable")
                    invalid += 1
                    continue
            
            # Verify is list
            if not isinstance(embedding, list):
                print(f"    ❌ {doc_id}: embedding is not a list (type={type(embedding).__name__})")
                invalid += 1
                continue
            
            # Verify length
            if len(embedding) != EMBEDDING_DIMENSIONS:
                print(f"    ❌ {doc_id}: embedding has {len(embedding)} dims, expected {EMBEDDING_DIMENSIONS}")
                invalid += 1
                continue
            
            # Verify all elements are floats
            if not all(isinstance(x, (int, float)) for x in embedding):
                print(f"    ❌ {doc_id}: embedding contains non-numeric values")
                invalid += 1
                continue
            
            print(f"    ✓ {doc_id}: valid embedding ({len(embedding)} dims)")
            valid += 1
        
        if valid == 0:
            print(f"  ❌ All {len(sample)} sampled embeddings are invalid")
            return False
        
        print(f"  ✓ {valid}/{len(sample)} sampled embeddings are valid")
        return True
    
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False


def get_pdf_files(folder: str) -> List[str]:
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]

def slugify_filename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0].strip().lower()
    # simple slug: keep alnum, replace others with underscore
    out = []
    prev_us = False
    for ch in name:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    slug = "".join(out).strip("_")
    return slug or "document"

def build_ids(doc_id: str, num_chunks: int) -> List[str]:
    """Generate stable IDs for chunks (kept for reference)"""
    return [f"{doc_id}::chunk_{i:05d}" for i in range(num_chunks)]


def main():
    """Main ingestion pipeline: PDF → Chunks → Embeddings → Supabase"""
    
    print("\n" + "="*70)
    print("🚀 STARTING PDF INGESTION TO SUPABASE VECTOR STORE")
    print("="*70)
    
    # 0. Validate configuration
    print("\n[STEP 0] Validating configuration...")
    openai_client = OpenAI()
    
    if not validate_embedding_config(openai_client):
        print("❌ Embedding configuration validation failed")
        return False
    
    # 1. Find all PDFs
    print("\n[STEP 1] Finding PDFs...")
    pdf_files = get_pdf_files(PDF_PATH)
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {PDF_PATH}")

    print(f"✓ Found {len(pdf_files)} PDFs")

    # 2. Initialize Supabase DB and embedder
    print("\n[STEP 2] Connecting to Supabase...")
    try:
        db = SupabaseDB()
        if not db.health_check():
            print("❌ Failed to connect to Supabase")
            return False
        print("✓ Connected to Supabase")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

    print(f"\n[STEP 3] Clearing old policy documents...")
    try:
        deleted = db.delete_documents_by_doc_type("policy")
        print(f"  ✓ Cleared {deleted} old chunks")
    except Exception as e:
        print(f"  ⚠️  Failed to clear old documents: {e}")
        print("  Continuing with ingest anyway...")

    print(f"\n[STEP 4] Processing {len(pdf_files)} PDFs with semantic chunking...")
    vs = SupabaseVectorStore(db)
    print("✓ Embedding client ready and configured")

    # 4. Process each PDF
    total_chunks_ingested = 0
    total_failed = 0

    for pdf_path in pdf_files:
        source_name = os.path.basename(pdf_path)
        doc_id = slugify_filename(pdf_path)

        print(f"\n  📥 {source_name} (doc_id: {doc_id})")
        
        try:
            # Load PDF
            text = load_pdf(pdf_path)
            print(f"     • Loaded {len(text):,} characters")

            # Split into semantic chunks (LLM-based)
            chunks = semantic_chunk_with_llm(text, doc_id, openai_client, max_size=SEMANTIC_CHUNK_MAX_SIZE)
            print(f"     • Semantic chunking: {len(chunks)} sections")

            # Generate embeddings and ingest
            batch_results = []
            print(f"     • Ingesting chunks...")
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    resp = openai_client.embeddings.create(
                        input=chunk,
                        model=EMBEDDING_MODEL,
                        dimensions=EMBEDDING_DIMENSIONS,
                    )
                    embedding = resp.data[0].embedding
                    
                    # Verify embedding dimensions
                    if len(embedding) != EMBEDDING_DIMENSIONS:
                        print(f"       ⚠️  Chunk {i}: embedding has {len(embedding)} dims, expected {EMBEDDING_DIMENSIONS}")
                        total_failed += 1
                        continue
                    
                    # Insert into Supabase
                    result = db.add_document(
                        content=chunk,
                        embedding=embedding,
                        source=doc_id,
                        doc_type="policy",
                        metadata={
                            "source_filename": source_name,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "original_doc_id": doc_id,
                            "chunk_method": "semantic_lm"
                        }
                    )
                    batch_results.append(result)
                    
                    # Progress indicator
                    if (i + 1) % 5 == 0:
                        print(f"       → {i + 1}/{len(chunks)} done")
                        
                except Exception as e:
                    print(f"       ⚠️  Chunk {i} error: {e}")
                    total_failed += 1
                    continue

            total_chunks_ingested += len(batch_results)
            print(f"     ✓ Ingested {len(batch_results)}/{len(chunks)} chunks")

        except Exception as e:
            print(f"     ❌ {source_name} error: {e}")
            total_failed += 1
            continue

    # 5. Post-ingest verification
    print("\n[STEP 5] Verifying ingestion quality...")
    verify_ok = verify_ingested_embeddings(db, sample_size=5)
    
    # 6. Summary
    print("\n" + "="*70)
    print("✅ INGESTION COMPLETE")
    print("="*70)
    print(f"Total chunks ingested: {total_chunks_ingested}")
    print(f"Total errors: {total_failed}")
    print(f"Embedding validation: {'✓ PASS' if verify_ok else '⚠ WARNING'}")

    # 7. Test vector search
    if total_chunks_ingested > 0:
        print("\n[STEP 5] Testing vector search...")
        test_queries = [
            "withdrawal policy",
            "emergency withdrawal",
            "identity verification"
        ]
        
        for query in test_queries:
            resp = openai_client.embeddings.create(
                input=query,
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
            )
            query_embedding = resp.data[0].embedding
            results = vs.search(query_embedding, limit=2, threshold=TEST_SEARCH_THRESHOLD)
            
            if results:
                print(f"\n  Query: '{query}'")
                for j, result in enumerate(results, 1):
                    similarity = result.get('similarity', 0)
                    content = result.get('content', '')[:70]
                    source = result.get('source', 'unknown')
                    print(f"    {j}. [{similarity:.2%}] {content}... ({source})")
            else:
                print(f"\n  Query: '{query}' → No results at threshold {TEST_SEARCH_THRESHOLD}")
    
    print("\n" + "="*70)
    return total_chunks_ingested > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)