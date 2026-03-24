import os
import json
from typing import List, Dict
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# -------- CONFIG --------
SYLLABI_DIR = "syllabi"
ADVISING_DIR = "advising"
DB_PATH = "rag_db"
COLLECTION_NAME = "cs_syllabi_advising"
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI()

def load_text_files(root_dir: str, label: str) -> List[Dict]:
    """
    Recursively load all .txt, .md files under root_dir.
    Returns list of dicts with {id, text, metadata}.
    metadata contains source_type (syllabus/advising) and path.
    """
    docs = []
    if not os.path.isdir(root_dir):
        return docs

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith((".txt", ".md")):
                continue
            fpath = os.path.join(dirpath, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                continue

            doc_id = f"{label}:{os.path.relpath(fpath, root_dir)}"
            docs.append({
                "id": doc_id,
                "text": text,
                "metadata": {
                    "source_type": label,
                    "rel_path": os.path.relpath(fpath, start="."),
                }
            })
    return docs

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks

def main():
    # 1) Load docs from syllabi + advising
    docs = []
    docs.extend(load_text_files(SYLLABI_DIR, label="syllabus"))
    docs.extend(load_text_files(ADVISING_DIR, label="advising"))

    print(f"Loaded {len(docs)} documents before chunking.")

    # 2) Chunk them
    chunk_ids = []
    chunk_texts = []
    metadatas = []

    for doc in docs:
        base_id = doc["id"]
        for idx, chunk in enumerate(chunk_text(doc["text"])):
            cid = f"{base_id}#chunk{idx}"
            chunk_ids.append(cid)
            chunk_texts.append(chunk)
            md = dict(doc["metadata"])
            md["parent_id"] = base_id
            md["chunk_index"] = idx
            metadatas.append(md)

    print(f"Created {len(chunk_ids)} chunks.")

    if not chunk_ids:
        print("No chunks to index. Exiting.")
        return

    # 3) Create Chroma DB and collection
    chroma_client = chromadb.PersistentClient(path=DB_PATH, settings=Settings())
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    coll = chroma_client.create_collection(name=COLLECTION_NAME)

    # 4) Embed in batches and add to collection
    BATCH_SIZE = 64
    for i in range(0, len(chunk_ids), BATCH_SIZE):
        batch_ids = chunk_ids[i:i+BATCH_SIZE]
        batch_texts = chunk_texts[i:i+BATCH_SIZE]
        batch_meta = metadatas[i:i+BATCH_SIZE]

        emb_resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch_texts,
        )
        embeddings = [e.embedding for e in emb_resp.data]

        coll.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"Indexed {i + len(batch_ids)} / {len(chunk_ids)} chunks...")

    print("Indexing completed.")

if __name__ == "__main__":
    main()

