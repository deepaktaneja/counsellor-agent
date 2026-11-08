"""Chroma vector store for observations and document chunks (RAG)."""
import chromadb
from chromadb.config import Settings
from config import CHROMA_PATH

_client = None
_collection = None


def _client_instance():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
    return _client


def _collection_instance():
    global _collection
    if _collection is None:
        _collection = _client_instance().get_or_create_collection("counsellor_memory")
    return _collection


def add_text(doc_id: str, text: str, metadata: dict = None):
    if not text or not text.strip():
        return
    _collection_instance().upsert(ids=[doc_id], documents=[text], metadatas=[metadata or {}])


def chunk_text(text, size=1000, overlap=200):
    """Split text into overlapping chunks of approx `size` characters.

    Prefers breaking at paragraph/sentence boundaries so chunks stay readable.
    """
    if not text:
        return []
    text = str(text)
    n = len(text)
    if n <= size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        if end < n:
            cut = max(chunk.rfind("\n\n"), chunk.rfind("\n"), chunk.rfind(". "))
            if cut > size // 2:
                end = start + cut + 1
                chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = end
    return chunks


def add_chunked(doc_id: str, text: str, metadata: dict = None,
                chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Chunk text and store each chunk with a unique id + shared metadata.

    Each chunk id is ``{doc_id}_chunk_{N}`` and metadata gains ``chunk_index``
    and ``parent_id``. Returns the list of chunk ids created.
    """
    if not text or not text.strip():
        return []
    chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        return []
    ids = ["%s_chunk_%d" % (doc_id, i) for i in range(len(chunks))]
    metas = []
    for i in range(len(chunks)):
        m = dict(metadata or {})
        m["chunk_index"] = i
        m["parent_id"] = doc_id
        metas.append(m)
    _collection_instance().upsert(ids=ids, documents=chunks, metadatas=metas)
    return ids


def query(text: str, k: int = 5):
    try:
        res = _collection_instance().query(query_texts=[text], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return list(zip(docs, metas))
    except Exception:
        return []


def count():
    try:
        return _collection_instance().count()
    except Exception:
        return 0

def delete(doc_id):
    try:
        _collection_instance().delete(ids=[doc_id])
    except Exception:
        pass

def delete_many(doc_ids):
    if not doc_ids:
        return
    try:
        _collection_instance().delete(ids=list(doc_ids))
    except Exception:
        pass


def delete_by_parent(parent_id):
    """Delete all chunks whose metadata.parent_id == parent_id."""
    try:
        res = _collection_instance().get(where={"parent_id": parent_id})
        ids = res.get("ids", [])
        if ids:
            _collection_instance().delete(ids=ids)
    except Exception:
        pass
