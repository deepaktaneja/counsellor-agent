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
