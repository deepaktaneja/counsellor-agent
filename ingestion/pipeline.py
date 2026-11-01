"""Ingestion pipeline: parse (pdf/image) -> extract -> store in db + chroma."""
import os, shutil
from pathlib import Path
from config import FILES_DIR, IMAGES_DIR
from ingestion import pdf_parser, image_parser, extractor
from memory import db, chroma_store


def ingest_pdf(path: str, filename: str) -> dict:
    text = pdf_parser.extract_pdf_text(path)
    # copy file
    dest = FILES_DIR / filename
    try:
        shutil.copy2(path, dest)
    except Exception:
        dest = Path(path)
    data = extractor.extract_structured(text)
    doc_id = db.save_document(filename=filename, doc_type=data.get("doc_type", "other"),
                              summary=data.get("summary", ""), full_text=text,
                              file_path=str(dest))
    _store_structured(doc_id, data)
    chroma_store.add_text(f"doc_{doc_id}", text[:4000], {"type": "document", "doc_id": doc_id})
    return {"doc_id": doc_id, "data": data, "text": text}


def ingest_image(path: str, filename: str) -> dict:
    # optional gemini
    from config import USE_GEMINI_VISION, GEMINI_API_KEY
    if USE_GEMINI_VISION and GEMINI_API_KEY:
        from ingestion.image_parser import gemini_describe_image
        text = gemini_describe_image(path)
        if not text or text.startswith("[gemini vision error"):
            text = image_parser.describe_image(path)
    else:
        text = image_parser.describe_image(path)
    dest = IMAGES_DIR / filename
    try:
        shutil.copy2(path, dest)
    except Exception:
        dest = Path(path)
    data = extractor.extract_structured(text)
    doc_id = db.save_document(filename=filename, doc_type=data.get("doc_type", "other"),
                              summary=data.get("summary", ""), full_text=text,
                              file_path=str(dest))
    _store_structured(doc_id, data)
    chroma_store.add_text(f"doc_{doc_id}", text[:4000], {"type": "document", "doc_id": doc_id})
    # if situational photo, also save as observation
    if data.get("doc_type") == "situational_photo":
        obs_text = data.get("summary") or text[:200]
        oid = db.save_observation(raw_text=text, observation=obs_text, summary=obs_text,
                                  tags=["photo", "situational"])
        chroma_store.add_text(f"obs_{oid}", obs_text, {"type": "observation", "obs_id": oid})
    return {"doc_id": doc_id, "data": data, "text": text}


def _store_structured(doc_id, data):
    for d in (data.get("diagnoses") or []):
        db.save_diagnosis(condition=d.get("condition", ""), diagnosed_by=d.get("diagnosed_by", ""),
                          date=d.get("date", ""), notes=d.get("notes", ""), source_doc_id=doc_id)
    for m in (data.get("medications") or []):
        db.save_medication(name=m.get("name", ""), dose=m.get("dose", ""),
                           frequency=m.get("frequency", ""), prescriber=m.get("prescriber", ""),
                           start_date=m.get("start_date", ""), end_date=m.get("end_date", ""),
                           status=m.get("status", "current"), source_doc_id=doc_id,
                           notes=m.get("notes", ""))
    for f in (data.get("findings") or []):
        db.save_finding(category=f.get("category", "other"), content=f.get("content", ""),
                        source_doc_id=doc_id)
    for t in (data.get("triggers") or []):
        db.save_finding(category="trigger", content=t, source_doc_id=doc_id)
