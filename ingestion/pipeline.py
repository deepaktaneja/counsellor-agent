"""Ingestion pipeline: parse (pdf/image) -> extract -> store in db + chroma."""
import os, shutil
from pathlib import Path
from config import FILES_DIR, IMAGES_DIR
from ingestion import pdf_parser, image_parser, extractor
from memory import db, chroma_store


def ingest_pdf(path: str, filename: str, doc_type_override: str = None, category_override: str = None) -> dict:
    from ingestion import pdf_parser
    pages = pdf_parser.extract_pdf_pages(path)
    text = "\n\n".join(f"--- Page {i} of {len(pages)} ---\n{p}" for i, p in enumerate(pages, 1))
    # copy file
    dest = FILES_DIR / filename
    try:
        shutil.copy2(path, dest)
    except Exception:
        dest = Path(path)
    data = extractor.extract_structured(text[:16000])
    doc_type = doc_type_override or data.get("doc_type", "other")
    category = category_override or _category_from_type(doc_type)
    doc_id = db.save_document(filename=filename, doc_type=doc_type, category=category,
                              summary=data.get("summary", ""), full_text=text,
                              file_path=str(dest))
    _store_structured(doc_id, data)
    chroma_store.add_text(f"doc_{doc_id}", text[:8000], {"type": "document", "doc_id": doc_id, "doc_type": doc_type, "category": category, "pages": len(pages)})
    pages_with_text = sum(1 for p in pages if p and not p.startswith("[No text"))
    return {"doc_id": doc_id, "data": data, "text": text, "doc_type": doc_type, "category": category, "pages": len(pages), "pages_with_text": pages_with_text}


def ingest_image(path: str, filename: str, doc_type_override: str = None, category_override: str = None) -> dict:
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
    doc_type = doc_type_override or data.get("doc_type", "other")
    category = category_override or _category_from_type(doc_type)
    doc_id = db.save_document(filename=filename, doc_type=doc_type, category=category,
                              summary=data.get("summary", ""), full_text=text,
                              file_path=str(dest))
    _store_structured(doc_id, data)
    chroma_store.add_text(f"doc_{doc_id}", text[:4000], {"type": "document", "doc_id": doc_id, "doc_type": doc_type, "category": category})
    # if situational photo, also save as observation
    if data.get("doc_type") == "situational_photo":
        obs_text = data.get("summary") or text[:200]
        oid = db.save_observation(raw_text=text, observation=obs_text, summary=obs_text,
                                  tags=["photo", "situational"])
        chroma_store.add_text(f"obs_{oid}", obs_text, {"type": "observation", "obs_id": oid})
    return {"doc_id": doc_id, "data": data, "text": text, "doc_type": doc_type, "category": category}


def _category_from_type(doc_type):
    dt = (doc_type or "").lower()
    if dt in ("medication_label", "prescription"):
        return "medication"
    if dt in ("assessment", "psych_report", "psychological_assessment", "medical_report"):
        return "assessment"
    if dt in ("therapy_note", "therapy_session", "therapy_report", "handwritten_note"):
        return "therapy"
    if dt == "situational_photo":
        return "other"
    return "other"


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
