"""LLM extractor: turn raw text into structured medical data (JSON)."""
import json
import ollama
from config import LLM_MODEL, OLLAMA_BASE_URL


def _chat(messages, format=None):
    client = ollama.Client(host=OLLAMA_BASE_URL)
    resp = client.chat(model=LLM_MODEL, messages=messages, format=format, options={"temperature": 0.1})
    return resp["message"]["content"]


SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string", "enum": ["medical_report", "prescription", "handwritten_note",
                                                "situational_photo", "medication_label", "assessment",
                                                "therapy_note", "other"]},
        "summary": {"type": "string"},
        "diagnoses": {"type": "array", "items": {"type": "object", "properties": {
            "condition": {"type": "string"}, "diagnosed_by": {"type": "string"}, "date": {"type": "string"}, "notes": {"type": "string"}
        }, "required": ["condition"]}},
        "medications": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "dose": {"type": "string"}, "frequency": {"type": "string"},
            "prescriber": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"},
            "status": {"type": "string"}, "notes": {"type": "string"}
        }, "required": ["name"]}},
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "category": {"type": "string"}, "content": {"type": "string"}
        }, "required": ["content"]}},
        "triggers": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["doc_type", "summary"],
}


def extract_structured(text: str) -> dict:
    if not text or not text.strip():
        return {"doc_type": "other", "summary": "", "diagnoses": [], "medications": [],
                "findings": [], "triggers": []}
    system = (
        "You extract structured medical/psychological information about an autistic young adult "
        "with severe OCD from a document. Return ONLY JSON matching the schema. "
        "doc_type: medical_report/prescription/handwritten_note/situational_photo/medication_label/assessment/therapy_note/other. "
        "medication status: 'current' or 'past'. findings categories: trigger, recommendation, observation, other. "
        "For 'diagnoses', extract any condition or disorder explicitly stated as affecting the patient "
        "(e.g., if text says 'autism' or 'OCD', include them). Do not invent conditions."
    )
    try:
        raw = _chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": text[:8000]}],
            format=SCHEMA,
        )
        data = json.loads(raw)
        # normalize
        data.setdefault("diagnoses", [])
        data.setdefault("medications", [])
        data.setdefault("findings", [])
        data.setdefault("triggers", [])
        data.setdefault("summary", "")
        data.setdefault("doc_type", "other")
        return data
    except Exception as e:
        return {"doc_type": "other", "summary": f"[extraction failed: {e}]", "diagnoses": [],
                "medications": [], "findings": [], "triggers": []}
