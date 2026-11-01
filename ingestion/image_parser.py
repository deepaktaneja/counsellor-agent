"""Image -> text via vision LLM (Ollama) with optional OCR cross-check."""
import base64
import os
from config import VISION_MODEL, OLLAMA_BASE_URL, USE_GEMINI_VISION, GEMINI_API_KEY, GEMINI_MODEL


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _ocr_image(path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


def describe_image(path: str) -> str:
    """Return extracted text/description from an image."""
    ocr_text = _ocr_image(path)
    # Vision LLM
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_BASE_URL)
        b64 = _encode_image(path)
        prompt = (
            "You are reading a medical/psychological document or photo related to an autistic "
            "young adult with OCD. Extract ALL readable text. If it's a photo of a situation, "
            "describe what you see. Return plain text only."
        )
        resp = client.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [b64]}],
            options={"temperature": 0.1},
        )
        vision_text = resp["message"]["content"].strip()
    except Exception as e:
        vision_text = f"[vision model unavailable: {e}]"
    # Prefer vision text; append OCR if both present and differ
    if ocr_text and ocr_text not in vision_text:
        return f"{vision_text}\n\n[OCR cross-check]:\n{ocr_text}"
    return vision_text or ocr_text


def gemini_describe_image(path: str) -> str:
    if not (USE_GEMINI_VISION and GEMINI_API_KEY):
        return ""
    try:
        import httpx, base64, json
        b64 = _encode_image(path)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [
                {"text": "Extract all readable text from this medical/psychological document or describe the photo. Plain text only."},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            ]}],
        }
        r = httpx.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"[gemini vision error: {e}]"
