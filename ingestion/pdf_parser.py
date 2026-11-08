"""PDF text extraction with OCR fallback for scanned pages."""
import pymupdf as fitz  # PyMuPDF


def _ocr_available():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_pdf_pages(path: str) -> list:
    """Return a list with one entry per page (text or placeholder)."""
    try:
        doc = fitz.open(path)
    except Exception:
        return []
    ocr_ok = _ocr_available()
    pages = []
    for page in doc:
        txt = (page.get_text() or "").strip()
        if len(txt) < 100:
            # likely scanned page -> render to image -> OCR (if available)
            if ocr_ok:
                try:
                    import pytesseract
                    from PIL import Image
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr = pytesseract.image_to_string(img)
                    if ocr and ocr.strip():
                        txt = ocr.strip()
                    else:
                        txt = txt or "[No text extracted (possibly scanned/image page)]"
                except Exception:
                    txt = txt or "[No text extracted (OCR failed)]"
            else:
                txt = txt or "[No text extracted (possibly scanned/image page; tesseract not installed)]"
        pages.append(txt)
    doc.close()
    return pages


def extract_pdf_text(path: str) -> str:
    pages = extract_pdf_pages(path)
    n = len(pages)
    if not n:
        return ""
    parts = []
    for i, t in enumerate(pages, 1):
        parts.append(f"--- Page {i} of {n} ---\n{t}")
    return "\n\n".join(parts)
