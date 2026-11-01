"""PDF text extraction with OCR fallback for scanned pages."""
import pymupdf as fitz  # PyMuPDF


def extract_pdf_text(path: str) -> str:
    try:
        doc = fitz.open(path)
    except Exception:
        return ""
    pages = []
    for page in doc:
        txt = page.get_text().strip()
        if len(txt) < 100:
            # likely scanned page -> render to image -> OCR
            try:
                import pytesseract
                from PIL import Image
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr = pytesseract.image_to_string(img)
                txt = ocr.strip() if ocr.strip() else txt
            except Exception:
                pass
        pages.append(txt)
    doc.close()
    return "\n\n".join(p for p in pages if p.strip())
