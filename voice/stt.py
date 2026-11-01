"""Speech-to-text using faster-whisper (local, open source)."""
import os
import tempfile
import ollama
from config import LLM_MODEL, OLLAMA_BASE_URL, WHISPER_MODEL, WHISPER_INITIAL_PROMPT, WHISPER_VOCAB


def _detect_ext(audio_bytes: bytes) -> str:
    """Detect audio format from magic bytes."""
    if audio_bytes[:4] == b"RIFF":
        return ".wav"
    if audio_bytes[:4] == b"\x1aE\xdf\xa3":
        return ".webm"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return ".mp3"
    if audio_bytes[:4] == b"OggS":
        return ".ogg"
    if audio_bytes[:4] == b"fLaC":
        return ".flac"
    if len(audio_bytes) > 8 and audio_bytes[4:8] == b"ftyp":
        return ".mp4"
    return ".wav"


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        _MODEL = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _MODEL


def _correct_transcription(text: str) -> str:
    """Use LLM to fix obvious transcription errors using known vocabulary."""
    if not text or not text.strip():
        return text
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        prompt = (
            "You are correcting a speech-to-text transcription from a parent talking about "
            "their autistic son with severe OCD. Fix any transcription errors (misspelled names, "
            "wrong words, misheard phrases). Common words: " + WHISPER_VOCAB + ". "
            "Preserve the original meaning and sentence structure. Only fix clear errors. "
            "Return only the corrected text, nothing else.\n\n"
            f"Transcription: {text}"
        )
        resp = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        corrected = resp["message"]["content"].strip()
        if len(corrected) < len(text) * 0.5 or len(corrected) > len(text) * 2:
            return text
        return corrected
    except Exception:
        return text


def transcribe(audio_bytes: bytes, correct: bool = True) -> str:
    """Transcribe audio bytes to text, with optional LLM correction."""
    if not audio_bytes:
        return ""
    ext = _detect_ext(audio_bytes)
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        model = _get_model()
        segments, info = model.transcribe(
            path, beam_size=5, initial_prompt=WHISPER_INITIAL_PROMPT
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if correct and text:
            text = _correct_transcription(text)
        return text
    except Exception as e:
        return f"[transcription error: {e}]"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
