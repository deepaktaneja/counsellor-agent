"""Text-to-speech: macOS Indian voice (say) primary, Piper fallback. Strips markdown before speaking."""
import os
import re
import subprocess
import tempfile
from config import (TTS_BACKEND, TTS_VOICE, PIPER_VOICE_NAME, PIPER_VOICE_URL,
                    PIPER_VOICE_CONFIG_URL, VOICES_DIR)


def clean_for_tts(text: str) -> str:
    """Strip markdown, punctuation that TTS reads aloud, and normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"[-*•]\s+", ". ", text)
    text = re.sub(r"\d+\.\s+", " ", text)  # numbered lists -> space
    text = text.replace("*", "").replace("#", "").replace("_", " ").replace("`", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[^\w\s.,?!;:'\-]", " ", text)
    text = re.sub(r"\bwill\b", "wil", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def _voice_paths():
    onnx = VOICES_DIR / f"{PIPER_VOICE_NAME}.onnx"
    jsonc = VOICES_DIR / f"{PIPER_VOICE_NAME}.onnx.json"
    return onnx, jsonc


def _ensure_voice():
    onnx, jsonc = _voice_paths()
    if onnx.exists() and jsonc.exists():
        return onnx, jsonc
    try:
        import urllib.request
        if not onnx.exists():
            urllib.request.urlretrieve(PIPER_VOICE_URL, onnx)
        if not jsonc.exists():
            urllib.request.urlretrieve(PIPER_VOICE_CONFIG_URL, jsonc)
    except Exception:
        pass
    return onnx, jsonc


def speak_say(text: str) -> bytes | None:
    """macOS 'say' with Indian English voice -> WAV bytes."""
    if not text.strip():
        return None
    try:
        fd, path = tempfile.mkstemp(suffix=".aiff")
        os.close(fd)
        subprocess.run(["say", "-v", TTS_VOICE, "-o", path, text],
                       check=True, capture_output=True)
        wav_path = path.replace(".aiff", ".wav")
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", path, wav_path],
                       check=True, capture_output=True)
        with open(wav_path, "rb") as f:
            data = f.read()
        for p in (path, wav_path):
            try:
                os.remove(p)
            except OSError:
                pass
        return data
    except Exception:
        return None


def speak_piper(text: str) -> bytes | None:
    """Piper TTS -> WAV bytes."""
    if not text.strip():
        return None
    onnx, jsonc = _ensure_voice()
    if not (onnx.exists() and jsonc.exists()):
        return None
    try:
        from piper import PiperVoice
        import io, wave
        voice = PiperVoice.load(onnx, config_path=jsonc)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        return buf.getvalue()
    except Exception:
        return None


def synthesize(text: str) -> bytes | None:
    text = clean_for_tts(text)
    if not text.strip():
        return None
    if TTS_BACKEND == "say":
        data = speak_say(text)
        if data is not None:
            return data
    data = speak_piper(text)
    if data is not None:
        return data
    return speak_say(text)


def speak_to_file(text: str, out_path: str) -> bool:
    data = synthesize(text)
    if data is None:
        return False
    with open(out_path, "wb") as f:
        f.write(data)
    return True
