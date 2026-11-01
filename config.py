"""Configuration for the counsellor app."""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "documents"
FILES_DIR = DOCS_DIR / "files"
IMAGES_DIR = DOCS_DIR / "images"
VOICES_DIR = DATA_DIR / "voices"
DB_PATH = DATA_DIR / "counsellor.db"
CHROMA_PATH = DATA_DIR / "chroma"
PROFILE_PATH = ROOT / "profile" / "son_profile.md"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5-vl:7b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_VOCAB = os.getenv(
    "WHISPER_VOCAB",
    "Anmol, OCD, contamination, compulsion, gargle, reassurance, meltdown, "
    "hygiene, decontamination, Angry Birds, pigs, jailbreak, B.Tech, "
    "fluoxetine, psychiatrist, therapist, ERP, trigger, de-escalation",
)
WHISPER_INITIAL_PROMPT = os.getenv(
    "WHISPER_INITIAL_PROMPT",
    "A parent speaking about their autistic son with severe OCD. "
    "Common words: Anmol, OCD, contamination, compulsion, gargle, reassurance, "
    "meltdown, hygiene, decontamination, Angry Birds, pigs, jailbreak, "
    "fluoxetine, psychiatrist, therapist, ERP, trigger, de-escalation, India.",
)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
USE_GEMINI_VISION = os.getenv("USE_GEMINI_VISION", "false").lower() == "true"

# TTS
TTS_BACKEND = os.getenv("TTS_BACKEND", "say")  # say (macOS Indian voice) or piper
TTS_VOICE = os.getenv("TTS_VOICE", "Aman")  # Aman/Rishi/Tara (en_IN) for macOS say
PIPER_VOICE_NAME = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
PIPER_VOICE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
PIPER_VOICE_CONFIG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
)
PIPER_VOICE_URL_IN = (
    "https://huggingface.co/navgurukul-ai-labs/text-to-speech-en-IN-piper/resolve/main/"
    "en_IN-dataset=spicor-english-base=ljspeech-epochs=1089.onnx"
)
PIPER_VOICE_CONFIG_URL_IN = (
    "https://huggingface.co/navgurukul-ai-labs/text-to-speech-en-IN-piper/resolve/main/"
    "en_IN-dataset=spicor-english-base=ljspeech-epochs=1089.onnx.json"
)

DISTRESS_KEYWORDS = [
    "kill myself", "suicide", "suicidal", "want to die", "end my life",
    "self harm", "self-harm", "cutting", "hurt myself", "no reason to live",
    "better off dead", "overdose",
]
THERAPIST_CONTACT = os.getenv("THERAPIST_CONTACT", "Therapist/Psychiatrist (fill in .env)")

for d in (DATA_DIR, DOCS_DIR, FILES_DIR, IMAGES_DIR, VOICES_DIR, CHROMA_PATH):
    d.mkdir(parents=True, exist_ok=True)
