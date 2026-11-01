# Counsellor Advisor (Parent-facing)

Local, private, open-source voice advisor for a parent of an autistic (Asperger's) son with severe OCD.

## Setup
```bash
cd /Users/anmol/counsellor-agent
# reuse the shared venv or create one
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env 2>/dev/null; # edit .env (THERAPIST_CONTACT, etc.)
ollama pull qwen3:4b-instruct
ollama pull qwen2.5-vl:7b   # for scanned/handwritten docs
brew install tesseract       # optional, for scanned PDFs
```

## Run
```bash
PYTHONPATH=. streamlit run app.py --server.port 8502
```
Open http://localhost:8502.

## First use
1. Fill in `profile/son_profile.md` (triggers, what calms him, OCD themes, therapist contact, meds).
2. Upload his reports/photos on "Upload Records".
3. Use "Advisor" page: record a situation, review the split observation/question, Save or Skip, get spoken advice.

## Tests
```bash
PYTHONPATH=. pytest -q
```

## Notes
- Everything runs locally (Ollama + faster-whisper + Piper + Chroma + SQLite). No cloud, no cost.
- Push-to-talk only. Transcript always shown alongside voice.
- Safety: distress keyword check surfaces therapist contact; output filter blocks reassurance-for-compulsions.
- Not a replacement for licensed care.
