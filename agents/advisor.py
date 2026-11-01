"""Advisor agent: extract observation/question, RAG context, generate advice."""
import json
import ollama
from config import LLM_MODEL, OLLAMA_BASE_URL, PROFILE_PATH
from memory import db, chroma_store
from safety.guard import sanitize_output


def load_profile() -> str:
    try:
        return PROFILE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "No profile yet. Please fill in profile/son_profile.md."


def _ollama_chat(messages, format=None):
    client = ollama.Client(host=OLLAMA_BASE_URL)
    resp = client.chat(model=LLM_MODEL, messages=messages, format=format, options={"temperature": 0.3})
    return resp["message"]["content"]


def extract_observation_question(text: str) -> dict:
    """Split a spoken utterance into {observation, question}."""
    if not text or not text.strip():
        return {"observation": "", "question": ""}
    system = (
        "You split a parent's spoken sentence into two parts. "
        "Return ONLY valid JSON with keys 'observation' and 'question'. "
        "'observation' = description of the son's behavior/situation (facts only). "
        "'question' = what the parent is asking for. "
        "If a part is absent, use empty string. Do not add commentary."
    )
    try:
        raw = _ollama_chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": text}],
            format={"type": "object", "properties": {
                "observation": {"type": "string"},
                "question": {"type": "string"},
            }, "required": ["observation", "question"]},
        )
        data = json.loads(raw)
        return {
            "observation": (data.get("observation") or "").strip(),
            "question": (data.get("question") or "").strip(),
        }
    except Exception:
        # fallback: treat whole text as observation
        return {"observation": text.strip(), "question": ""}


def build_context(question: str, observation: str = "", prev_question: str = "", prev_advice: str = "") -> str:
    profile = load_profile()
    structured = db.profile_text()
    meds = db.current_medications_text()
    dx = db.diagnoses_text()
    # RAG retrieval
    query_text = " ".join([question, observation]).strip()
    hits = chroma_store.query(query_text, k=5) if query_text else []
    past = []
    for doc, meta in hits:
        past.append(f"- {doc[:300]}")
    past_block = "\n".join(past) if past else "None yet."
    prev_block = ""
    if prev_question or prev_advice:
        prev_block = f"\nPREVIOUS CONVERSATION (do not repeat this advice):\nPrevious question: {prev_question}\nPrevious advice given: {prev_advice}\n"
    return f"""SON PROFILE (structured):
{structured}

SON PROFILE (free-form notes):
{profile}

CURRENT MEDICATIONS: {meds}
DIAGNOSES: {dx}

RELEVANT PAST OBSERVATIONS:
{past_block}
{prev_block}"""


def advise(question: str, observation: str = "", prev_question: str = "", prev_advice: str = "") -> str:
    if not question and not observation:
        return "I didn't catch that. Could you say it again?"
    ctx = build_context(question, observation, prev_question, prev_advice)
    follow_up_instruction = ""
    if prev_question or prev_advice:
        follow_up_instruction = (
            "This is a follow-up question. You already gave advice above (shown in PREVIOUS CONVERSATION). "
            "Do NOT repeat that same advice. Address ONLY the new question briefly. "
            "Keep it to 1-2 short sentences. Reference the previous advice only if needed for context. "
        )
    system = (
        "You are a calm, practical advisor for a parent of a 20-year-old autistic (Asperger's) "
        "son with severe OCD. Give concise, spoken-style advice in 2 short points: "
        "(1) what is likely happening, (2) what to do right now (graded, small steps). "
        "Never suggest medication changes. Never provide reassurance that feeds OCD compulsions "
        "(do not say 'it's locked' / 'you checked enough'). "
        "Do NOT suggest escalating to or contacting a therapist/counsellor - the parent visits "
        "the counsellor weekly and she is not reachable between sessions. "
        "Consider his likes, dislikes and sensory preferences when suggesting activities; prefer calm, familiar options. "
        "Use plain language, short sentences. Max ~80 words total."
        + follow_up_instruction
    )
    user_msg = f"{ctx}\n\nPARENT SAYS:\nObservation: {observation}\nQuestion: {question}\n\nAdvice:"
    try:
        raw = _ollama_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])
        return sanitize_output(raw.strip())
    except Exception as e:
        return f"Sorry, I couldn't generate advice right now: {e}"


def summarize_observation(text: str) -> dict:
    """Generate summary + tags for a saved observation."""
    system = (
        "Summarize a parent's observation about their autistic son with OCD. "
        "Return ONLY JSON with keys 'summary' (one sentence) and 'tags' (list of 1-4 short tags)."
    )
    try:
        raw = _ollama_chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": text}],
            format={"type": "object", "properties": {
                "summary": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            }, "required": ["summary", "tags"]},
        )
        data = json.loads(raw)
        return {"summary": data.get("summary", text[:80]), "tags": data.get("tags", [])}
    except Exception:
        return {"summary": text[:80], "tags": []}
