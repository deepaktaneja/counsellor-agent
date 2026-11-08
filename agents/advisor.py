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
    psych = db.psychology_text()
    psych_summary = db.get_latest_psychology_summary()
    # RAG retrieval
    query_text = " ".join([question, observation]).strip()
    hits = chroma_store.query(query_text, k=8) if query_text else []
    past = []
    for doc, meta in hits:
        page = meta.get("page")
        tag = " [p%d]" % page if page else ""
        past.append("- %s%s" % (tag, doc[:700]))
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

PSYCHOLOGY ASSESSMENTS & THERAPY:
{psych}
CACHED PSYCHOLOGY SUMMARY: {psych_summary or "None yet."}

RELEVANT PAST OBSERVATIONS:
{past_block}
{prev_block}"""


def advise(question: str, observation: str = "", prev_question: str = "", prev_advice: str = "", extra_context: str = "") -> str:
    if not question and not observation:
        return "I didn't catch that. Could you say it again?"
    ctx = build_context(question, observation, prev_question, prev_advice)
    if extra_context.strip():
        ctx = ctx + f"\n\nADDITIONAL OBSERVATION CONTEXT (from past observations):\n{extra_context.strip()}\n"
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


def summarize_observations(previous_summary, observations_text):
    system = (
        "You maintain a concise memory of a parent observations about their autistic son with severe OCD. "
        "Merge the NEW OBSERVATIONS into the PREVIOUS SUMMARY. "
        "Keep it under 120 words. Focus on recurring triggers, patterns, what helped, and current concerns. "
        "Do not invent facts. Return plain text only."
    )
    user = (
        f"PREVIOUS SUMMARY:\n{previous_summary or chr(39)+chr(39)}\n\n"
        f"NEW OBSERVATIONS:\n{observations_text}\n\n"
        "Updated summary:"
    )
    try:
        return _ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ).strip()
    except Exception:
        return previous_summary or ""


def _dedupe_observations(rows):
    seen = set()
    out = []
    for r in rows:
        key = (
            (r.get("observation") or "").strip(),
            (r.get("question") or "").strip(),
            (r.get("raw_text") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def compact_observations(threshold=20, batch_size=10, keep_recent=10):
    try:
        rows = db.list_observations(limit=1000)
    except Exception:
        return ""
    rows = sorted(rows, key=lambda r: r.get("id") or 0)
    deduped = _dedupe_observations(rows)
    previous_summary = db.get_latest_observation_summary() or ""
    if len(deduped) <= threshold:
        return previous_summary
    n_compact = max(0, len(deduped) - keep_recent)
    n_compact = min(n_compact, batch_size)
    if n_compact <= 0:
        return previous_summary
    to_compact = deduped[:n_compact]
    lines = []
    for r in to_compact:
        lines.append(
            f"- observation: {r.get(chr(39)+"observation"+chr(39), chr(39)+chr(39))} | question: {r.get(chr(39)+"question"+chr(39), chr(39)+chr(39))} | raw: {r.get(chr(39)+"raw_text"+chr(39), chr(39)+chr(39))}"
        )
    observations_text = "\n".join(lines)
    new_summary = summarize_observations(previous_summary, observations_text)
    if not new_summary:
        return previous_summary
    db.save_observation_summary(new_summary, len(to_compact))
    try:
        chroma_store.add_text("obs_summary_latest", new_summary, {"type": "observation_summary"})
    except Exception:
        pass
    for r in to_compact:
        try:
            db.delete_observation(r.get("id"))
        except Exception:
            pass
        try:
            chroma_store.delete("obs_" + str(r.get("id")))
        except Exception:
            pass
    return new_summary


def summarize_psychology_incrementally() -> str:
    """Three-pass psychology summarization.

    Pass 1: Summarize each page individually and store per-page summaries in DB.
    Pass 2: Cumulatively accumulate page summaries into one running summary.
    Pass 3: Final polish pass on the complete accumulated summary.
    """
    from memory import db

    docs = []
    for cat in ("assessment", "therapy"):
        docs.extend(db.list_documents(category=cat))
    # dedupe by id
    seen = set()
    unique_docs = []
    for d in docs:
        if d.get("id") not in seen:
            seen.add(d.get("id"))
            unique_docs.append(d)
    if not unique_docs:
        return "No psychology assessments or therapy notes recorded."

    # collect pages in order across all docs
    pages = []  # (doc_id, filename, doc_type, page_num, page_text)
    for d in unique_docs:
        full_text = (d.get("full_text") or "").strip()
        if not full_text:
            continue
        page_marker = "--- Page "
        parts = full_text.split(page_marker)
        if len(parts) > 1:
            for part in parts[1:]:
                header_end = part.find(" ---\n")
                if header_end != -1:
                    header = part[:header_end]
                    page_num = header.split(" of ")[0].strip()
                    page_text = part[header_end + 5:]
                else:
                    page_num = "?"
                    page_text = part
                pages.append((d.get("id"), d.get("filename", "doc"), d.get("doc_type", ""), page_num, page_text.strip()))
        else:
            pages.append((d.get("id"), d.get("filename", "doc"), d.get("doc_type", ""), "?", full_text))

    if not pages:
        return "No psychology content found."

    # ---- Pass 1: summarize each page individually, store in DB ----
    db.delete_psychology_page_summaries()  # clear stale entries
    page_summaries = []  # (doc_id, filename, doc_type, page_num, summary)
    for idx, (doc_id, fname, dtype, pnum, ptext) in enumerate(pages, 1):
        if not ptext or ptext.startswith("[No text"):
            continue
        system = (
            "You are summarizing one page of a psychology/therapy assessment report "
            "about a young autistic adult with severe OCD. Extract only what is on this page. "
            "Focus on diagnoses, observations, test scores, medication, therapy, concerns. "
            "Do not invent facts. Return plain text only, under 150 words."
        )
        user = f"PAGE {pnum} of {fname} ({dtype}):\n{ptext[:3000]}\n\nSummary of this page:"
        try:
            page_summary = _ollama_chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}]
            ).strip()
            if page_summary:
                db.save_psychology_page_summary(doc_id, pnum, page_summary)
                page_summaries.append((doc_id, fname, dtype, pnum, page_summary))
        except Exception:
            pass

    if not page_summaries:
        return "No psychology content could be summarized."

    # ---- Pass 2: cumulative accumulation across pages ----
    running_summary = ""
    total = len(page_summaries)
    for idx, (doc_id, fname, dtype, pnum, psummary) in enumerate(page_summaries, 1):
        system = (
            "You maintain a concise clinical summary of a young autistic adult with severe OCD. "
            "Merge the NEW PAGE SUMMARY into the PREVIOUS SUMMARY. "
            "Keep it under 200 words. Focus on diagnoses, current therapy approach, goals, progress, concerns. "
            "Do not invent facts. Return plain text only."
        )
        user = (
            f"PREVIOUS SUMMARY:\n{running_summary or '(none yet)'}\n\n"
            f"NEW PAGE SUMMARY (page {pnum} of {fname}, {dtype}):\n{psummary}\n\n"
            "Updated summary:"
        )
        try:
            new_summary = _ollama_chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}]
            ).strip()
            if new_summary:
                running_summary = new_summary
        except Exception:
            pass

    # ---- Pass 3: final polish pass on the complete accumulated summary ----
    if running_summary:
        system = (
            "You are producing a final, concise clinical summary of a young autistic adult with severe OCD. "
            "Polish the following accumulated summary into a clear, well-organized paragraph. "
            "Keep it under 200 words. Focus on diagnoses, current therapy approach, goals, progress, concerns. "
            "Do not invent facts. Return plain text only."
        )
        user = f"ACCUMULATED SUMMARY:\n{running_summary}\n\nFinal polished summary:"
        try:
            final_summary = _ollama_chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}]
            ).strip()
            if final_summary:
                running_summary = final_summary
        except Exception:
            pass

    return running_summary or "No psychology content could be summarized."


def summarize_and_clear_observations():
    try:
        rows = db.list_observations(limit=10000)
    except Exception:
        return "Could not load observations."
    rows = sorted(rows, key=lambda r: r.get("id") or 0)
    deduped = _dedupe_observations(rows)
    if not deduped:
        return db.get_latest_observation_summary() or "No observations to summarize."
    previous_summary = db.get_latest_observation_summary() or ""
    lines = []
    for r in deduped:
        obs = r.get("observation") or ""
        q = r.get("question") or ""
        raw = r.get("raw_text") or ""
        lines.append("- observation: " + obs + " | question: " + q + " | raw: " + raw)
    observations_text = "\n".join(lines)
    new_summary = summarize_observations(previous_summary, observations_text)
    if not new_summary:
        return previous_summary or "Summarization failed."
    db.save_observation_summary(new_summary, len(deduped))
    try:
        chroma_store.add_text("obs_summary_latest", new_summary, {"type": "observation_summary"})
    except Exception:
        pass
    for r in deduped:
        try:
            db.delete_observation(r.get("id"))
        except Exception:
            pass
        try:
            chroma_store.delete("obs_" + str(r.get("id")))
        except Exception:
            pass
    return new_summary
