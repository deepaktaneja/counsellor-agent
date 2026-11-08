import streamlit as st
import sqlite3
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

from config import DB_PATH as APP_DB_PATH, WHISPER_INITIAL_PROMPT
from memory.db import save_profile, get_profile, add_profile_history, list_profile_history, _now, get_latest_observation_summary, get_latest_psychology_summary, save_psychology_summary, psychology_text, assessments_text, therapy_text
from agents.advisor import advise, extract_observation_question, compact_observations, summarize_and_clear_observations
from ingestion.pipeline import ingest_pdf, ingest_image
import tempfile, os as _os


st.set_page_config(page_title="AI Counsellor", layout="wide")

env_path = Path(".env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()


def get_db():
    conn = sqlite3.connect(str(APP_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, raw_text TEXT, summary TEXT, tags TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, uploaded_at TEXT, doc_type TEXT, category TEXT, summary TEXT, full_text TEXT)")
    conn.commit()
    conn.close()

def load_observations():
    conn = get_db(); c = conn.cursor(); c.execute("SELECT * FROM observations ORDER BY id DESC"); rows = c.fetchall(); conn.close()
    return [dict(r) for r in rows]

def load_documents():
    conn = get_db(); c = conn.cursor(); c.execute("SELECT * FROM documents ORDER BY id DESC"); rows = c.fetchall(); conn.close()
    return [dict(r) for r in rows]

def save_observation(raw_text, summary, tags=""):
    conn = get_db(); c = conn.cursor()
    c.execute("INSERT INTO observations (created_at, raw_text, summary, tags) VALUES (?,?,?,?)",
              (datetime.now().isoformat(), raw_text, summary, tags))
    conn.commit(); conn.close()

def save_document(filename, doc_type, summary, full_text=""):
    conn = get_db(); c = conn.cursor()
    c.execute("INSERT INTO documents (filename, uploaded_at, doc_type, summary, full_text) VALUES (?,?,?,?,?)",
              (filename, datetime.now().isoformat(), doc_type, summary, full_text))
    conn.commit(); conn.close()


def is_duplicate_observation(observation: str, question: str) -> bool:
    """Best-effort duplicate check against recent observations."""
    try:
        from memory.db import list_observations
        for row in list_observations(limit=20):
            if (row.get("observation") or "") == (observation or "") and (row.get("question") or "") == (question or ""):
                return True
    except Exception:
        pass
    return False


def save_turn_to_db(observation: str, question: str, transcript: str) -> bool:
    """Save an observation turn to the same DB that page_observations reads from."""
    if not (observation or question):
        return False
    if is_duplicate_observation(observation or "", question or ""):
        return False
    from memory.db import save_observation
    save_observation(
        raw_text=transcript or "",
        observation=observation or "",
        question=question or "",
        summary=(observation or question or transcript or "")[:200],
        tags=["advisor"],
    )
    return True

def save_medication(name, dose, frequency, prescriber, start_date, end_date, status="current", notes=""):
    conn = get_db(); c = conn.cursor()
    c.execute("INSERT INTO medications (name, dose, frequency, prescriber, start_date, end_date, status, notes) VALUES (?,?,?,?,?,?,?,?)",
              (name, dose, frequency, prescriber, start_date, end_date, status, notes))
    conn.commit(); conn.close()

import faster_whisper

@st.cache_resource
def load_whisper_model():
    return faster_whisper.WhisperModel(os.getenv("WHISPER_MODEL", "base"), compute_type="float32")

whisper_model = load_whisper_model()

def _detect_ext(buf):
    if buf[:4] == b'RIFF' and buf[8:12] == b'WAVE': return '.wav'
    if buf[:2] == b'\xff\xfb': return '.mp3'
    if buf[:4] in (b'ogg ', b'Opus'): return '.ogg'
    if buf[:4] == b'fLaC': return '.flac'
    if buf[:4] == b'\x1A\x45\xF3\x0A': return '.webm'  # WebM Opus/VP8
    return '.wav'

def transcribe_audio(audio_bytes):
    try:
        if hasattr(audio_bytes, "read"):
            audio_bytes = audio_bytes.read()
        if not audio_bytes:
            st.warning("No audio data received.")
            return ""
        ext = _detect_ext(audio_bytes)
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(audio_bytes); tmp.close()
        segments, info = whisper_model.transcribe(tmp.name, beam_size=5, initial_prompt=WHISPER_INITIAL_PROMPT)
        text = "".join(s.text for s in segments).strip()
        os.unlink(tmp.name)
        # Normalize common Whisper mis-hearings of "Anmol"
        for wrong in ("anmol", "a mole", "an all", "anmol ", "anmol.", "anmol,"):
            text = text.replace(wrong, "Anmol")
        # Also fix "my son" -> keep as is, but ensure Anmol is capitalized
        if not text:
            st.warning("Whisper returned no text.")
        return text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        import traceback; traceback.print_exc()
        return ""

def clean_for_tts(text):
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\_(.+?)\_', r'\1', text)
    text = re.sub(r'\`(.+?)\`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'\1', text)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\-\*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s.,?!;:\'\\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bwill\b', 'wil', text, flags=re.IGNORECASE)
    text = text.strip()
    if text and not any(text.endswith(p) for p in ".!?"):
        text += "."
    return text

def synthesize(text, backend=None, voice=None):
    text = clean_for_tts(text)
    backend = backend or os.getenv("TTS_BACKEND", "say")
    voice = voice or os.getenv("TTS_VOICE", "Aman")
    try:
        if backend == "say":
            txt_file = tempfile.mktemp(suffix=".txt")
            wav_file = tempfile.mktemp(suffix=".wav")
            with open(txt_file, "w") as f:
                f.write(text)
            wav_ok = False
            # Try specified voice first, then fall back to default system voice
            for v in ([voice] if voice else []) + [None]:
                try:
                    cmd = ["say", "-o", wav_file, txt_file]
                    if v:
                        cmd = ["say", "-v", v, "-o", wav_file, txt_file]
                    subprocess.run(cmd, check=True, capture_output=True)
                    wav_ok = True
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

            if not wav_ok:
                # say command unavailable or voice not installed; TTS is optional
                st.info("🔊 Audio playback not available on this system; advice text is shown instead")
                for fp in [txt_file, wav_file]:
                    try:
                        os.unlink(fp)
                    except Exception:
                        pass
                return None

            with open(wav_file, "rb") as f:
                wav_bytes = f.read()
            for fp in [txt_file, wav_file]:
                try:
                    os.unlink(fp)
                except Exception:
                    pass
            return wav_bytes
        return None
    except Exception:
        return None

def distress_response():
    return "I'm concerned about what you've shared. Please consult with his counsellor."

def check_distress(text):
    keywords = ["suicide", "kill myself", "end my life", "hurt myself", "die", "self-harm", "overdose", "cutting"]
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower: return True
    return False

# Advisor page
def page_advisor():
    st.title("🎤 Voice Advisor")
    
    with st.sidebar:
        st.header("Profile Summary (from data)")
        profile = get_profile() or {}
        if profile.get("age"): st.caption(f"Age: {profile.get('age')}")
        if profile.get("ocd_themes"): st.caption(f"OCD themes: {profile.get('ocd_themes')[:80]}...")
        if profile.get("deescalation"): st.caption(f"Calms: {profile.get('deescalation')[:60]}...")

        st.markdown("---")
        st.header("Observation Summary")
        _os = get_latest_observation_summary()
        if _os:
            st.caption(_os)
        else:
            st.caption("None yet.")
    
    chat_mode = st.toggle("Chat mode (off = Recording)", value=False, key="advisor_chat_mode")

    if chat_mode:
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []

        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        prompt = st.chat_input("Type your question…")
        if prompt and prompt.strip():
            user_text = prompt.strip()
            st.session_state["chat_messages"].append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.write(user_text)
            with st.spinner("Generating advice..."):
                _obs_summary = compact_observations()
                st.session_state["last_obs_summary"] = _obs_summary
                advice = advise(question=user_text, extra_context=_obs_summary)
                save_turn_to_db(observation="", question=user_text, transcript=user_text)
            st.session_state["chat_messages"].append({"role": "assistant", "content": advice})
            with st.chat_message("assistant"):
                st.write(advice)
                wav = synthesize(advice)
                if wav:
                    st.audio(wav, format="audio/wav", autoplay=True)

        if st.button("Clear chat", key="clear_chat"):
            st.session_state["chat_messages"] = []
            st.rerun()
        return

    st.markdown("**Press & hold to speak**")
    audio_bytes = st.audio_input("🎤", label_visibility="collapsed")
    
    if audio_bytes is not None:
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(audio_bytes)
        if transcript:
            split = extract_observation_question(transcript)
            st.session_state["observation"] = split.get("observation", "")
            st.session_state["question"] = split.get("question", transcript) or transcript
        else:
            st.warning("Transcription returned no text. Please try again.")
    
    q = st.session_state.get("question", "")
    obs = st.session_state.get("observation", "")
    c1, c2 = st.columns(2)
    with c1:
        obs = st.text_area("📝 What did he do/say? (observation, to save)", value=obs, height=80)
    with c2:
        quest = st.text_area("❓ What do I do right now? (your question)", value=q, height=80)
    
    if st.button("🔍 Get Advice") and (obs.strip() or quest.strip()):
        with st.spinner("Generating advice..."):
            _obs_summary = compact_observations()
            st.session_state["last_obs_summary"] = _obs_summary
            advice = advise(observation=obs.strip(), question=quest.strip(), extra_context=_obs_summary)
            save_turn_to_db(
                observation=obs.strip(),
                question=quest.strip(),
                transcript=obs.strip() + " " + quest.strip(),
            )
        st.session_state["advice"] = advice
        st.session_state["advice_played"] = False
        st.session_state["orig_advice"] = advice
        st.session_state["orig_question"] = quest.strip()
        if "conversation" not in st.session_state:
            st.session_state["conversation"] = []
        st.session_state["conversation"].append({
            "question": quest.strip(),
            "observation": obs.strip(),
            "advice": advice,
        })
        st.rerun()

    # Display existing advice
    if st.session_state.get("advice"):
        st.markdown("### Advice")
        st.write(st.session_state["advice"])
        _ctx = st.session_state.get("last_obs_summary")
        if _ctx:
            with st.expander("Additional context sent to LLM (observation summary)"):
                st.write(_ctx)
        if not st.session_state.get("advice_played", False):
            wav = synthesize(st.session_state["advice"])
            if wav:
                st.audio(wav, format="audio/wav", autoplay=True)
                st.session_state["advice_played"] = True
        


    # Follow-up section - always available after first advice
    if st.session_state.get("advice"):
        st.markdown("---")
        st.markdown("**Follow-up question**")
        fu_audio = st.audio_input("🎤", label_visibility="collapsed", key="fu_audio_input")
        if fu_audio is not None:
            with st.spinner("Transcribing follow-up..."):
                fu_transcript = transcribe_audio(fu_audio)
            if fu_transcript:
                st.session_state["fu_question"] = fu_transcript
            else:
                st.warning("Transcription returned no text. Please try again.")

        fu_q = st.session_state.get("fu_question", "")
        fu_text = st.text_input("Or type your follow-up question:", value=fu_q, key="fu_text_input")

        if st.button("💬 Submit Follow-up") and fu_text.strip():
            with st.spinner("Generating follow-up advice..."):
                conv = st.session_state.get("conversation", [])
                prev_q = conv[-1]["question"] if conv else ""
                prev_a = conv[-1]["advice"] if conv else ""
                _obs_summary = compact_observations()
                st.session_state["last_obs_summary"] = _obs_summary
                fu_advice = advise(observation="", question=fu_text.strip(), prev_question=prev_q, prev_advice=prev_a, extra_context=_obs_summary)
            st.session_state["fu_advice"] = fu_advice
            st.session_state["fu_advice_played"] = False
            st.session_state["conversation"].append({
                "question": fu_text.strip(),
                "observation": "",
                "advice": fu_advice,
            })
            st.session_state["fu_question"] = ""
            st.rerun()

    # Display follow-up advice
    if st.session_state.get("fu_advice"):
        st.markdown("**Follow-up advice:**")
        st.write(st.session_state["fu_advice"])
        if not st.session_state.get("fu_advice_played", False):
            wav2 = synthesize(st.session_state["fu_advice"])
            if wav2:
                st.audio(wav2, format="audio/wav", autoplay=True)
                st.session_state["fu_advice_played"] = True
    if st.button("Start New Query"):
        for key in list(st.session_state.keys()):
            if "advice" in key or "play" in key or "question" in key or "orig" in key or "fu_" in key or "conversation" in key or "observation" in key:
                del st.session_state[key]
        st.rerun()

def page_profile():
    st.title("👤 Son Profile")
    profile = get_profile() or {}
    
    with st.form("profile_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Name", value=profile.get("name") or "")
            age = st.number_input("Age", 0, 100, int(profile.get("age") or 20))
            height_cm = st.number_input("Height (cm)", 0, 250, int(profile.get("height_cm") or 0))
            weight_kg = st.number_input("Weight (kg)", 0, 300, int(profile.get("weight_kg") or 0))
            comm_style = st.text_area("Communication style", value=profile.get("communication_style") or "")
            ocd_themes = st.text_area("OCD themes & compulsions", value=profile.get("ocd_themes") or "")


        with c2:
            likes = st.text_area("Likes", value=profile.get("likes") or "")
            dislikes = st.text_area("Dislikes", value=profile.get("dislikes") or "")
            hobbies = st.text_area("Hobbies", value=profile.get("hobbies") or "")
            sensory = st.text_area("Sensory notes", value=profile.get("sensory_notes") or "")
            deescalation = st.text_area("What calms him / de-escalation", value=profile.get("deescalation") or "")
        
        family_context = st.text_area("Family & social context", value=profile.get("family_context") or "")
        notes = st.text_area("Additional notes / daily life", value=profile.get("notes") or "")
        
        submitted = st.form_submit_button("Save Profile")
        if submitted:
            save_profile(
                age=age if age > 0 else None,
                height_cm=height_cm if height_cm > 0 else None,
                weight_kg=weight_kg if weight_kg > 0 else None,
                communication_style=comm_style or "",
                likes=likes or "",
                dislikes=dislikes or "",
                hobbies=hobbies or "",
                sensory_notes=sensory or "",
                emergency_contact=profile.get("emergency_contact", ""),
                notes=notes or "",
                ocd_themes=ocd_themes or "",
                triggers=profile.get("triggers", ""),
                deescalation=deescalation or "",
                family_context=family_context or "",
                name=name or ""
            )
            add_profile_history(
                height_cm=height_cm if height_cm > 0 else None,
                weight_kg=weight_kg if weight_kg > 0 else None,
                notes=notes or "",
            )

            st.success("✅ Profile saved!")
            st.rerun()
    
    st.markdown("### Profile History")
    history = list_profile_history()
    if history:
        for h in history:
            with st.expander(f"Updated at {h.get('recorded_at','')}"):
                st.write(f"Height: {h.get('height_cm')} cm, Weight: {h.get('weight_kg')} kg, Notes: {h.get('notes','')}")
    else:
        st.info("No profile history yet.")

def page_medications():
    st.title("💊 Medications")
    from memory.db import (list_medications, save_medication, update_medication,
                           delete_medication, list_documents, update_document_category)

    with st.sidebar:
        st.header("Data")
        c1, c2 = st.columns(2)
        c1.metric("Medications", len(list_medications()))
        c2.metric("Med Docs", len(list_documents(category="medication")))

    _upload_widget("meds")

    st.markdown("### Current / All Medications")
    meds = list_medications()
    if meds:
        for m in meds:
            with st.expander(f"{m.get('name','')} — {m.get('dose','')} ({m.get('status','')})"):
                st.write(f"**Frequency:** {m.get('frequency','')}")
                st.write(f"**Prescriber:** {m.get('prescriber','')}")
                st.write(f"**Start:** {m.get('start_date','')}  **End:** {m.get('end_date','')}")
                st.write(f"**Notes:** {m.get('notes','')}")
                c1, c2 = st.columns(2)
                if c1.button("Mark past", key=f"past_{m['id']}"):
                    update_medication(m["id"], status="past")
                    st.rerun()
                if c2.button("Delete", key=f"delmed_{m['id']}"):
                    delete_medication(m["id"])
                    st.rerun()
    else:
        st.info("No medications recorded yet.")

    with st.expander("➕ Add medication manually"):
        with st.form("add_med_form"):
            name = st.text_input("Name")
            dose = st.text_input("Dose")
            freq = st.text_input("Frequency")
            presc = st.text_input("Prescriber")
            start = st.text_input("Start date")
            end = st.text_input("End date")
            status = st.selectbox("Status", ["current", "past"])
            notes = st.text_area("Notes")
            if st.form_submit_button("Save"):
                if name.strip():
                    save_medication(name=name.strip(), dose=dose, frequency=freq,
                                    prescriber=presc, start_date=start, end_date=end,
                                    status=status, notes=notes)
                    st.success("Saved")
                    st.rerun()

    st.markdown("### Medication Documents")
    docs = list_documents(category="medication")
    if docs:
        for d in docs:
            with st.expander(f"{d.get('filename','Unknown')} ({d.get('doc_type','')})"):
                st.caption(f"Uploaded: {d.get('uploaded_at','')}")
                st.write(d.get("summary", "")[:500])
                if st.session_state.get(f"recls_{d['id']}"):
                    newcat = st.selectbox("New category", ["medication", "assessment", "therapy", "other"],
                                          key=f"nc_{d['id']}")
                    if st.button("Apply", key=f"app_{d['id']}"):
                        update_document_category(d["id"], newcat)
                        st.rerun()
                else:
                    if st.button("Reclassify", key=f"reclsbtn_{d['id']}"):
                        st.session_state[f"recls_{d['id']}"] = True
                        st.rerun()
                # delete button for medication document
                from memory.db import delete_document_cascade
                from memory import chroma_store
                did = d.get("id")
                cdel_key = f"confirm_delmed_{did}"
                if st.button("🗑️ Delete document", key=f"btn_delmed_{did}"):
                    st.session_state[cdel_key] = True
                    st.rerun()
                if st.session_state.get(cdel_key):
                    st.warning("Delete this medication document? Linked medications will also be removed.")
                    cc1, cc2 = st.columns([1, 3])
                    if cc1.button("Yes, delete", key=f"yes_delmed_{did}"):
                        try:
                            chroma_store.delete("doc_%s" % did)
                            chroma_store.delete_by_parent(did)
                        except Exception:
                            pass
                        delete_document_cascade(did)
                        st.session_state.pop(cdel_key, None)
                        st.success("Deleted.")
                        st.rerun()
                    if cc2.button("Cancel", key=f"no_delmed_{did}"):
                        st.session_state.pop(cdel_key, None)
                        st.rerun()
    else:
        st.info("No medication documents yet.")

def _handle_upload(uploaded, category_override=None):
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            return ingest_pdf(tmp_path, uploaded.name, category_override=category_override)
        else:
            return ingest_image(tmp_path, uploaded.name, category_override=category_override)
    finally:
        try:
            _os.remove(tmp_path)
        except Exception:
            pass


def _upload_widget(key_prefix):
    st.markdown("### Upload PDFs / Images")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Choose file", type=["pdf", "jpg", "png", "jpeg"], key=f"up_{key_prefix}")
    with col2:
        choice = st.selectbox(
            "Classify as",
            ["Auto-detect", "Medication", "Assessment", "Therapy", "Other"],
            key=f"cls_{key_prefix}",
        )
    if uploaded is not None:
        cat_map = {"Auto-detect": None, "Medication": "medication",
                   "Assessment": "assessment", "Therapy": "therapy", "Other": "other"}
        co = cat_map.get(choice)
        if st.button("Ingest", key=f"ing_{key_prefix}"):
            with st.spinner("Ingesting..."):
                res = _handle_upload(uploaded, category_override=co)
            msg = f"Ingested as '{res.get('category','')}' (doc_type: {res.get('doc_type','')})"
            if res.get("pages"):
                msg += f" — Read {res['pages']} page(s), {res.get('pages_with_text', res['pages'])} with text."
            st.success(msg)
            if res.get("pages") and res.get("pages_with_text", res["pages"]) < res["pages"]:
                st.warning(f"{res['pages'] - res.get('pages_with_text', 0)} page(s) had no extractable text (possibly scanned images).")
            st.caption((res.get("data", {}) or {}).get("summary", "")[:300])
            st.rerun()



def page_observations():
    st.title("📝 Observations History")
    _summary = get_latest_observation_summary()
    with st.expander("🧠 Latest observation summary", expanded=bool(_summary)):
        if _summary:
            st.write(_summary)
        else:
            st.info("No observation summary yet.")
    obs = load_observations()
    if obs:
        for o in obs:
            with st.expander(f"At {o.get('created_at','')} - {o.get('summary','')[:60]}"):
                st.write(f"Notes: {o.get('summary','')}")
                st.write(f"Full text: {o.get('raw_text','')[:200]}")
    else:
        st.info("No observations yet.")

    if st.button("🧠 Summarize & clear observations", key="summarize_clear_obs"):
        with st.spinner("Summarizing observations..."):
            summary = summarize_and_clear_observations()
        st.success("Observations summarized and cleared.")
        if summary:
            st.markdown("**Summary saved:**")
            st.write(summary)
        st.rerun()

def _render_doc_card(d, category):
    from memory import chroma_store
    from memory.db import delete_document_cascade, list_documents
    did = d.get("id")
    fname = d.get("filename", "Unknown")
    doc_type = d.get("doc_type", "")
    uploaded = d.get("uploaded_at", "")
    summary = d.get("summary", "") or ""
    full_text = d.get("full_text", "") or ""

    # count chroma chunks for this document (page-level chunks)
    try:
        _res = chroma_store._collection_instance().get(where={"parent_id": did})
        chunk_ids = _res.get("ids", [])
    except Exception:
        chunk_ids = []
    # count distinct pages from metadata
    try:
        _metas = chroma_store._collection_instance().get(where={"parent_id": did}).get("metadatas", [])
        pages = sorted({m.get("page") for m in _metas if m and m.get("page")})
    except Exception:
        pages = []
    page_label = "pages %s" % (", ".join(str(p) for p in pages) if pages else "0")

    with st.expander(f"📄 {fname} ({doc_type}) — {page_label}", expanded=False):
        st.caption(f"Uploaded: {uploaded}  |  ID: {did}  |  Chroma chunks: {len(chunk_ids)}")
        st.markdown("**Summary:**")
        st.write(summary or "_No summary extracted._")
        # per-page text
        st.markdown("**Pages:**")
        page_marker = "--- Page "
        parts = full_text.split(page_marker)
        if len(parts) > 1:
            for part in parts[1:]:
                pnum = part.split(" of ")[0].strip()
                ptext = part.split("\n", 1)[1] if "\n" in part else part
                st.markdown(f"**Page {pnum}:**")
                st.text(ptext[:1200])
        elif full_text.strip():
            st.text(full_text[:1200])
        else:
            st.caption("No full text stored.")

        # delete action
        c1, c2 = st.columns([1, 4])
        confirm_key = f"confirm_delete_{did}"
        if c1.button("🗑️ Delete document", key=f"btn_delete_{did}"):
            st.session_state[confirm_key] = True
            st.rerun()
        if st.session_state.get(confirm_key):
            st.warning("Delete this document? This also removes linked diagnoses/medications/findings.")
            cc1, cc2 = st.columns([1, 3])
            if cc1.button("Yes, delete", key=f"yes_delete_{did}"):
                try:
                    chroma_store.delete("doc_%s" % did)
                    chroma_store.delete_by_parent(did)
                except Exception:
                    pass
                delete_document_cascade(did)
                st.session_state.pop(confirm_key, None)
                st.success("Deleted.")
                st.rerun()
            if cc2.button("Cancel", key=f"no_delete_{did}"):
                st.session_state.pop(confirm_key, None)
                st.rerun()


def page_psychology():
    st.title("🧠 Psychology & Therapy")
    from memory.db import (list_documents, update_document_category,
                           get_latest_psychology_summary, save_psychology_summary,
                           psychology_text)

    with st.sidebar:
        st.header("Data")
        c1, c2 = st.columns(2)
        c1.metric("Assessments", len(list_documents(category="assessment")))
        c2.metric("Therapy", len(list_documents(category="therapy")))

    _upload_widget("psych")

    tabA, tabT = st.tabs(["Assessments", "Therapy"])
    with tabA:
        docs = list_documents(category="assessment")
        if docs:
            for d in docs:
                _render_doc_card(d, category="assessment")
        else:
            st.info("No assessment documents yet.")
    with tabT:
        docs = list_documents(category="therapy")
        if docs:
            for d in docs:
                _render_doc_card(d, category="therapy")
        else:
            st.info("No therapy documents yet.")

    st.markdown("---")
    st.markdown("### Consolidated Psychology Summary")
    _ps = get_latest_psychology_summary()
    if _ps:
        st.write(_ps)
    else:
        st.caption("None yet.")
    # show per-page summaries for verification
    try:
        from memory.db import get_psychology_page_summaries
        _page_sums = get_psychology_page_summaries()
        if _page_sums:
            with st.expander(f"View per-page summaries ({len(_page_sums)} pages)"):
                for ps in _page_sums:
                    st.markdown(f"**Doc {ps['doc_id']} — Page {ps['page_num']}:**")
                    st.write(ps["page_summary"])
    except Exception:
        pass
    if st.button("🧠 Generate / refresh summary"):
        from agents.advisor import summarize_psychology_incrementally
        from memory.db import list_documents
        # report how many pages were read so the user can confirm all pages are included
        total_pages = 0
        for cat in ("assessment", "therapy"):
            for d in list_documents(category=cat):
                ft = d.get("full_text") or ""
                total_pages += ft.count("--- Page ")
        try:
            with st.spinner(f"Summarizing {total_pages} page(s) one at a time..."):
                new = summarize_psychology_incrementally()
            if new and new not in ("No psychology assessments or therapy notes recorded.",
                                   "No psychology content found.",
                                   "No psychology content could be summarized."):
                save_psychology_summary(new, len(list_documents(category="assessment")) + len(list_documents(category="therapy")))
                st.success("Saved.")
                st.caption(f"Pages read from uploaded documents: {total_pages}")
                st.write(new)
                st.rerun()
            else:
                st.info(new)
        except Exception as e:
            st.error(f"Failed: {e}")



def main():
    init_db()
    with st.sidebar:
        st.title("🤖 AI Counsellor")
        page = st.radio("Go to", ["Advisor", "Medications", "Psychology & Therapy", "Profile", "Observations"], index=0)
    if page == "Advisor": page_advisor()
    elif page == "Medications": page_medications()
    elif page == "Psychology & Therapy": page_psychology()
    elif page == "Profile": page_profile()
    elif page == "Observations": page_observations()

if __name__ == "__main__": main()
