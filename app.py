import streamlit as st
import sqlite3
import os
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="AI Counsellor", layout="wide")

env_path = Path(".env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

DB_PATH = Path("memory/counsellor.db")

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, raw_text TEXT, summary TEXT, tags TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, uploaded_at TEXT, doc_type TEXT, summary TEXT, full_text TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS profile (id INTEGER PRIMARY KEY, updated_at TEXT, age INTEGER, height_cm REAL, weight_kg REAL, communication_style TEXT, likes TEXT, dislikes TEXT, hobbies TEXT, sensory_notes TEXT, emergency_contact TEXT, notes TEXT, ocd_themes TEXT, triggers TEXT, deescalation TEXT, family_context TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS profile_history (id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT, height_cm REAL, weight_kg REAL, notes TEXT)")
    conn.commit()
    conn.close()

def load_profile():
    conn = get_db(); c = conn.cursor(); c.execute("SELECT * FROM profile WHERE id=1"); row = c.fetchone(); conn.close()
    if row:
        cols = [d[0] for d in c.description]; return dict(zip(cols, row))
    return {}

def load_observations():
    conn = get_db(); c = conn.cursor(); c.execute("SELECT * FROM observations ORDER BY id DESC"); rows = c.fetchall(); conn.close()
    return [dict(r) for r in rows]

def load_documents():
    conn = get_db(); c = conn.cursor(); c.execute("SELECT * FROM documents ORDER BY id DESC"); rows = c.fetchall(); conn.close()
    return [dict(r) for r in rows]

def load_profile_history():
    conn = get_db(); c = conn.cursor(); c.execute("SELECT * FROM profile_history ORDER BY id DESC LIMIT 30"); rows = c.fetchall(); conn.close()
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
    return '.wav'

def transcribe_audio(audio_bytes):
    try:
        ext = _detect_ext(audio_bytes)
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(audio_bytes); tmp.close()
        segments, info = whisper_model.transcribe(tmp.name, beam_size=5)
        text = "".join(s.text for s in segments).strip()
        os.unlink(tmp.name)
        return text
    except Exception as e:
        st.error(f"Transcription error: {e}"); return ""

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
            subprocess.run(["say", "-v", voice, "-o", wav_file, txt_file], check=True, capture_output=True)
            with open(wav_file, "rb") as f:
                wav_bytes = f.read()
            for fp in [txt_file, wav_file]:
                try: os.unlink(fp)
                except: pass
            return wav_bytes
        return None
    except Exception as e:
        st.error(f"TTS error: {e}"); return None

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
        profile = load_profile()
        if profile.get("age"): st.caption(f"Age: {profile.get('age')}")
        if profile.get("ocd_themes"): st.caption(f"OCD themes: {profile.get('ocd_themes')[:80]}...")
        if profile.get("deescalation"): st.caption(f"Calms: {profile.get('deescalation')[:60]}...")
    
    st.markdown("**Press & hold to speak**")
    audio_bytes = st.audio_input("🎤", label_visibility="collapsed")
    
    if audio_bytes is not None:
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(audio_bytes)
        st.session_state["question"] = transcript
        st.session_state["observation"] = ""
    
    q = st.session_state.get("question", "")
    c1, c2 = st.columns(2)
    with c1:
        obs = st.text_area("📝 What did he do/say? (observation, to save)", value="", height=80)
    with c2:
        quest = st.text_area("❓ What do I do right now? (your question)", value=q, height=80)
    
    if st.button("🔍 Get Advice") and (obs.strip() or quest.strip()):
        with st.spinner("Generating advice..."):
            advice = advise(observation=obs.strip(), question=quest.strip())
        st.session_state["advice"] = advice
        st.session_state["advice_played"] = False
        st.session_state["orig_advice"] = advice
        st.session_state["orig_question"] = quest.strip()
        st.markdown("### Advice")
        st.write(advice)
        if not st.session_state.get("advice_played", False):
            wav = synthesize(advice)
            if wav:
                st.audio(wav, format="audio/wav", autoplay=True)
                st.session_state["advice_played"] = True
        
        st.markdown("---")
        st.markdown("**Follow-up question**")
        fu_audio = st.audio_input("🎤", label_visibility="collapsed", key="fu_audio_input")
        if fu_audio is not None:
            with st.spinner("Transcribing follow-up..."):
                fu_transcript = transcribe_audio(fu_audio)
            st.session_state["fu_question"] = fu_transcript
        fu_q = st.session_state.get("fu_question", "")
        if fu_q.strip():
            with st.spinner("Generating follow-up advice..."):
                fu_advice = advise(observation="", question=fu_q.strip(), prev_question=st.session_state.get("orig_question",""), prev_advice=st.session_state.get("orig_advice",""))
            st.session_state["fu_advice"] = fu_advice
            st.session_state["fu_advice_played"] = False
            st.markdown("**Follow-up advice:**")
            st.write(st.session_state["fu_advice"])
            if not st.session_state.get("fu_advice_played", False):
                wav2 = synthesize(st.session_state["fu_advice"])
                if wav2:
                    st.audio(wav2, format="audio/wav", autoplay=True)
                    st.session_state["fu_advice_played"] = True
    
    if st.button("Start New Query"):
        for key in list(st.session_state.keys()):
            if "advice" in key or "play" in key or "question" in key or "orig" in key:
                del st.session_state[key]
        st.rerun()

def page_profile():
    st.title("👤 Son Profile")
    profile = load_profile()
    
    with st.form("profile_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 0, 100, profile.get("age") or 20)
            height_cm = st.number_input("Height (cm)", 0, 250, profile.get("height_cm") or 0)
            weight_kg = st.number_input("Weight (kg)", 0, 300, profile.get("weight_kg") or 0)
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
            save_profile({
                "age": age, "height_cm": height_cm if height_cm > 0 else None,
                "weight_kg": weight_kg if weight_kg > 0 else None,
                "communication_style": comm_style or "", "likes": likes or "",
                "dislikes": dislikes or "", "hobbies": hobbies or "",
                "sensory_notes": sensory or "", "emergency_contact": profile.get("emergency_contact"),
                "notes": notes or "", "ocd_themes": ocd_themes or "", "triggers": profile.get("triggers") or "",
                "deescalation": deescalation or "", "family_context": family_context or ""
            })
            st.success("✅ Profile saved!")
            st.rerun()
    
    st.markdown("### Profile History")
    history = load_profile_history()
    if history:
        for h in history:
            with st.expander(f"Updated at {h.get('recorded_at','')}"):
                st.write(f"Height: {h.get('height_cm')} cm, Weight: {h.get('weight_kg')} kg, Notes: {h.get('notes','')}")
    else:
        st.info("No profile history yet.")

def page_records():
    st.title("📄 Medical Records")
    profile = load_profile()
    
    with st.sidebar:
        st.header("Data")
        c1, c2 = st.columns(2)
        c1.metric("PDF Uploads", len(load_documents()))
        c2.metric("Observations", len(load_observations()))
    
    st.markdown("### Upload PDFs / Images")
    uploaded = st.file_uploader("Upload PDF or image", type=["pdf", "jpg", "png", "jpeg"], key="upload_records")
    if uploaded:
        with st.spinner("Ingesting..."):
            ingest_file(uploaded)
        st.success("✅ Ingested!")
        st.rerun()
    
    st.markdown("### All Medical Records")
    docs = load_documents()
    if docs:
        for doc in docs:
            with st.expander(doc.get("filename", "Unknown")):
                st.caption(f"Uploaded: {doc.get('uploaded_at','')} | Type: {doc.get('doc_type','')}")
                st.caption(doc.get("summary", "")[:200])
    else:
        st.info("No medical records yet.")
    
    if st.button("Export All Records"):
        st.download_button("Download CSV", "records.csv")

def page_observations():
    st.title("📝 Observations History")
    obs = load_observations()
    if obs:
        for o in obs:
            with st.expander(f"At {o.get('created_at','')} - {o.get('summary','')[:60]}"):
                st.write(f"Notes: {o.get('summary','')}")
                st.write(f"Full text: {o.get('raw_text','')[:200]}")
    else:
        st.info("No observations yet.")

def main():
    init_db()
    with st.sidebar:
        st.title("🤖 AI Counsellor")
        page = st.radio("Go to", ["Advisor", "Profile", "Records", "Observations"], index=0)
    if page == "Advisor": page_advisor()
    elif page == "Profile": page_profile()
    elif page == "Records": page_records()
    elif page == "Observations": page_observations()

if __name__ == "__main__": main()
