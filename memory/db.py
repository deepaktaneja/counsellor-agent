import sqlite3, json, datetime
from config import DB_PATH


def _conn():
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = _conn()
    cur = c.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        raw_text TEXT NOT NULL,
        observation TEXT,
        question TEXT,
        summary TEXT,
        tags TEXT,
        mood TEXT,
        trigger TEXT,
        outcome TEXT,
        embedding_id TEXT
    );
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        uploaded_at TEXT NOT NULL,
        doc_type TEXT,
        category TEXT,
        summary TEXT,
        full_text TEXT,
        file_path TEXT
    );
    CREATE TABLE IF NOT EXISTS medications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        dose TEXT,
        frequency TEXT,
        prescriber TEXT,
        start_date TEXT,
        end_date TEXT,
        status TEXT DEFAULT 'current',
        source_doc_id INTEGER,
        notes TEXT
    );
    CREATE TABLE IF NOT EXISTS diagnoses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        condition TEXT NOT NULL,
        diagnosed_by TEXT,
        date TEXT,
        notes TEXT,
        source_doc_id INTEGER
    );
    CREATE TABLE IF NOT EXISTS findings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        content TEXT NOT NULL,
        source_doc_id INTEGER
    );
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT NOT NULL,
        mode TEXT,
        transcript TEXT
    );
    CREATE TABLE IF NOT EXISTS observation_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        summary TEXT NOT NULL,
        obs_count INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS psychology_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        summary TEXT NOT NULL,
        doc_count INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS psychology_page_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL,
        page_num TEXT NOT NULL,
        page_summary TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)
    c.commit()
    try:
        c.execute("ALTER TABLE documents ADD COLUMN category TEXT")
        c.commit()
    except Exception:
        pass

    c.close()


def _now():
    return datetime.datetime.utcnow().isoformat()


def save_observation(raw_text, observation="", question="", summary="", tags=None,
                     mood="", trigger="", outcome="", embedding_id=None):
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO observations (created_at, raw_text, observation, question, summary, tags, mood, trigger, outcome, embedding_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (_now(), raw_text, observation, question, summary, json.dumps(tags or []), mood, trigger, outcome, embedding_id),
    )
    c.commit()
    rid = cur.lastrowid
    c.close()
    return rid


def list_observations(limit=100):
    c = _conn()
    rows = c.execute("SELECT * FROM observations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def delete_observation(oid):
    c = _conn()
    c.execute("DELETE FROM observations WHERE id=?", (oid,))
    c.commit()
    c.close()


def save_document(filename, doc_type="", summary="", full_text="", file_path="", category=""):
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO documents (filename, uploaded_at, doc_type, category, summary, full_text, file_path) VALUES (?,?,?,?,?,?,?)",
        (filename, _now(), doc_type, category, summary, full_text, file_path),
    )
    c.commit()
    rid = cur.lastrowid
    c.close()
    return rid


def list_documents(doc_type=None, category=None):
    c = _conn()
    clauses = []
    params = []
    if doc_type:
        clauses.append("doc_type=?")
        params.append(doc_type)
    if category:
        clauses.append("category=?")
        params.append(category)
    sql = "SELECT * FROM documents"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC"
    rows = c.execute(sql, tuple(params)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def list_documents_by_types(types):
    if not types:
        return []
    placeholders = ",".join("?" for _ in types)
    c = _conn()
    rows = c.execute(f"SELECT * FROM documents WHERE doc_type IN ({placeholders}) ORDER BY id DESC", tuple(types)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def update_document_type(did, doc_type):
    c = _conn()
    c.execute("UPDATE documents SET doc_type=? WHERE id=?", (doc_type, did))
    c.commit()
    c.close()


def update_document_category(did, category):
    c = _conn()
    c.execute("UPDATE documents SET category=? WHERE id=?", (category, did))
    c.commit()
    c.close()


def delete_document(did):
    c = _conn()
    c.execute("DELETE FROM documents WHERE id=?", (did,))
    c.commit()
    c.close()


def delete_document_cascade(did):
    """Delete a document and everything linked to it (diagnoses, medications, findings)."""
    c = _conn()
    c.execute("DELETE FROM documents WHERE id=?", (did,))
    c.execute("DELETE FROM diagnoses WHERE source_doc_id=?", (did,))
    c.execute("DELETE FROM medications WHERE source_doc_id=?", (did,))
    c.execute("DELETE FROM findings WHERE source_doc_id=?", (did,))
    c.execute("DELETE FROM psychology_page_summaries WHERE doc_id=?", (did,))
    c.commit()
    c.close()


def get_document(did):
    c = _conn()
    r = c.execute("SELECT * FROM documents WHERE id=?", (did,)).fetchone()
    c.close()
    return dict(r) if r else None
def save_medication(name, dose="", frequency="", prescriber="", start_date="", end_date="",
                    status="current", source_doc_id=None, notes=""):
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO medications (name, dose, frequency, prescriber, start_date, end_date, status, source_doc_id, notes) VALUES (?,?,?,?,?,?,?,?,?)",
        (name, dose, frequency, prescriber, start_date, end_date, status, source_doc_id, notes),
    )
    c.commit()
    rid = cur.lastrowid
    c.close()
    return rid


def list_medications(status=None):
    c = _conn()
    if status:
        rows = c.execute("SELECT * FROM medications WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    else:
        rows = c.execute("SELECT * FROM medications ORDER BY id DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]


def update_medication(mid, name=None, dose=None, frequency=None, prescriber=None,
                      start_date=None, end_date=None, status=None, notes=None):
    c = _conn()
    row = c.execute("SELECT * FROM medications WHERE id=?", (mid,)).fetchone()
    if not row:
        c.close()
        return False
    cur = c.cursor()
    cur.execute(
        """UPDATE medications SET
           name=COALESCE(?,name), dose=COALESCE(?,dose), frequency=COALESCE(?,frequency),
           prescriber=COALESCE(?,prescriber), start_date=COALESCE(?,start_date),
           end_date=COALESCE(?,end_date), status=COALESCE(?,status), notes=COALESCE(?,notes)
           WHERE id=?""",
        (name, dose, frequency, prescriber, start_date, end_date, status, notes, mid),
    )
    c.commit()
    c.close()
    return True


def delete_medication(mid):
    c = _conn()
    c.execute("DELETE FROM medications WHERE id=?", (mid,))
    c.commit()
    c.close()


def current_medications_text():
    meds = list_medications(status="current")
    if not meds:
        meds = list_medications()
    if not meds:
        return "No medications recorded."
    return "; ".join(f"{m['name']} {m['dose']} ({m['frequency']})" for m in meds)


def save_diagnosis(condition, diagnosed_by="", date="", notes="", source_doc_id=None):
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO diagnoses (condition, diagnosed_by, date, notes, source_doc_id) VALUES (?,?,?,?,?)",
        (condition, diagnosed_by, date, notes, source_doc_id),
    )
    c.commit()
    rid = cur.lastrowid
    c.close()
    return rid


def list_diagnoses():
    c = _conn()
    rows = c.execute("SELECT * FROM diagnoses ORDER BY id DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]


def diagnoses_text():
    dx = list_diagnoses()
    if not dx:
        return "No diagnoses recorded."
    return "; ".join(d["condition"] for d in dx)


def _docs_text_for_types(types, label, category=None, max_chars=15000):
    docs = []
    if category:
        docs = list_documents(category=category)
    legacy = list_documents_by_types(types)
    # merge unique
    seen = {d.get("id") for d in docs}
    for d in legacy:
        if d.get("id") not in seen:
            docs.append(d)
            seen.add(d.get("id"))
    if not docs:
        return f"No {label} recorded."
    parts = []
    total = 0
    for d in docs:
        s = (d.get("summary") or "").strip()
        if s:
            line = f"- {d.get('filename','doc')} ({d.get('doc_type','')}): {s}"
            parts.append(line)
            total += len(line)
        full_text = (d.get("full_text") or "").strip()
        if full_text:
            # include full page-tagged text so the LLM reads every page
            header = f"\nFULL TEXT for {d.get('filename','doc')} ({d.get('doc_type','')}):\n"
            block = header + full_text
            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    parts.append(block[:remaining] + "\n...[truncated]...")
                break
            parts.append(block)
            total += len(block)
        if total >= max_chars:
            break
    # also include findings linked to these docs
    try:
        all_findings = list_findings()
        doc_ids = {d.get("id") for d in docs}
        linked = [f for f in all_findings if f.get("source_doc_id") in doc_ids]
        if linked:
            parts.append("Findings:")
            for f in linked:
                parts.append(f"  - [{f.get('category','other')}] {f.get('content','')}")
    except Exception:
        pass
    return "\n".join(parts) if parts else f"No {label} recorded."


def assessments_text():
    return _docs_text_for_types(["assessment", "psych_report", "psychological_assessment"], "assessments", category="assessment")


def therapy_text():
    return _docs_text_for_types(["therapy_note", "therapy_session", "therapy_report"], "therapy notes", category="therapy")


def psychology_text():
    a = assessments_text()
    t = therapy_text()
    if a.startswith("No assessments") and t.startswith("No therapy"):
        return "No psychology assessments or therapy notes recorded."
    return f"ASSESSMENTS:\n{a}\n\nTHERAPY:\n{t}"


def save_finding(category, content, source_doc_id=None):
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO findings (category, content, source_doc_id) VALUES (?,?,?)",
        (category, content, source_doc_id),
    )
    c.commit()
    rid = cur.lastrowid

# ---- Profile (structured) ----
def init_profile_table():
    c = _conn()
    c.execute("""CREATE TABLE IF NOT EXISTS profile (
        id INTEGER PRIMARY KEY CHECK (id=1),
        updated_at TEXT NOT NULL,
        age INTEGER,
        height_cm REAL,
        weight_kg REAL,
        communication_style TEXT,
        likes TEXT,
        dislikes TEXT,
        hobbies TEXT,
        sensory_notes TEXT,
        emergency_contact TEXT,
        notes TEXT,
        ocd_themes TEXT,
        triggers TEXT,
        deescalation TEXT,
        family_context TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS profile_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at TEXT NOT NULL,
        height_cm REAL,
        weight_kg REAL,
        notes TEXT
    )""")
    # migrate older schemas: add missing columns
    existing = {r[1] for r in c.execute("PRAGMA table_info(profile)").fetchall()}
    for col in ("ocd_themes", "triggers", "deescalation", "family_context", "name"):
        if col not in existing:
            c.execute(f"ALTER TABLE profile ADD COLUMN {col} TEXT")
    c.commit()
    c.close()


def get_profile():
    c = _conn()
    r = c.execute("SELECT * FROM profile WHERE id=1").fetchone()
    c.close()
    return dict(r) if r else None


def save_profile(age=None, height_cm=None, weight_kg=None, communication_style="",
                 likes="", dislikes="", hobbies="", sensory_notes="",
                 emergency_contact="", notes="", ocd_themes="", triggers="",
                 deescalation="", family_context="", name=""):
    c = _conn()
    cur = c.cursor()
    cur.execute("""INSERT INTO profile (id, updated_at, age, height_cm, weight_kg, communication_style,
                    likes, dislikes, hobbies, sensory_notes, emergency_contact, notes,
                    ocd_themes, triggers, deescalation, family_context, name)
                   VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     updated_at=excluded.updated_at, age=excluded.age, height_cm=excluded.height_cm,
                     weight_kg=excluded.weight_kg, communication_style=excluded.communication_style,
                     likes=excluded.likes, dislikes=excluded.dislikes, hobbies=excluded.hobbies,
                     sensory_notes=excluded.sensory_notes, emergency_contact=excluded.emergency_contact,
                     notes=excluded.notes, ocd_themes=excluded.ocd_themes,
                     triggers=excluded.triggers, deescalation=excluded.deescalation,
                     family_context=excluded.family_context, name=excluded.name""",
                (_now(), age, height_cm, weight_kg, communication_style, likes, dislikes,
                 hobbies, sensory_notes, emergency_contact, notes, ocd_themes, triggers,
                 deescalation, family_context, name))
    c.commit()
    c.close()


def add_profile_history(height_cm=None, weight_kg=None, notes=""):
    c = _conn()
    cur = c.cursor()
    cur.execute("INSERT INTO profile_history (recorded_at, height_cm, weight_kg, notes) VALUES (?,?,?,?)",
                (_now(), height_cm, weight_kg, notes))
    c.commit()
    rid = cur.lastrowid
    c.close()
    return rid


def list_profile_history(limit=50):
    c = _conn()
    rows = c.execute("SELECT * FROM profile_history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def profile_text():
    p = get_profile()
    if not p:
        return "No structured profile recorded."
    parts = []
    if p.get("age") is not None:
        parts.append(f"Age: {p['age']}")
    if p.get("height_cm") is not None:
        parts.append(f"Height: {p['height_cm']} cm")
    if p.get("weight_kg") is not None:
        parts.append(f"Weight: {p['weight_kg']} kg")
    if p.get("communication_style"):
        parts.append(f"Communication: {p['communication_style']}")
    if p.get("likes"):
        parts.append(f"Likes: {p['likes']}")
    if p.get("dislikes"):
        parts.append(f"Dislikes: {p['dislikes']}")
    if p.get("hobbies"):
        parts.append(f"Hobbies: {p['hobbies']}")
    if p.get("sensory_notes"):
        parts.append(f"Sensory: {p['sensory_notes']}")
    if p.get("emergency_contact"):
        parts.append(f"Emergency contact: {p['emergency_contact']}")
    if p.get("ocd_themes"):
        parts.append(f"OCD themes: {p['ocd_themes']}")
    if p.get("triggers"):
        parts.append(f"Triggers: {p['triggers']}")
    if p.get("deescalation"):
        parts.append(f"De-escalation: {p['deescalation']}")
    if p.get("family_context"):
        parts.append(f"Family/social context: {p['family_context']}")
    if p.get("notes"):
        parts.append(f"Notes: {p['notes']}")
    return "\n".join(parts) if parts else "No structured profile recorded."


init_profile_table()


def list_findings(category=None):
    c = _conn()
    if category:
        rows = c.execute("SELECT * FROM findings WHERE category=? ORDER BY id DESC", (category,)).fetchall()
    else:
        rows = c.execute("SELECT * FROM findings ORDER BY id DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]


init_db()


def save_observation_summary(summary, obs_count=0):
    c = _conn()
    cur = c.cursor()
    cur.execute("INSERT INTO observation_summaries (created_at, summary, obs_count) VALUES (?,?,?)", (_now(), summary, obs_count))
    c.commit()
    rid = cur.lastrowid
    c.close()
    return rid


def get_latest_observation_summary():
    c = _conn()
    row = c.execute("SELECT summary FROM observation_summaries ORDER BY id DESC LIMIT 1").fetchone()
    c.close()
    return (row["summary"] if row else "") or ""


def list_observation_summaries(limit=10):
    c = _conn()
    rows = c.execute("SELECT * FROM observation_summaries ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def save_psychology_summary(summary, doc_count=0):
    c = _conn()
    cur = c.cursor()
    cur.execute("INSERT INTO psychology_summaries (created_at, summary, doc_count) VALUES (?,?,?)", (_now(), summary, doc_count))
    c.commit()
    rid = cur.lastrowid
    c.close()
    return rid


def get_latest_psychology_summary():
    c = _conn()
    row = c.execute("SELECT summary FROM psychology_summaries ORDER BY id DESC LIMIT 1").fetchone()
    c.close()
    return (row["summary"] if row else "") or ""


def list_psychology_summaries(limit=10):
    c = _conn()
    rows = c.execute("SELECT * FROM psychology_summaries ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def save_psychology_page_summary(doc_id, page_num, page_summary):
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO psychology_page_summaries (doc_id, page_num, page_summary, created_at) VALUES (?,?,?,?)",
        (doc_id, str(page_num), page_summary, _now()),
    )
    c.commit()
    rid = cur.lastrowid
    c.close()
    return rid


def get_psychology_page_summaries(doc_id=None):
    c = _conn()
    if doc_id is not None:
        rows = c.execute(
            "SELECT * FROM psychology_page_summaries WHERE doc_id=? ORDER BY page_num", (doc_id,)
        ).fetchall()
    else:
        rows = c.execute("SELECT * FROM psychology_page_summaries ORDER BY doc_id, page_num").fetchall()
    c.close()
    return [dict(r) for r in rows]


def delete_psychology_page_summaries(doc_id=None):
    c = _conn()
    if doc_id is not None:
        c.execute("DELETE FROM psychology_page_summaries WHERE doc_id=?", (doc_id,))
    else:
        c.execute("DELETE FROM psychology_page_summaries")
    c.commit()
    c.close()
