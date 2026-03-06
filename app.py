from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import joblib
import numpy as np
import os, json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'gram_swasthya_secret_key_2024'  # Required for session management

# Database configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gram_swasthya.db")

# SQLite Database Setup
import sqlite3

def init_db():
    """Initialize database with tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            address TEXT,
            phone TEXT UNIQUE,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add password_hash column if it doesn't exist (for existing databases)
    try:
        c.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
        conn.commit()
    except Exception:
        pass  # Column already exists
    
    # Add role column if it doesn't exist
    try:
        c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'patient'")
        conn.commit()
    except Exception:
        pass  # Column already exists

    # Assessments history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            symptoms TEXT,
            disease TEXT,
            confidence INTEGER,
            urgency TEXT,
            severity_score INTEGER,
            patient_name TEXT,
            assessed_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Add patient_name and assessed_by columns if they don't exist
    for col_def in ["patient_name TEXT", "assessed_by INTEGER"]:
        try:
            c.execute(f'ALTER TABLE assessments ADD COLUMN {col_def}')
            conn.commit()
        except Exception:
            pass

    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Load model
model        = joblib.load(os.path.join(BASE_DIR, "disease_prediction_model.pkl"))
symptom_list = list(model.feature_names_in_)

# Load support data from JSON only
with open(os.path.join(BASE_DIR, "support_data.json"), "r") as f:
    _s = json.load(f)

DISEASE_URGENCY = _s["disease_urgency"]
DISEASE_RISK    = _s["disease_risk"]
SEV_DICT        = _s["sev_dict"]
DESC_DICT       = _s["desc_dict"]
PREC_DICT       = _s["prec_dict"]

# Language code mapping for deep-translator
LANG_CODE_MAP = {
    "en": "en",
    "hi": "hi",
    "bn": "bn",
    "mr": "mr",
    "gu": "gu",
    "pa": "pa",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "ml": "ml",
    "or": "or",
    "ur": "ur",
}

def translate_text(text, target_lang):
    """Translate text to target language using deep-translator."""
    if not text or target_lang == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        lang_code = LANG_CODE_MAP.get(target_lang, "en")
        if lang_code == "en":
            return text
        translator = GoogleTranslator(source="en", target=lang_code)
        result = translator.translate(text)
        return result if result else text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def get_urgency(disease, symptoms):
    d = disease.strip().lower()
    if d in DISEASE_URGENCY:
        return DISEASE_URGENCY[d]
    score = sum(SEV_DICT.get(s.strip().lower(), 0) for s in symptoms)
    if score >= 20: return "High"
    if score >= 10: return "Medium"
    return "Low"

def get_risk(disease, urgency):
    d = disease.strip().lower()
    if d in DISEASE_RISK:
        return DISEASE_RISK[d]
    return {"High": "This condition may be serious. Please consult a doctor immediately.",
            "Medium": "Please consult a doctor within the next 1-2 days.",
            "Low": "Monitor your symptoms. See a doctor if they worsen."}.get(urgency, "Consult a healthcare professional.")

def get_description(disease):
    return DESC_DICT.get(disease.strip().lower(), "Consult a healthcare professional for a detailed evaluation.")

def get_precautions(disease):
    p = PREC_DICT.get(disease.strip().lower(), [])
    return p if p else ["Rest and stay hydrated", "Avoid self-medication", "Monitor your symptoms closely", "Consult a doctor if symptoms worsen"]

def get_top3(input_df):
    probs = model.predict_proba(input_df)[0]
    top3idx = np.argsort(probs)[::-1][:3]
    return [{"disease": model.classes_[i], "confidence": round(float(probs[i]) * 100)} for i in top3idx if probs[i] > 0.01]

@app.route("/")
def index():
    welcome = os.path.join(BASE_DIR, "templates", "welcome.html")
    if os.path.exists(welcome):
        return render_template("welcome.html")
    return render_template("index.html", symptoms=symptom_list)

@app.route("/home")
def home():
    return render_template("index.html", symptoms=symptom_list)

@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json()
    selected = data.get("symptoms", [])
    if not selected:
        return jsonify({"error": "No symptoms provided"}), 400

    try:
        input_df = pd.DataFrame([[1 if col in selected else 0 for col in symptom_list]], columns=symptom_list)

        top3       = get_top3(input_df)
        disease    = top3[0]["disease"]
        confidence = top3[0]["confidence"]
        urgency    = get_urgency(disease, selected)

        sev_score = sum(SEV_DICT.get(s.strip().lower(), 0) for s in selected)
        max_score = len(selected) * 7
        sev_pct   = round((sev_score / max_score * 100) if max_score else 0)

        return jsonify({
            "disease":        disease,
            "confidence":     confidence,
            "urgency":        urgency,
            "risk":           get_risk(disease, urgency),
            "description":    get_description(disease),
            "precautions":    get_precautions(disease),
            "severity_score": sev_score,
            "severity_pct":   sev_pct,
            "top3":           top3,
            "symptom_count":  len(selected),
        })
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction: " + str(e)}), 500

# Cache for symptom translations (language -> {eng_symptom: translated})
_symptom_cache = {}

@app.route("/translate_symptoms", methods=["POST"])
def translate_symptoms():
    """
    Translate symptom list to target language.
    Request: { lang }
    Returns: { translations: { "symptom_name": "translated" } }
    """
    data = request.get_json()
    lang = data.get("lang", "en")

    if lang == "en":
        return jsonify({"translations": {s: s.replace("_", " ") for s in symptom_list}})

    # Return cached if available
    if lang in _symptom_cache:
        return jsonify({"translations": _symptom_cache[lang]})

    try:
        from deep_translator import GoogleTranslator
        lang_code = LANG_CODE_MAP.get(lang, "en")
        translator = GoogleTranslator(source="en", target=lang_code)

        translations = {}
        # Translate in batches of 10 to avoid rate limits
        batch_size = 10
        syms = symptom_list[:]
        eng_texts = [s.replace("_", " ") for s in syms]

        for i in range(0, len(eng_texts), batch_size):
            batch = eng_texts[i:i+batch_size]
            try:
                for j, txt in enumerate(batch):
                    translated = translator.translate(txt)
                    translations[syms[i+j]] = translated if translated else txt
            except Exception:
                for j, txt in enumerate(batch):
                    translations[syms[i+j]] = txt

        _symptom_cache[lang] = translations
        return jsonify({"translations": translations})
    except Exception as e:
        print(f"Symptom translation error: {e}")
        return jsonify({"translations": {s: s.replace("_", " ") for s in symptom_list}}), 200


def translate_batch():
    """
    Batch-translate medical content into a target language.
    Request: { lang, disease, description, risk, precautions: [...], top3: [...] }
    Returns the same structure with all strings translated.
    """
    data = request.get_json()
    lang = data.get("lang", "en")

    if lang == "en":
        return jsonify(data)

    try:
        from deep_translator import GoogleTranslator
        lang_code = LANG_CODE_MAP.get(lang, "en")
        translator = GoogleTranslator(source="en", target=lang_code)

        def tx(text):
            if not text:
                return text
            try:
                r = translator.translate(str(text))
                return r if r else text
            except:
                return text

        translated = {
            "disease":     tx(data.get("disease", "")),
            "description": tx(data.get("description", "")),
            "risk":        tx(data.get("risk", "")),
            "precautions": [tx(p) for p in data.get("precautions", [])],
            "top3":        [{"disease": tx(item["disease"]), "confidence": item["confidence"]}
                            for item in data.get("top3", [])],
        }
        return jsonify(translated)
    except Exception as e:
        print(f"Batch translation error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# NEW ROUTES FOR USER MANAGEMENT & DATABASE
# ============================================================

@app.route("/register_page", methods=["GET"])
def register_page():
    return render_template("register.html")

@app.route("/register", methods=["POST"])
def register_user():
    data = request.get_json()
    name = data.get("name", "").strip()
    age = data.get("age")
    gender = data.get("gender", "")
    address = data.get("address", "")
    phone = data.get("phone", "").strip()
    password = data.get("password", "").strip()
    role = data.get("role", "patient")  # 'patient' or 'asha'
    
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400
    if not phone:
        return jsonify({"success": False, "error": "Phone number is required"}), 400
    if not password:
        return jsonify({"success": False, "error": "Password is required"}), 400
    if len(password) < 4:
        return jsonify({"success": False, "error": "Password must be at least 4 characters"}), 400
    
    password_hash = generate_password_hash(password)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE phone = ?", (phone,))
        existing = c.fetchone()
        if existing:
            conn.close()
            return jsonify({"success": False, "error": "Phone number already registered. Please login instead."}), 400
        
        c.execute(
            "INSERT INTO users (name, age, gender, address, phone, password_hash, role) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, age, gender, address, phone, password_hash, role)
        )
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        
        session["user_id"] = user_id
        session["user_name"] = name
        session["user_age"] = age
        session["user_gender"] = gender
        session["user_address"] = address
        session["user_role"] = role
        
        redirect_url = "/asha_dashboard" if role == "asha" else "/home"
        return jsonify({"success": True, "user_id": user_id, "message": "Registered successfully", "redirect": redirect_url})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/login_page", methods=["GET"])
def login_page():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_user():
    data = request.get_json()
    phone = data.get("phone", "").strip()
    password = data.get("password", "").strip()
    
    if not phone or not password:
        return jsonify({"success": False, "error": "Phone and password are required"}), 400
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE phone = ?", (phone,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({"success": False, "error": "No account found with this phone number"}), 401
        if not user["password_hash"]:
            return jsonify({"success": False, "error": "Account has no password. Please re-register."}), 401
        if not check_password_hash(user["password_hash"], password):
            return jsonify({"success": False, "error": "Incorrect password. Please try again."}), 401
        
        role = user["role"] if user["role"] else "patient"
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        session["user_age"] = user["age"]
        session["user_gender"] = user["gender"]
        session["user_address"] = user["address"]
        session["user_role"] = role
        
        redirect_url = "/asha_dashboard" if role == "asha" else "/home"
        return jsonify({"success": True, "user_id": user["id"], "name": user["name"], "role": role, "redirect": redirect_url})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/save_assessment", methods=["POST"])
def save_assessment():
    data = request.get_json()
    user_id = session.get("user_id")
    name = session.get("user_name", "Guest")
    role = session.get("user_role", "patient")
    
    symptoms = data.get("symptoms", [])
    disease = data.get("disease", "")
    confidence = data.get("confidence", 0)
    urgency = data.get("urgency", "Low")
    severity_score = data.get("severity_score", 0)
    patient_name = data.get("patient_name", "")  # ASHA worker flow
    
    # For ASHA workers, assessed_by = their user_id; user_id in record = None (guest patient)
    assessed_by = user_id if role == "asha" else None
    record_user_id = None if role == "asha" else user_id
    record_name = patient_name if (role == "asha" and patient_name) else name
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO assessments (user_id, name, symptoms, disease, confidence, urgency, severity_score, patient_name, assessed_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (record_user_id, record_name, ",".join(symptoms), disease, confidence, urgency, severity_score, patient_name, assessed_by)
        )
        assessment_id = c.lastrowid
        conn.commit()
        conn.close()
        return jsonify({"success": True, "assessment_id": assessment_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/get_user", methods=["GET"])
def get_user():
    user_id = session.get("user_id")
    if user_id:
        return jsonify({
            "user_id": user_id,
            "name": session.get("user_name", ""),
            "age": session.get("user_age"),
            "gender": session.get("user_gender", ""),
            "address": session.get("user_address", ""),
            "role": session.get("user_role", "patient")
        })
    return jsonify({"user_id": None})


@app.route("/history_page")
def history_page():
    """Serve the history page"""
    return render_template("history.html")

@app.route("/history", methods=["GET"])
def get_history():
    """
    Get assessment history for current user.
    Returns: { assessments: [...] }
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"assessments": []})
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT * FROM assessments WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
            (user_id,)
        )
        rows = c.fetchall()
        conn.close()
        
        assessments = [dict(row) for row in rows]
        return jsonify({"assessments": assessments})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logout", methods=["POST", "GET"])
def logout():
    """Clear user session and redirect to welcome page"""
    session.clear()
    return redirect(url_for("index"))


# ============================================================
# PMJAY + GENERIC MEDICINES DATA
# ============================================================

PMJAY_COVERED = {
    "malaria": True, "dengue": True, "typhoid": True, "tuberculosis": True,
    "pneumonia": True, "hepatitis b": True, "hepatitis c": True,
    "jaundice": True, "urinary tract infection": True, "chickenpox": True,
    "heart attack": True, "hypertension": True, "diabetes": True,
    "chronic kidney disease": True, "arthritis": True, "asthma": True,
    "anemia": True, "gastroenteritis": True, "bronchial asthma": True,
    "peptic ulcer disease": True, "cervical spondylosis": True,
    "paralysis (brain hemorrhage)": True, "hyperthyroidism": True,
    "hypothyroidism": True, "osteoarthritis": True, "allergy": False,
    "common cold": False, "acne": False, "drug reaction": False,
    "migraine": False, "psoriasis": False, "fungal infection": False,
}

GENERIC_MEDICINES = {
    "malaria": "Chloroquine / Artemether (free at govt. PHC under NVBDCP)",
    "dengue": "Paracetamol 500mg for fever — available at Jan Aushadhi stores",
    "typhoid": "Ciprofloxacin 500mg — available at Jan Aushadhi ₹5–10/tab",
    "tuberculosis": "DOTS therapy — completely FREE at all govt. health centres",
    "pneumonia": "Amoxicillin 500mg — available at Jan Aushadhi ₹2–5/tab",
    "urinary tract infection": "Nitrofurantoin 100mg — available at Jan Aushadhi stores",
    "diabetes": "Metformin 500mg — available at Jan Aushadhi ₹1–3/tab",
    "hypertension": "Amlodipine 5mg — available at Jan Aushadhi ₹1–2/tab",
    "anemia": "Ferrous Sulphate + Folic Acid — free under NHM scheme",
    "asthma": "Salbutamol inhaler — available at Jan Aushadhi stores",
    "hepatitis b": "Tenofovir 300mg — available at govt. hospitals free",
    "gastroenteritis": "ORS sachets + Zinc — free at all Anganwadi / PHC centres",
    "chickenpox": "Acyclovir 400mg — available at Jan Aushadhi stores",
    "arthritis": "Ibuprofen 400mg / Diclofenac — available at Jan Aushadhi ₹2/tab",
    "migraine": "Sumatriptan 50mg — available at Jan Aushadhi stores",
    "fungal infection": "Clotrimazole cream — available at Jan Aushadhi ₹15–20",
    "allergy": "Cetirizine 10mg — available at Jan Aushadhi ₹1/tab",
    "common cold": "Paracetamol + Cetirizine — available at Jan Aushadhi ₹5 for 10 tabs",
}

@app.route("/disease_info", methods=["POST"])
def disease_info():
    """
    Returns PM-JAY coverage and generic medicine info for a disease.
    Request: { disease }
    Returns: { pmjay_covered, generic_medicine, jan_aushadhi_tip }
    """
    data = request.get_json()
    disease = data.get("disease", "").strip().lower()
    
    pmjay = PMJAY_COVERED.get(disease, None)
    generic = GENERIC_MEDICINES.get(disease, "Visit your nearest Jan Aushadhi store for affordable generic medicines.")
    
    return jsonify({
        "pmjay_covered": pmjay,
        "generic_medicine": generic,
        "jan_aushadhi_url": "https://janaushadhi.gov.in/StoreLocater.aspx",
        "ayushman_url": "https://pmjay.gov.in"
    })


# ============================================================
# ASHA WORKER ROUTES
# ============================================================

@app.route("/asha_dashboard")
def asha_dashboard():
    """Serve ASHA worker dashboard — protected"""
    if session.get("user_role") != "asha":
        return redirect(url_for("index"))
    return render_template("asha_dashboard.html")


@app.route("/asha_patients", methods=["GET"])
def asha_patients():
    """
    Get all patients assessed by the logged-in ASHA worker.
    Returns: { assessments: [...], stats: {...} }
    """
    user_id = session.get("user_id")
    if not user_id or session.get("user_role") != "asha":
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT * FROM assessments WHERE assessed_by = ? ORDER BY created_at DESC LIMIT 50",
            (user_id,)
        )
        rows = c.fetchall()
        conn.close()
        
        assessments = [dict(row) for row in rows]
        
        # Compute stats
        total = len(assessments)
        high_urgency = sum(1 for a in assessments if a.get("urgency") == "High")
        today = datetime.now().date().isoformat()
        today_count = sum(1 for a in assessments if a.get("created_at", "")[:10] == today)
        
        # Most common disease
        from collections import Counter
        diseases = [a["disease"] for a in assessments if a.get("disease")]
        most_common = Counter(diseases).most_common(1)
        top_disease = most_common[0][0] if most_common else "—"
        
        return jsonify({
            "assessments": assessments,
            "stats": {
                "total": total,
                "high_urgency": high_urgency,
                "today_count": today_count,
                "top_disease": top_disease
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
