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

TRANSLATION_CACHE_FILE = os.path.join(BASE_DIR, "translation_cache.json")
translation_cache = {}

if os.path.exists(TRANSLATION_CACHE_FILE):
    try:
        with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
            translation_cache = json.load(f)
    except Exception:
        translation_cache = {}

import threading
cache_lock = threading.Lock()

def save_translation_cache():
    with cache_lock:
        try:
            with open(TRANSLATION_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(translation_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

_bedrock_client = None

def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        _bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    return _bedrock_client

# ============================================================
# RAG SYSTEM INITIALIZATION (Knowledge Base + Titan Embeddings)
# ============================================================
KNOWLEDGE_BASE_DOCS = []
KNOWLEDGE_BASE_EMBEDDINGS = []

def get_embedding(text):
    """Get embedding from Amazon Titan Text V2 via Bedrock"""
    try:
        bedrock = get_bedrock_client()
        body = json.dumps({
            "inputText": text,
            "dimensions": 512,
            "normalize": True
        })
        response = bedrock.invoke_model(
            body=body,
            modelId="amazon.titan-embed-text-v2:0",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get('body').read())
        return response_body.get('embedding')
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def init_rag_system():
    global KNOWLEDGE_BASE_DOCS, KNOWLEDGE_BASE_EMBEDDINGS
    print("Initializing RAG Knowledge Base...")
    kb_dir = os.path.join(BASE_DIR, "knowledge_base")
    
    if not os.path.exists(kb_dir):
        print("Knowledge base directory not found. RAG will fallback to general LLM knowledge.")
        return
        
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(kb_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        # Split into smaller chunks (simple paragraph split)
                        chunks = content.split('\n\n')
                        for chunk in chunks:
                            chunk = chunk.strip()
                            if len(chunk) > 20:
                                import time
                                time.sleep(0.3)
                                emb = get_embedding(chunk)
                                if emb:
                                    KNOWLEDGE_BASE_DOCS.append(chunk)
                                    KNOWLEDGE_BASE_EMBEDDINGS.append(emb)
            except Exception as e:
                print(f"Error reading KB file {filename}: {e}")
                
    print(f"RAG Initialized: {len(KNOWLEDGE_BASE_DOCS)} document chunks indexed.")

# Try to initialize RAG in the background or immediately
try:
    import threading
    threading.Thread(target=init_rag_system, daemon=True).start()
except Exception as e:
    print(f"Could not start RAG thread: {e}")

def translate_text(text, target_lang):
    """Translate text to target language using deep-translator and local cache."""
    if not text or target_lang == "en":
        return text
        
    text_str = str(text).strip()
    if not text_str:
        return text

    if target_lang not in translation_cache:
        translation_cache[target_lang] = {}

    if text_str in translation_cache[target_lang]:
        return translation_cache[target_lang][text_str]

    try:
        import json
        
        # Initialize Bedrock client
        bedrock_runtime = get_bedrock_client()
        
        lang_name = target_lang # We pass the language name or code
        
        # We use Claude 3 Haiku as it is fast, cheap, and very smart for translation
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
        prompt = f"You are an expert medical translator. Translate the following English medical text into the {lang_name} language. Return ONLY the translated text, nothing else. Text to translate: '{text_str}'"
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1
        })

        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        translated_text = response_body.get('content')[0].get('text').strip()
        
        if translated_text:
            translation_cache[target_lang][text_str] = translated_text
            save_translation_cache()
            return translated_text
            
        return text
    except Exception as e:
        print(f"Bedrock Translation error: {e}")
        # Fallback to returning original text if AWS fails or isn't configured yet
        return text

@app.context_processor
def inject_translations():
    lang = request.cookies.get("gsm_lang", "en")
    def _t(text):
        return translate_text(text, lang)
    return dict(_t=_t, current_lang=lang)

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
        if not top3:
            return jsonify({"error": "Symptom match too low. Please consult a doctor."}), 400
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
        import json
        
        lang_name = lang # We pass the language name or code
        bedrock_runtime = get_bedrock_client()
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"

        syms = symptom_list[:]
        eng_texts = [s.replace("_", " ") for s in syms]
        
        prompt = f"You are a medical translator. Translate this JSON array of English medical symptoms into the {lang} language. Return ONLY a valid JSON array of strings in the exact same order as the input. Array to translate: {json.dumps(eng_texts)}"

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1
        })

        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        translated_array_str = response_body.get('content')[0].get('text').strip()
        
        # Parse the JSON array returned by Bedrock
        try:
            translated_array = json.loads(translated_array_str)
        except json.JSONDecodeError:
            print("Failed to parse Bedrock JSON response array")
            return jsonify({"translations": {s: s.replace("_", " ") for s in symptom_list}}), 200

        translations = {}
        for i, sym in enumerate(syms):
            if i < len(translated_array):
                translations[sym] = translated_array[i]
            else:
                translations[sym] = eng_texts[i]

        _symptom_cache[lang] = translations
        return jsonify({"translations": translations})
    except Exception as e:
        print(f"Bedrock Symptom translation error: {e}")
        return jsonify({"translations": {s: s.replace("_", " ") for s in symptom_list}}), 200


@app.route("/translate_batch", methods=["POST"])
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
        def tx(text):
            return translate_text(text, lang)

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
# AWS RAG: HEALTH ASSISTANT ROUTE (Titan + Claude 3)
# ============================================================

@app.route("/ask_assistant", methods=["POST"])
def ask_assistant():
    """RAG-powered health assistant endpoint."""
    data = request.get_json()
    query = data.get("query", "").strip()
    lang = data.get("lang", "en")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
        
    try:
        import numpy as np
        
        # 1. RETRIEVE: Embed the user's query
        query_embedding = get_embedding(query)
        context_text = ""
        
        if query_embedding and KNOWLEDGE_BASE_EMBEDDINGS:
            # Calculate cosine similarity using numpy
            q_vec = np.array(query_embedding)
            similarities = []
            
            for doc_emb in KNOWLEDGE_BASE_EMBEDDINGS:
                d_vec = np.array(doc_emb)
                sim = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
                similarities.append(sim)
                
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            # If the best match is reasonably relevant, use it as context
            if best_score > 0.4:
                context_text = KNOWLEDGE_BASE_DOCS[best_idx]
                
        # 2. GENERATE: Prompt Claude 3 with the context
        bedrock_runtime = get_bedrock_client()
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
        system_prompt = f"You are a helpful, empathetic health assistant for rural India. Answer the user's question safely and accurately in the {lang} language."
        if context_text:
            system_prompt += f" Use this verified medical context to ground your answer: '{context_text}'."
        else:
            system_prompt += " Since no specific context was found, provide general safe advice and recommend consulting a local doctor or PHC."
            
        system_prompt += " Keep the answer concise (under 4 sentences). Do not prescribe specific new medications unless it is in the context (like Paracetamol for fever). Always add a disclaimer to consult a doctor."

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": f"System Context: {system_prompt} \n\n User Question: {query}"
                }
            ],
            "temperature": 0.2
        })

        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=model_id,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        assistant_reply = response_body.get('content')[0].get('text').strip()
        
        return jsonify({"response": assistant_reply, "context_used": bool(context_text)})
        
    except Exception as e:
        import traceback
        debug_path = os.path.join(BASE_DIR, "rag_debug.txt")
        try:
            with open(debug_path, "w") as f:
                f.write(traceback.format_exc())
        except:
            pass
        print(f"RAG Assistant error: {e}")
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500


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
