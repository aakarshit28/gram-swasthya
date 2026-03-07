# GramSwasthya Mitra 🏥🤖

GramSwasthya Mitra is an AI-powered, multilingual rural healthcare guidance platform designed to assist patients and ASHA (Accredited Social Health Activist) workers in rural India. It provides instant disease predictions based on symptoms, multi-language accessibility, AI-driven medical context, and information on Ayushman Bharat (PM-JAY) and Jan Aushadhi generic medicines.

## ✨ Key Features

- **🩺 Predictive Symptom Assessment:** Select from over 130 symptoms to receive top disease predictions, confidence percentages, urgency levels, and severity scores using a Machine Learning model.
- **🌐 12-Language Support:** Fully translated UI and on-the-fly medical translation using AWS Bedrock (Claude 3 Haiku), breaking down language barriers across India.
- **🤖 RAG-Powered Health Assistant:** A chat feature grounded in a local knowledge base of verified health guidelines using **Amazon Titan Text Embeddings V2** and **Claude 3 Haiku** to answer follow-up medical questions accurately and safely.
- **👩‍⚕️ ASHA Worker Mode:** A dedicated portal and dashboard for rural health workers to log guest patient assessments, maintain histories, and track community health trends.
- **⚕️ Free Treatment & Medicine Locator:** Automatically checks if a predicted disease is covered under **PM-JAY** (Ayushman Bharat) and suggests affordable generic medicines available at **Jan Aushadhi** stores, complete with a locator map.
- **👵 Accessibility First:** Features an "Elder Mode" (larger fonts/high contrast) and a Text-to-Speech "Read Aloud" function for the visually impaired or illiterate.

## 🛠️ Technology Stack

- **Backend:** Python, Flask, SQLite3
- **Machine Learning:** Scikit-Learn (Joblib), Pandas, Numpy
- **Cloud & AI:** AWS Bedrock (`anthropic.claude-3-haiku-20240307-v1:0` & `amazon.titan-embed-text-v2:0`), Boto3
- **Frontend:** Vanilla HTML5, CSS3, JavaScript, Progressive Web App (PWA) Manifest & Service Workers

## 🚀 Setup & Installation (Local Development)

### Prerequisites
- Python 3.9+ installed
- An active AWS Account with IAM credentials configured
- Amazon Bedrock Model Access granted for **Claude 3 Haiku** and **Titan Text Embeddings V2**

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/gram-swasthya.git
cd gram-swasthya
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure AWS Credentials
To use the AI translation and RAG assistant features, you must configure your AWS credentials.
```bash
aws configure
# Enter your Access Key, Secret Key, and set the default region to us-east-1
```
*Note: Ensure your AWS account has requested and enabled Model Access for Anthropic Claude models via the AWS Bedrock Console Playground.*

### 4. Run the Application
```bash
python app.py
```
The application will be available at `http://127.0.0.1:5000`.

## 📂 Project Structure

- `app.py`: The main Flask backend containing all routing, AWS Bedrock integration, and DB logic.
- `disease_prediction_model.pkl`: The trained Scikit-Learn tree/SVM model for disease prediction.
- `support_data.json`: Static dictionaries mapping diseases to precautions, urgency, and severity.
- `knowledge_base/`: A directory containing `.txt` files of medical guidelines. These are embedded by AWS Titan on startup to power the RAG AI chatbot.
- `templates/`: Contains HTML files (`index.html`, `login.html`, `register.html`, `history.html`, `asha_dashboard.html`, `welcome.html`).
- `static/`: Contains `style.css`, frontend assets, and `sw.js` (Service Worker) for PWA caching.
- `gram_swasthya.db`: (Auto-generated) SQLite database storing user accounts and assessment history.

## ☁️ Deployment to AWS EC2

1. Launch an Ubuntu/Linux EC2 instance.
2. Ensure the instance has an IAM Role attached with `AmazonBedrockFullAccess`.
3. SSH into the server, clone the repository, and install the libraries from `requirements.txt`.
4. Install a WSGI server like `gunicorn`: 
   `pip install gunicorn`
5. Run the app using Gunicorn:
   `gunicorn -w 4 -b 0.0.0.0:80 app:app`
6. (Optional) Set up Nginx as a reverse proxy to manage port 80/443 traffic.

## ⚠️ Disclaimer
GramSwasthya Mitra provides AI-driven preliminary assessments and is **not a substitute for professional medical advice, diagnosis, or treatment.** Always seek the advice of an accredited healthcare provider or visit the nearest Primary Health Centre (PHC) with any questions regarding a medical condition.
