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

## ⚠️ Disclaimer
GramSwasthya Mitra provides AI-driven preliminary assessments and is **not a substitute for professional medical advice, diagnosis, or treatment.** Always seek the advice of an accredited healthcare provider or visit the nearest Primary Health Centre (PHC) with any questions regarding a medical condition.
