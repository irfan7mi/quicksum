import torch
import whisper
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import pdfplumber
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from googletrans import Translator
from werkzeug.utils import secure_filename
import wikipediaapi
import urllib.parse
import spacy
from summa import summarizer
import mysql.connector
from flask import Flask, request, jsonify
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import fitz 

app = Flask(__name__)

#  Allow credentials & all origins
CORS(app, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173" 
    response.headers["Access-Control-Allow-Credentials"] = "true" 
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root123',
    'database': 'CIP',
    'port': 3307
}

try:
    test_conn = mysql.connector.connect(**DB_CONFIG)
    print("✅ Connected successfully!")
    test_conn.close()
except Exception as e:
    print("❌ Connection failed:", e)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model for audio transcription
whisper_model = whisper.load_model("tiny")

# Create folders for file uploads and summaries
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "summaries"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model_name = "google/long-t5-tglobal-base"
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)


translator = Translator()

print("Models loaded successfully!")


import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score

nltk.download('punkt')

# Function to compute accuracy metrics
def compute_accuracy(reference, hypothesis):
    """Computes ROUGE, BLEU, and BERTScore for accuracy evaluation."""
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)
    
    # BLEU score
    reference_tokens = [word_tokenize(reference.lower())]
    hypothesis_tokens = word_tokenize(hypothesis.lower())
    bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, weights=(0.5, 0.5))
    
    # BERTScore
    P, R, F1 = score([hypothesis], [reference], lang="en", verbose=False)
    
    return {
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'bleu': bleu_score,
        'bertscore': F1.item()
    }


import re
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text, keywords=None):
    """Cleans and preprocesses text, preserving prompted keywords."""
    if not text:
        return ""
    # Normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    # Remove URLs, emails, dates
    text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', text)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
    # Get sentences
    sentences = spacy_tokenizer(text)
    if not sentences:
        return ""
    # Extract keywords if not provided
    extracted_keywords = extract_keywords(text) if not keywords else []
    all_keywords = list(set((keywords or []) + extracted_keywords))
    # Filter sentences: keep if >=2 words, contains keywords, or high TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        tfidf_scores = tfidf_matrix.sum(axis=1).A1
    except ValueError:
        tfidf_scores = [0] * len(sentences)
    filtered_sentences = [
        s for i, s in enumerate(sentences)
        if len(s.split()) >= 2 or
           any(kw.lower() in s.lower() for kw in all_keywords) or
           tfidf_scores[i] > 0.03  # Relaxed threshold
    ]
    return ' '.join(filtered_sentences) if filtered_sentences else ""

def extract_keywords(text):
    """Extracts key phrases using spaCy and TF-IDF."""
    doc = nlp(text)
    # Noun chunks and entities
    keywords = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
    keywords.extend([ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "NORP"]])
    # TF-IDF keywords
    sentences = spacy_tokenizer(text)
    if not sentences:
        return list(set(keywords))[:3]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        top_indices = tfidf_scores.argsort()[-3:][::-1]
        tfidf_keywords = [feature_names[i] for i in top_indices]
        keywords.extend(tfidf_keywords)
    except ValueError:
        pass
    return list(set(keywords))[:3]  # Limit to top 3

def summarize_text(text, summary_level=0.5, keywords=None):
    if not text.strip():
        return "No content to summarize.", {}

    # Preprocess text, passing prompted keywords
    text = preprocess_text(text, keywords)
    if not text.strip():
        return "No content to summarize after preprocessing.", {}

    # Use provided keywords or extract new ones
    if keywords:
        all_keywords = keywords
    else:
        all_keywords = extract_keywords(text)
    keyword_prompt = f"Summarize the text, prioritizing these concepts: {', '.join(all_keywords)}. Text: "
    input_text = keyword_prompt + text

    summary_level = max(0.1, min(1.0, summary_level))

    inputs = tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True).to(device)
    # Dynamic length
    input_length = len(inputs.input_ids[0])
    sentence_count = len(spacy_tokenizer(text))
    max_output_length = int(input_length * summary_level * (1 + 0.02 * sentence_count / 10))
    min_output_length = max(30, int(max_output_length * 0.7))
    # Balanced length penalty
    length_penalty = 1.0

    summary_ids = summarization_model.generate(
        inputs.input_ids,
        max_length=max_output_length,
        min_length=min_output_length,
        length_penalty=length_penalty,
        num_beams=12,  # Increased for precision
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
        top_k=40,
        top_p=0.85,  # Tighter for coherence
        temperature=0.7,  # Control randomness
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Compute accuracy
    accuracy_scores = compute_accuracy(text, summary)
    
    return summary, accuracy_scores
nlp = spacy.load("en_core_web_sm")

def spacy_tokenizer(text):
    """Tokenizes text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def summarize_text_extractive(text, summary_level=0.5):
    """Performs extractive summarization using TextRank."""
    if not text.strip():
        return "No content to summarize."

    summary = summarizer.summarize(text, ratio=summary_level)  
    return summary if summary.strip() else "⚠️ Unable to generate summary!"

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()

                if not page_text:
                    img = page.to_image().original
                    page_text = pytesseract.image_to_string(img)

                text += page_text + "\n"

        return text.strip() if text else None
    except Exception as e:
        return None


def save_summary_to_pdf(summary, output_pdf_path):
    try:
        doc = fitz.open()  # Create a new PDF document
        page = doc.new_page()  # Create a new page
        page.insert_text((50, 50), summary)  # Insert summarized text
        doc.save(output_pdf_path)  # Save the summary as a PDF
        doc.close()
    except Exception as e:
        print(f"Error saving summary to PDF: {e}")
        
@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    text = data.get("text", "")
    summary_level = float(data.get("summary_level", 0.5))
    summaryType = data.get("summaryType", "").lower()

    if not text:
        return jsonify({"error": "No text provided!"}), 400

    if summaryType == "abstractive":
        summary, accuracy_scores = summarize_text(text, summary_level)
    elif summaryType == "extractive":
        summary = summarize_text_extractive(text, summary_level)
        accuracy_scores = compute_accuracy(text, summary)  # Add for extractive
    else:
        return jsonify({"error": "Invalid summary type! Choose 'abstractive' or 'extractive'."}), 400

    return jsonify({"summary": summary, "accuracy": accuracy_scores})

@app.route("/summarize_file", methods=["POST"])
def summarize_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["file"]
    summary_level = float(request.form.get("summary_level", 0.5))
    summary_type = request.form.get("summaryType", "extractive").lower()

    if file.filename == "":
        return jsonify({"error": "No file selected!"}), 400

    filename = secure_filename(file.filename)
    file_ext = filename.split(".")[-1].lower()
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(file_path)

    text = None
    if file_ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_ext == "pdf":
        text = extract_text_from_pdf(file_path) 
    else:
        return jsonify({"error": "Unsupported file type! Only .txt and .pdf allowed."}), 400

    if not text or text.strip() == "":
        return jsonify({"error": "Could not extract text from the file!"}), 400

    if summary_type == "abstractive":
        summary, accuracy_scores = summarize_text(text, summary_level)
    elif summary_type == "extractive":
        summary1 = summarize_text_extractive(text, summary_level)
        summary = summarize_text_extractive(summary1, summary_level)
        accuracy_scores = compute_accuracy(text, summary)  # Add for extractive
    else:
        return jsonify({"error": "Invalid summary type! Choose 'abstractive' or 'extractive'."}), 400

    summary_filename = f"summary_{filename}"
    summary_path = os.path.join(OUTPUT_FOLDER, summary_filename)

    if file_ext == "txt":
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
    elif file_ext == "pdf":
        save_summary_to_pdf(summary, summary_path)  # Use a helper function

    return jsonify({
        "text": text,
        "summary": summary,
        "download_url": f"/download_summary/{summary_filename}",
        "accuracy": accuracy_scores
    })

#  API for downloading summarized file
@app.route("/download_summary/<filename>", methods=["GET"])
def download_summary(filename):
    summary_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(summary_path):
        return send_file(summary_path, as_attachment=True)
    return jsonify({"error": "Summary file not found!"}), 404


#  API for transcribing audio files (`.mp3`)
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(file_path)

    print("\nTranscribing audio... Please wait...")
    result = whisper_model.transcribe(file_path)
    
    transcript = result["text"]
    return jsonify({"transcript": transcript})


#  API for summarizing audio files (`.mp3`)
@app.route("/audio_summarize", methods=["POST"])
def audio_summarize():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(file_path)

    print("\nTranscribing audio... Please wait...")
    result = whisper_model.transcribe(file_path)

    transcript = result["text"]
    summary = summarize_text_extractive(transcript, 0.5)
    summary1 = summarize_text_extractive(summary, 0.5)
    print(summary)
    accuracy_scores = compute_accuracy(transcript, summary1)  # Add for extractive

    return jsonify({"transcript": transcript, "summary": summary1, "accuracy": accuracy_scores})


@app.route("/translate", methods=["OPTIONS", "POST"])
def translate_text():
    if request.method == "OPTIONS":
        return jsonify({"message": "Preflight OK"}), 200

    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        target_lang = data.get("language", "en")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        sentences = text.split(". ")
        translated_sentences = []
        detected_languages = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                detected_lang = translator.detect(sentence).lang
                detected_languages.append({sentence: detected_lang})

                if detected_lang != target_lang:
                    translated_text = translator.translate(sentence, src=detected_lang, dest=target_lang).text
                else:
                    translated_text = sentence
                
                translated_sentences.append(translated_text)

        translated_output = ". ".join(translated_sentences)

        return jsonify({
            "original_text": text,
            "translated_text": translated_output,
            "detected_languages": detected_languages
        })

    except Exception as e:
        return jsonify({"translated_text": "Translation failed", "error": str(e)}), 500
    

@app.route("/save_history", methods=["POST"])
def save_history():
    try:
        data = request.json 
        input_text = data.get("text", "").strip()
        output_summ = data.get("summary", "").strip()
        email = data.get("email", "").strip()

        if not input_text or not output_summ or not email:
            return jsonify({"error": "Missing input, summary, or email!"}), 400

        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        sql = "INSERT INTO Text_summ_history (input_text, output_summ, created_at, email) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (input_text, output_summ, datetime.now(), email))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"message": "Summary saved successfully!"})

    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

@app.route("/get_history", methods=["GET"])
def get_history():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)  
        email = request.args.get("email", "").strip()
        print("Received email:", email) 

        if not email:
            return jsonify({"error": "Email parameter is required!"}), 400

        cursor.execute("SELECT * FROM text_summ_history WHERE email = %s ORDER BY created_at DESC", (email,))
        data = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify(data)

    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    mobile = data.get("mobile")
    password = data.get("password")

    if not (username and email and mobile and password):
        return jsonify({"error": "All fields are required!"}), 400

    hashed_password = generate_password_hash(password)

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, mobile, password) VALUES (%s, %s, %s, %s)",
            (username, email, mobile, hashed_password)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "User registered successfully!"}), 201

    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not (email and password):
        return jsonify({"error": "Email and password required!"}), 400

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user["password"], password):
            return jsonify({"message": "Login successful!"})
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    except mysql.connector.Error as err:
        print("Database error:", err)  # 👈 Add this
        return jsonify({"error": str(err)}), 500


#  Start the Flask server
if __name__ == "__main__":
    app.run(debug=True, port=5000)