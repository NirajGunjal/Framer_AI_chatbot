import os
import random
import string
import re
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from gtts import gTTS
from groq import Groq
from ultralytics import YOLO
import torch
from langdetect import detect

# ✅ Fix for PyTorch >= 2.6 compatibility with YOLO
if hasattr(torch.serialization, "add_safe_globals"):
    import types
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals({
        'ultralytics.nn.tasks.DetectionModel': DetectionModel,
        'str': str,
        'builtins.str': str,
        'types.SimpleNamespace': types.SimpleNamespace,
    })

# ✅ App config
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_AUDIO'] = {'webm', 'wav', 'mp3', 'm4a'}
app.config['ALLOWED_IMAGE'] = {'jpg', 'jpeg', 'png'}

# ✅ Load YOLO models
plant_model = YOLO(r"E:\Farmer_Chatbot-(RajYug Solutions)\plant-identification.pt")
disease_model = YOLO(r"E:\Farmer_Chatbot-(RajYug Solutions)\plant-disease -detector.pt")

# ✅ Groq API
groq_client = Groq(api_key=" ")

# ✅ Utility functions
def is_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AUDIO']

def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_IMAGE']

def clean_text_for_tts(text):
    return re.sub(r'[•●▪️★→*,•‣]', '', text).replace('\n', ' ').strip()

def transcribe_audio_groq(filepath):
    with open(filepath, "rb") as f:
        response = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
        )
        return response.text

def detect_language(text):
    try:
        detected_lang = detect(text)
        return detected_lang if detected_lang in ['hi', 'mr'] else 'en'
    except:
        return 'en'

def get_answer_groq(prompt, lang):
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful agriculture chatbot for Indian farmers."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content, lang

def text_to_audio(text, filename, lang):
    clean_for_tts = clean_text_for_tts(text)
    tts = gTTS(clean_for_tts, lang=lang if lang in ['en', 'hi', 'mr'] else 'en')
    audio_path = os.path.join("static/audio", f"{filename}.mp3")
    tts.save(audio_path)
    return audio_path

def detect_disease(image_path):
    try:
        print("[INFO] Detecting plant type...")
        plant_results = plant_model(image_path)
        print("[INFO] Detecting disease...")
        disease_results = disease_model(image_path)

        plant, disease = "Unknown", "Unknown"

        if plant_results and plant_results[0].boxes and len(plant_results[0].boxes.cls) > 0:
            cls_id = int(plant_results[0].boxes.cls[0])
            plant = plant_model.names.get(cls_id, "Unknown")

        if disease_results and disease_results[0].boxes and len(disease_results[0].boxes.cls) > 0:
            cls_id = int(disease_results[0].boxes.cls[0])
            disease = disease_model.names.get(cls_id, "Unknown")

        print(f"[RESULT] Plant: {plant}, Disease: {disease}")
        return plant, disease
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return "Unknown", "Unknown"

# ✅ Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_lang = request.form.get('lang', 'en')

    # 🎤 Audio Input
    if 'audio' in request.files:
        audio = request.files['audio']
        if audio and is_audio_file(audio.filename):
            filename = secure_filename(audio.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio.save(filepath)

            transcription = transcribe_audio_groq(filepath)
            detected_lang = detect_language(transcription)

            if detected_lang == 'mr':
                prompt = f"प्रश्न: {transcription}\nकृपया मराठीत उत्तर द्या."
            elif detected_lang == 'hi':
                prompt = f"प्रश्न: {transcription}\nकृपया हिंदी में उत्तर दें."
            else:
                prompt = f"Question: {transcription}\nPlease answer in English."

            answer, _ = get_answer_groq(prompt, detected_lang)

            voice_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            text_to_audio(answer, voice_filename, detected_lang)

            return jsonify({
                'text': answer,
                'voice': url_for('static', filename='audio/' + voice_filename + '.mp3'),
                'transcription': transcription
            })

    # ✍️ Text Input
    elif 'text' in request.form:
        question = request.form['text']
        detected_lang = detect_language(question)

        if detected_lang == 'mr':
            prompt = f"प्रश्न: {question}\nकृपया मराठीत उत्तर द्या."
        elif detected_lang == 'hi':
            prompt = f"प्रश्न: {question}\nकृपया हिंदी में उत्तर दें."
        else:
            prompt = f"Question: {question}\nPlease answer in English."

        answer, _ = get_answer_groq(prompt, detected_lang)

        voice_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        text_to_audio(answer, voice_filename, detected_lang)

        return jsonify({
            'text': answer,
            'voice': url_for('static', filename='audio/' + voice_filename + '.mp3'),
            'transcription': None
        })

    # 🖼️ Image Input
    elif 'image' in request.files:
        image = request.files['image']
        if image and is_image_file(image.filename):
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)

            plant, disease = detect_disease(filepath)

            if plant == "Unknown" or disease == "Unknown":
                return jsonify({'text': "Sorry, I couldn't process your image. Please try again."}), 200

            prompt = f"The detected plant is '{plant}' and it has the disease '{disease}'. Suggest treatment."
            if user_lang == 'mr':
                prompt += " कृपया मराठीत उत्तर द्या."
            elif user_lang == 'hi':
                prompt += " कृपया हिंदी में उत्तर दें."
            else:
                prompt += " Please answer in English."

            answer, _ = get_answer_groq(prompt, user_lang)

            voice_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            text_to_audio(answer, voice_filename, user_lang)

            return jsonify({
                'text': f"Plant: {plant}\nDisease: {disease}\nCure: {answer}",
                'voice': url_for('static', filename='audio/' + voice_filename + '.mp3'),
                'transcription': None
            })

    return jsonify({'text': 'No valid input found'}), 400

# ✅ Ensure Folders Exist
if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static/audio", exist_ok=True)
    app.run(debug=True, use_reloader=False)
