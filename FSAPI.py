import os
import random
import string
import types
import base64
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from groq import Groq
from ultralytics import YOLO
import torch
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables")

# PyTorch patch for newer versions
if hasattr(torch.serialization, "add_safe_globals"):
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals({
        'ultralytics.nn.tasks.DetectionModel': DetectionModel,
        'str': str,
        'builtins.str': str,
        'types.SimpleNamespace': types.SimpleNamespace,
    })

# FastAPI Initialization
app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory Setup
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'static/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Model loading
plant_model = YOLO("E:/Farmer_Chatbot-(RajYug Solutions)/plant-identification.pt")
disease_model = YOLO("E:/Farmer_Chatbot-(RajYug Solutions)/plant-disease -detector.pt")

# Groq Client
groq_client = Groq(api_key=" ")

# Request Schema
class ChatRequest(BaseModel):
    text: str | None = None
    lang: str = "en"
    audio_base64: str | None = None
    image_base64: str | None = None

# Transcription
def transcribe_audio_groq(filepath):
    try:
        with open(filepath, "rb") as f:
            response = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
            )
        return response.text
    except Exception as e:
        print(f"Audio transcription error: {e}")
        return None

# Chat Response
def get_answer_groq(question, lang):
    system_prompt = {
        'mr': "तू भारतीय शेतकऱ्यांसाठी उपयुक्त शेतीविषयक chatbot आहेस. मराठीत उत्तर दे.",
        'hi': "आप एक भारतीय किसानों के लिए सहायक कृषि चैटबॉट हैं। कृपया हिंदी में उत्तर दें।",
        'en': "You are a helpful agriculture chatbot for Indian farmers. Reply in English."
    }.get(lang, "You are a helpful agriculture chatbot for Indian farmers. Reply in English.")

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
        )
        return response.choices[0].message.content, lang
    except Exception as e:
        print(f"Groq response error: {e}")
        return "❌ Sorry, failed to get response from Groq.", lang

# Text-to-Speech
def text_to_audio(text, filename, lang):
    try:
        if lang not in ['en', 'hi', 'mr']:
            lang = 'en'
        tts = gTTS(text, lang=lang)
        audio_path = os.path.join(AUDIO_FOLDER, f"{filename}.mp3")
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# Image Prediction
def detect_disease(image_path):
    plant_results = plant_model(image_path)
    disease_results = disease_model(image_path)

    plant = "Unknown plant"
    disease = "Unknown disease"

    if plant_results and plant_results[0].boxes:
        cls_id = int(plant_results[0].boxes.cls[0])
        plant = plant_model.names[cls_id]

    if disease_results and disease_results[0].boxes:
        cls_id = int(disease_results[0].boxes.cls[0])
        disease = disease_model.names[cls_id]

    return plant, disease

# Routes
@app.get("/")
def root():
    return {"message": "✅ FastAPI Farmer Chatbot is running. Use POST /chat."}

@app.post("/chat")
async def chat(req: ChatRequest):
    text = req.text
    lang = req.lang
    audio_b64 = req.audio_base64
    image_b64 = req.image_base64

    try:
        # AUDIO INPUT
        if audio_b64:
            filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) + ".mp3"
            audio_path = os.path.join(UPLOAD_FOLDER, filename)

            try:
                audio_data = base64.b64decode(audio_b64.strip())
                with open(audio_path, "wb") as f:
                    f.write(audio_data)
            except Exception as e:
                print(f"Audio decoding error: {e}")
                return JSONResponse({"error": "Invalid audio base64"}, status_code=400)

            transcription = transcribe_audio_groq(audio_path)
            if not transcription:
                return JSONResponse({"error": "Audio processing failed"}, status_code=500)

            answer, lang = get_answer_groq(transcription, lang)
            voice_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            audio_file = text_to_audio(answer, voice_filename, lang)

            return JSONResponse({
                "text": f"\U0001F3A4 Transcribed: {transcription}\n\n🤖 उत्तर: {answer}",
                "voice": f"/static/audio/{voice_filename}.mp3" if audio_file else None
            })

        # TEXT INPUT
        elif text:
            answer, lang = get_answer_groq(text, lang)
            voice_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            audio_file = text_to_audio(answer, voice_filename, lang)
            return JSONResponse({
                "text": answer,
                "voice": f"/static/audio/{voice_filename}.mp3" if audio_file else None
            })

        # IMAGE INPUT
        elif image_b64:
            image_path = os.path.join(UPLOAD_FOLDER, "input_image.jpg")
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image_b64))

            plant, disease = detect_disease(image_path)
            prompt = f"The plant is {plant} and it has the disease {disease}. Suggest treatment."
            answer, lang = get_answer_groq(prompt, lang)
            voice_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            audio_file = text_to_audio(answer, voice_filename, lang)

            return JSONResponse({
                "text": f"🪴 Plant: {plant}\n🪠 Disease: {disease}\n\n💊 Cure: {answer}",
                "voice": f"/static/audio/{voice_filename}.mp3" if audio_file else None
            })

        return JSONResponse({"text": "❌ No valid input received."}, status_code=400)

    except Exception as e:
        print(f"Unexpected error: {e}")
        return JSONResponse({"error": "Something went wrong."}, status_code=500)

@app.get("/static/audio/{filename}")
async def get_audio_file(filename: str):
    return FileResponse(path=os.path.join(AUDIO_FOLDER, filename))
