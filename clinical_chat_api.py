import os
import uuid
import json
import base64
# FastAPI imports removed - endpoints now in main.py
from fastapi import Form
from fastapi.responses import JSONResponse
from gtts import gTTS
from deep_translator import GoogleTranslator
from langdetect import detect
import openai

# FastAPI app and CORS removed - now handled in main.py

AUDIO_FOLDER = "audios"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Load API keys from config.json file
config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
try:
    with open(config_path) as config_file:
        config = json.load(config_file)
        OPENAI_API_KEY = config.get("OPENAI_API_KEY")
except FileNotFoundError:
    OPENAI_API_KEY = None
    print("Warning: config.json not found")

# Conversation history
conversation_history = []

# Translation function


def translate_text(text, output_lang):
    if not text.strip():
        return ""
    try:
        input_lang = detect(text)
    except:
        input_lang = "en"  # Default to English if detection fails
    return GoogleTranslator(source=input_lang, target=output_lang).translate(text)

# Text-to-speech function that returns base64 audio data


def text_to_speech_base64(text: str) -> str:
    if not text.strip():
        raise ValueError("No text to convert to audio")

    filename = f"temp_audio_{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)

    tts = gTTS(text)
    tts.save(filepath)

    with open(filepath, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

    os.remove(filepath)  # Clean up temp file
    return audio_base64

# Simulated clinical chat


def clinical_chat(prompt: str) -> str:
    openai.api_key = OPENAI_API_KEY  # Use the API key from config.json
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful clinical assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# Clinical chat function (used by main.py)
# Note: Endpoint moved to main.py - this file now only contains utility functions


# API info function (used by main.py)
def get_api_info():
    return {
        "message": "Clinical Chat API",
        "description": "AI-powered medical assistant with multi-language support",
        "endpoints": {
            "clinical_chat": "POST /clinical_chat",
            "api_info": "POST /api_info",
            "documentation": "/docs"
        },
        "usage": {
            "endpoint": "/clinical_chat",
            "method": "POST",
            "parameters": {
                "query": "Your medical question",
                "output_language": "Language code (e.g., 'en', 'es', 'fr')"
            }
        },
        "response_format": {
            "audio_base64": "Base64 encoded MP3 audio data",
            "audio_format": "mp3"
        }
    }
