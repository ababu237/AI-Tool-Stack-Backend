import os
import uuid
import json
import base64
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from deep_translator import GoogleTranslator
from langdetect import detect

app = FastAPI(title="Text Translation API",
              description="Multi-language text translation with TTS")

# Add root endpoint


# Endpoint moved to main.py - @app.post("/api_info")
async def get_api_info():
    return {
        "message": "Text Translation API",
        "description": "Text translation between multiple languages with text-to-speech",
        "endpoints": {
            "translate_text": "POST /translate_text",
            "api_info": "POST /api_info",
            "documentation": "/docs"
        },
        "usage": {
            "endpoint": "/translate_text",
            "method": "POST",
            "parameters": {
                "text": "Text to translate",
                "output_language": "Target language code (e.g., 'es', 'fr', 'de')"
            }
        },
        "response_format": {
            "audio_base64": "Base64 encoded MP3 audio data",
            "audio_format": "mp3"
        }
    }

# Add CORS middleware RIGHT AFTER creating the app
app.add_middleware(
    CORSMiddleware,
    # Specify your frontend origin
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

AUDIO_FOLDER = "temp_audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Load API keys from config.json file
config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
try:
    with open(config_path) as config_file:
        config = json.load(config_file)
        TRANSLATION_API_KEY = config.get("TRANSLATION_API_KEY")
except FileNotFoundError:
    TRANSLATION_API_KEY = None
    print("Warning: config.json not found")

# Function to translate text


def translate_text(text: str, output_lang: str) -> str:
    if not text.strip():
        return ""
    # Auto-detect the input language
    try:
        input_lang = detect(text)
    except:
        input_lang = "en"  # Default to English if detection fails

    # Check if the input language is the same as the output language
    if input_lang == output_lang:
        return text  # Return the original text if languages are the same

    # Here you would use the TRANSLATION_API_KEY if needed
    return GoogleTranslator(source=input_lang, target=output_lang).translate(text)

# Endpoint: Translate text and return audio


# Endpoint moved to main.py - @app.post("/translate_text")
async def api_translate_text(
    text: str = Form(...),
    output_language: str = Form(...)
):
    try:
        # Step 1: Translate the text
        translated = translate_text(text, output_language)

        # Step 2: Generate speech as base64
        filename = f"temp_audio_{uuid.uuid4()}.mp3"
        filepath = os.path.join(AUDIO_FOLDER, filename)

        tts = gTTS(text=translated, lang=output_language)
        tts.save(filepath)

        with open(filepath, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

        os.remove(filepath)  # Clean up temp file

        return {
            "original_text": text,
            "translated_text": translated,
            "audio_base64": audio_base64,
            "audio_format": "mp3"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Audio serving no longer needed - using base64 responses
