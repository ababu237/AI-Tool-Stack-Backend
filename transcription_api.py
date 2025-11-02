import os
import uuid
import json
import base64
import openai
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from deep_translator import GoogleTranslator
from langdetect import detect

app = FastAPI(title="Audio Transcription API",
              description="Audio file transcription using OpenAI Whisper")

# Add root endpoint


# Endpoint moved to main.py - @app.post("/api_info")
async def get_api_info():
    return {
        "message": "Audio Transcription API",
        "description": "Audio file transcription using OpenAI Whisper with translation",
        "endpoints": {
            "transcribe_audio": "POST /transcribe_audio",
            "test_transcription": "POST /test_transcription",
            "api_info": "POST /api_info",
            "documentation": "/docs"
        },
        "usage": {
            "endpoint": "/transcribe_audio",
            "method": "POST",
            "parameters": {
                "audio_file": "Audio file to transcribe",
                "output_language": "Target language code (e.g., 'en', 'es', 'fr')"
            }
        },
        "response_format": {
            "audio_base64": "Base64 encoded MP3 audio data",
            "audio_format": "mp3"
        }
    }

# Add CORS middleware RIGHT AFTER app creation
app.add_middleware(
    CORSMiddleware,
    # Specify your frontend origin
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

AUDIO_FOLDER = "audio_transcription"
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

# Translate text


def translate_text(text, output_lang):
    if not text.strip():
        return ""
    try:
        input_lang = detect(text)
    except:
        input_lang = "en"  # Default to English if detection fails
    return GoogleTranslator(source=input_lang, target=output_lang).translate(text)

# Text-to-speech that returns base64


def text_to_speech_base64(text: str) -> str:
    if not text.strip():
        raise ValueError("No text to send to TTS API")
    filename = f"temp_audio_{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)
    tts = gTTS(text)
    tts.save(filepath)

    with open(filepath, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

    os.remove(filepath)  # Clean up temp file
    return audio_base64


# Endpoint moved to main.py - @app.post("/transcribe_audio")
async def api_transcribe_audio(
    audio_file: UploadFile = File(...),
    output_language: str = Form("en")
):
    try:
        # Save temporary audio file
        audio_path = f"temp_{audio_file.filename}"
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        # Transcribe audio using OpenAI's Whisper model
        openai.api_key = OPENAI_API_KEY  # Use the API key from config.json
        with open(audio_path, "rb") as f:
            transcription = openai.Audio.transcribe(
                model="whisper-1",
                file=f
            )

        transcription_text = transcription['text']

        # Translate transcription text
        translated_response = translate_text(
            transcription_text, output_language)

        # Convert translated text to speech
        audio_base64 = text_to_speech_base64(translated_response)

        # Store conversation history
        conversation_entry = {
            "input_audio_file": audio_file.filename,
            "transcribed_text": transcription_text,
            "translated_text": translated_response,
            "audio_base64": audio_base64
        }
        conversation_history.append(conversation_entry)

        # Clean up temporary audio file
        os.remove(audio_path)

        return {
            "transcribed_text": transcription_text,
            "translated_text": translated_response,
            "audio_base64": audio_base64,
            "audio_format": "mp3",
            "conversation_history": conversation_history
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Audio serving no longer needed - using base64 responses


# Endpoint moved to main.py - @app.post("/test_transcription")
async def test_transcription_api(
    test_text: str = Form("Hello, this is a test transcription"),
    output_language: str = Form("en")
):
    """Simple test endpoint that doesn't require audio file upload"""
    try:
        # Simulate transcription response
        response = f"Transcription API test successful! Simulated transcription: '{test_text}' in language: {output_language}"

        # Generate test audio response as base64
        audio_base64 = text_to_speech_base64(response)

        return {
            "message": "Transcription API test successful",
            "simulated_transcription": test_text,
            "response": response,
            "output_language": output_language,
            "audio_base64": audio_base64,
            "audio_format": "mp3",
            "note": "This is a test endpoint. For actual audio transcription, use /transcribe_audio with audio file upload"
        }
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}
