# FastAPI imports removed - endpoints now in main.py
from fastapi import File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import base64
from langchain_core.agents import create_csv_agent
from langchain_community.chat_models import ChatOpenAI
from deep_translator import GoogleTranslator
from gtts import gTTS
from langdetect import detect
from typing import Optional
import datetime
import os
import io
import uuid
import pandas as pd
import json
import traceback
# FastAPI app and CORS removed - now handled in main.py

# Constants and Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio_outputs")
CSV_DIR = os.path.join(BASE_DIR, "translated_csvs")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Create necessary directories
for directory in [AUDIO_DIR, CSV_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Static directories no longer needed - using base64 responses


def load_config():
    config_path = os.path.join(CONFIG_DIR, "config.json")
    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
            api_key = config.get("OPENAI_API_KEY")
            if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
                raise ValueError("Invalid API key in config.json")
            return api_key
    except Exception as e:
        print(f"Config error: {str(e)}")
        return None


OPENAI_API_KEY = load_config()


def detect_language(text: str) -> str:
    try:
        if not text or len(text.strip()) < 10:
            return "en"
        return detect(text)
    except Exception:
        return "en"


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if source_lang == target_lang:
        return text
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text


def text_to_speech_base64(text: str, lang: str = 'en') -> Optional[str]:
    filename = f"temp_{uuid.uuid4().hex}.mp3"
    path = os.path.join(AUDIO_DIR, filename)
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(path)

        with open(path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

        os.remove(path)  # Clean up temp file
        return audio_base64
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        if lang != 'en':
            try:
                tts_en = gTTS(text=text, lang='en')
                tts_en.save(path)

                with open(path, "rb") as audio_file:
                    audio_base64 = base64.b64encode(
                        audio_file.read()).decode('utf-8')

                os.remove(path)
                return audio_base64
            except:
                return None
        return None


def translate_csv(file_bytes: bytes, output_lang: str) -> str:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        translated_df_data = {}

        # Translate headers
        translated_headers = [
            translate_text(str(col), detect_language(str(col)), output_lang)
            for col in df.columns
        ]

        # Translate content
        for col_idx, original_col_name in enumerate(df.columns):
            translated_col_name = translated_headers[col_idx]
            translated_df_data[translated_col_name] = [
                translate_text(str(item), detect_language(
                    str(item)), output_lang)
                for item in df[original_col_name]
            ]

        output_df = pd.DataFrame(translated_df_data)
        output_filename = f"translated_{uuid.uuid4().hex}.csv"
        output_path = os.path.join(CSV_DIR, output_filename)
        output_df.to_csv(output_path, index=False)

        return f"/csv/{output_filename}"
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error translating CSV: {str(e)}")


# Temporary storage for files
temp_storage = {}


def cleanup_temp_storage():
    current_time = datetime.datetime.now()
    keys_to_remove = [
        key for key, (_, timestamp) in temp_storage.items()
        if current_time - timestamp > datetime.timedelta(minutes=30)
    ]
    for key in keys_to_remove:
        if key in temp_storage:
            temp_storage.pop(key, None)
            print(f"Cleaned up expired file_id: {key}")


# Function for CSV processing (used by main.py)
async def process_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    file_id: str = Form(None),
    question: str = Form(...),
    output_language: str = Form("en")
):
    background_tasks.add_task(cleanup_temp_storage)

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500, detail="OpenAI API key not configured.")

    if file and not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=400, detail="Only CSV files are supported.")

    try:
        if file and question:
            # Handle initial file upload and question
            file_content = await file.read()
            file_id = str(uuid.uuid4())
            conversation_history = []
            temp_storage[file_id] = (
                file_content, datetime.datetime.now(), conversation_history)

            # Process the file and answer the question
            temp_csv_path = os.path.join(
                BASE_DIR, f"temp_{uuid.uuid4().hex}.csv")
            with open(temp_csv_path, 'wb') as temp_f:
                temp_f.write(file_content)

            agent = create_csv_agent(
                ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY,
                           model_name="gpt-3.5-turbo"),
                temp_csv_path,
                verbose=True,
                allow_dangerous_code=True
            )

            question_lang = detect_language(question)
            english_question = translate_text(question, question_lang, "en")
            answer_in_english = agent.run(english_question)
            final_answer = translate_text(
                answer_in_english, "en", output_language)
            audio_base64 = text_to_speech_base64(
                final_answer, lang=output_language)

            # Append the first Q&A pair to the history
            conversation_history.append(
                {"question": question, "answer": final_answer})

            # Clean up the temporary file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)

            # Return file_id, response, audio_base64, and history for initial request
            return {
                "file_id": file_id,
                "response": final_answer,
                "audio_base64": audio_base64 if audio_base64 else None,
                "audio_format": "mp3" if audio_base64 else None,
                "conversation_history": conversation_history
            }

        elif file_id and question:
            # Handle subsequent questions with file_id
            file_data = temp_storage.get(file_id)
            if not file_data:
                raise HTTPException(
                    status_code=404, detail="File ID not found or expired. Please upload the document again.")

            file_content, _, conversation_history = file_data
            temp_csv_path = os.path.join(
                BASE_DIR, f"temp_{uuid.uuid4().hex}.csv")
            with open(temp_csv_path, 'wb') as temp_f:
                temp_f.write(file_content)

            agent = create_csv_agent(
                ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY,
                           model_name="gpt-3.5-turbo"),
                temp_csv_path,
                verbose=True,
                allow_dangerous_code=True
            )

            question_lang = detect_language(question)
            english_question = translate_text(question, question_lang, "en")
            answer_in_english = agent.run(english_question)
            final_answer = translate_text(
                answer_in_english, "en", output_language)
            audio_base64 = text_to_speech_base64(
                final_answer, lang=output_language)

            # Append the new Q&A pair to the history
            conversation_history.append(
                {"question": question, "answer": final_answer})

            # Clean up the temporary file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)

            # Return response, audio_base64, and history for subsequent requests
            return {
                "response": final_answer,
                "audio_base64": audio_base64 if audio_base64 else None,
                "audio_format": "mp3" if audio_base64 else None,
                "conversation_history": conversation_history
            }

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request. Provide 'file' and 'question' for initial processing, or 'file_id' and 'question' for subsequent queries."
            )

    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error processing CSV: {str(e)}")


# Test function (used by main.py)
async def test_csv_api(
    test_question: str = Form("What insights can you provide?"),
    output_language: str = Form("en")
):
    """Simple test endpoint that doesn't require CSV file upload"""
    try:
        # Simulate CSV analysis response
        response = f"CSV Processing API test successful! Your question: '{test_question}' in language: {output_language}"

        # Simulate CSV insights
        insights = {
            "summary": "This is a test response for CSV analysis",
            "row_count": "Sample: 100 rows",
            "columns": "Sample: 5 columns",
            "analysis": f"Analysis for: {test_question}"
        }

        # Generate test audio response as base64
        audio_base64 = text_to_speech_base64(response, lang=output_language)

        return {
            "message": "CSV Processing API test successful",
            "question": test_question,
            "response": response,
            "insights": insights,
            "output_language": output_language,
            "audio_base64": audio_base64 if audio_base64 else None,
            "audio_format": "mp3" if audio_base64 else None,
            "note": "This is a test endpoint. For actual CSV processing, use /process_csv with CSV file upload"
        }
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}


# API info function (used by main.py)
def get_csv_api_info():
    return {
        "message": "CSV Processing API",
        "description": "CSV file processing and translation with AI-powered analysis",
        "endpoints": {
            "process_csv": "POST /process_csv",
            "test_csv": "POST /test_csv",
            "api_info": "POST /api_info",
            "documentation": "/docs"
        },
        "usage": {
            "endpoint": "/process_csv",
            "method": "POST",
            "parameters": {
                "file": "CSV file to process",
                "question": "Question about the CSV data",
                "output_language": "Language code (e.g., 'en', 'es', 'fr')"
            }
        },
        "features": [
            "CSV data analysis",
            "Multi-language translation",
            "AI-powered Q&A",
            "Base64 audio output"
        ]
    }


# Run the application
if __name__ == "__main__":
    import uvicorn

    # Create default config if it doesn't exist
    config_path = os.path.join(CONFIG_DIR, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({"OPENAI_API_KEY": "YOUR_OPENAI_API_KEY_HERE"}, f)
        print(
            f"Created config file at {config_path}. Please add your OpenAI API key.")

    uvicorn.run(app, host="0.0.0.0", port=8000)
