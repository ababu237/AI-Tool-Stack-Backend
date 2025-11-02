# PDF Documents API
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from deep_translator import GoogleTranslator
from PyPDF2 import PdfReader
from gtts import gTTS
from langdetect import detect
import base64
import os
import io
import uuid
import pandas as pd
import json
import time
import traceback
import uvicorn
from datetime import datetime, timedelta

app = FastAPI(title="Document Processing API",
              description="PDF document analysis and Q&A")

# Add root endpoint


@app.get("/")
async def root():
    return {
        "message": "Document Processing API",
        "description": "PDF document analysis and Q&A with multi-language support",
        "endpoints": {
            "process_document": "POST /process_document",
            "audio": "/audio/{filename}",
            "csv": "/csv/{filename}",
            "documentation": "/docs"
        },
        "usage": {
            "endpoint": "/process_document",
            "method": "POST",
            "parameters": {
                "file": "PDF document to analyze (first time) OR file_id for follow-up",
                "question": "Question about the document",
                "output_language": "Language code for response (e.g., 'en', 'es', 'fr')"
            }
        },
        "features": ["PDF text extraction", "Vector search", "Multi-language Q&A", "Text-to-speech"]
    }

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up folders
AUDIO_DIR = "audio_outputs"
CSV_DIR = "translated_csvs"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Static directories no longer needed - using base64 responses

# Load API keys from config.json file
try:
    with open("config/config.json") as config_file:
        config = json.load(config_file)
        OPENAI_API_KEY = config.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("WARNING: OPENAI_API_KEY not found or is placeholder in config.json. OpenAI dependent functions will fail.")
except FileNotFoundError:
    print("ERROR: config/config.json not found. Please ensure the configuration file exists.")
    OPENAI_API_KEY = None
except json.JSONDecodeError:
    print("ERROR: Error decoding config.json. Please ensure it is valid JSON.")
    OPENAI_API_KEY = None

# Temporary storage for files
temp_storage = {}

# --- Utility Functions ---


def extract_text_from_pdf(pdf_file_stream):
    reader = PdfReader(pdf_file_stream)
    return ''.join(page.extract_text() or "" for page in reader.pages)


def detect_language(text):
    try:
        if not text or len(text.strip()) < 10:
            return "en"
        return detect(text)
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"


def translate_text(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        return translated if translated else text
    except Exception as e:
        print(
            f"Translation error from {source_lang} to {target_lang}: {str(e)}")
        return text


def text_to_speech_base64(text, lang='en'):
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
        print(f"gTTS error for lang {lang}: {str(e)}")
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
                pass
        return None


def cleanup_temp_storage():
    current_time = datetime.now()
    keys_to_remove = [
        key for key, (_, _, timestamp) in temp_storage.items()
        if current_time - timestamp > timedelta(minutes=30)
    ]
    for key in keys_to_remove:
        if key in temp_storage:
            temp_storage.pop(key, None)
            print(f"Cleaned up expired file_id: {key}")

# --- Main Endpoint ---


# Endpoint moved to main.py - @app.post("/process_document")
async def process_document(
    background_tasks: BackgroundTasks,
    file_id: str = Form(None),
    file: UploadFile = File(None),
    question: str = Form(...),
    output_language: str = Form("en")  # Default to English
):
    background_tasks.add_task(cleanup_temp_storage)

    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "OpenAI API key not configured."})

    try:
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        knowledge_base = None
        source_language = "en"
        current_file_id = None
        conversation_history = []

        if file and question:
            # Handle initial file upload and question
            file_content = await file.read()
            current_file_id = str(uuid.uuid4())
            temp_storage[current_file_id] = (
                file_content, datetime.now(), conversation_history)

            # Process the file and answer the question
            filename = "temp_file.pdf"  # Assuming PDF for simplicity

            if filename.endswith(".pdf"):
                pdf = io.BytesIO(file_content)
                text = extract_text_from_pdf(pdf)
                source_language = detect_language(text)

                splitter = CharacterTextSplitter(
                    separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
                )
                chunks = splitter.split_text(text)

                if not chunks:
                    return JSONResponse(status_code=400, content={"error": "Could not split text into chunks. The document might be too short or have an unsupported structure."})

                knowledge_base = FAISS.from_texts(chunks, embeddings)
                temp_storage[current_file_id] = (
                    knowledge_base, source_language, datetime.now(), conversation_history)

                docs = knowledge_base.similarity_search(
                    question, k=4)
                chain = load_qa_chain(llm, chain_type="stuff")

                with get_openai_callback() as cb:
                    answer_in_source_lang = chain.run(
                        input_documents=docs, question=question)
                    print(f"OpenAI API call: {cb}")

                final_answer = translate_text(
                    answer_in_source_lang, source_language, output_language)

                audio_base64 = text_to_speech_base64(
                    final_answer, lang=output_language)

                # Append the first Q&A pair to the history
                conversation_history.append(
                    {"question": question, "answer": final_answer})

                # Return file_id, response, audio_base64, and history for initial request
                return {
                    "file_id": current_file_id,
                    "response": final_answer,
                    "audio_base64": audio_base64 if audio_base64 else None,
                    "audio_format": "mp3" if audio_base64 else None,
                    "conversation_history": conversation_history
                }
            else:
                return JSONResponse(status_code=400, content={"error": "Only PDF files are supported."})

        elif file_id and question:
            # Handle subsequent questions with file_id
            file_data = temp_storage.get(file_id)
            if not file_data:
                return JSONResponse(status_code=404, content={"error": "File ID not found or expired. Please upload the document again."})

            if len(file_data) == 4:
                knowledge_base, source_language, timestamp, conversation_history = file_data
            else:
                knowledge_base, source_language, timestamp = file_data
                conversation_history = []

            # Process the question
            docs = knowledge_base.similarity_search(question, k=4)
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                answer_in_source_lang = chain.run(
                    input_documents=docs, question=question)
                print(f"OpenAI API call: {cb}")

            final_answer = translate_text(
                answer_in_source_lang, source_language, output_language)

            audio_base64 = text_to_speech_base64(
                final_answer, lang=output_language)

            # Append the new Q&A pair to the history
            conversation_history.append(
                {"question": question, "answer": final_answer})

            # Update the entry in temp_storage with the updated history and a fresh timestamp
            temp_storage[file_id] = (
                knowledge_base, source_language, datetime.now(), conversation_history)

            # Return response, audio_base64, and history for subsequent requests
            return {
                "response": final_answer,
                "audio_base64": audio_base64 if audio_base64 else None,
                "audio_format": "mp3" if audio_base64 else None,
                "conversation_history": conversation_history
            }

        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid request. Provide 'file' and 'question' for initial processing, or 'file_id' and 'question' for subsequent queries."}
            )

    except Exception as e:
        print(
            f"An unexpected error occurred processing document (File ID: {current_file_id}): {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"An internal server error occurred: {str(e)}"})

# --- Test Endpoint ---


# Endpoint moved to main.py - @app.post("/test_document")
async def test_document_api(
    test_query: str = Form("What is this API for?"),
    output_language: str = Form("en")
):
    """Simple test endpoint that doesn't require file upload"""
    try:
        # Simulate a simple response without file processing
        response = f"Document Processing API test successful! Your query: '{test_query}' in language: {output_language}"

        # Generate test audio response
        tts = gTTS(text=response, lang=output_language)
        filename = f"test_audio_{uuid.uuid4()}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        tts.save(filepath)

        return {
            "message": "Document API test successful",
            "test_query": test_query,
            "response": response,
            "output_language": output_language,
            "audio_url": f"/audio/{filename}",
            "note": "This is a test endpoint. For actual document processing, use /process_document with file upload"
        }
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists("config"):
        os.makedirs("config")
    config_path = "config/config.json"
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({"OPENAI_API_KEY": "YOUR_OPENAI_API_KEY_HERE"}, f)
        print(
            f"Created a dummy {config_path}. Please replace with your actual OpenAI API key.")

    uvicorn.run(app, host="0.0.0.0", port=8000)
