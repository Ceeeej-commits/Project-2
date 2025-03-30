from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import openai
import json
import aiofiles
import fitz  # PyMuPDF for PDFs
import pandas as pd
import os
import pytesseract
from PIL import Image
import whisper
import logging
import subprocess
subprocess.run(["pip", "install", "openai-whisper", "pymupdf"], check=True)

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduced log verbosity

# Load OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please check your environment variables.")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Load a smaller Whisper model
whisper_model = whisper.load_model("tiny")  # Switched from "base" to "tiny"

class Query(BaseModel):
    question: str

@app.get("/")
async def read_root():
    return {"message": "TDS Solver API is running!"}

@app.post("/api/ask")
async def ask_question(query: Query):
    """Handles direct text-based queries."""
    try:
        logging.info(f"Received question: {query.question[:100]}")  # Log only first 100 chars
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer IIT Madras graded assignment questions accurately."},
                {"role": "user", "content": query.question}
            ]
        )
        answer = response.choices[0].message.content
        logging.info(f"OpenAI response: {answer[:100]}")  # Log only first 100 chars
        return {"answer": answer}
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles file uploads and extracts relevant text."""
    file_ext = file.filename.split(".")[-1].lower()
    temp_file = f"/tmp/temp.{file_ext}"  # Store in /tmp for serverless compatibility

    async with aiofiles.open(temp_file, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        extracted_text = extract_text(temp_file, file_ext)
        os.remove(temp_file)  # Cleanup
        logging.info(f"Extracted text: {extracted_text[:200]}")  # Limit log size
        return await ask_question(Query(question=extracted_text))
    except Exception as e:
        os.remove(temp_file)
        logging.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

def extract_text(filepath, file_ext):
    """Extract text based on file type."""
    try:
        if file_ext == "pdf":
            return extract_text_from_pdf(filepath)
        elif file_ext in ["xls", "xlsx"]:
            return extract_text_from_xlsx(filepath)
        elif file_ext == "json":
            return extract_text_from_json(filepath)
        elif file_ext in ["png", "jpg", "jpeg"]:
            return extract_text_from_image(filepath)
        elif file_ext in ["mp3", "wav"]:
            return transcribe_audio(filepath)
    except Exception as e:
        logging.error(f"Error extracting text from {file_ext}: {str(e)}")
        return f"Error extracting text: {str(e)}"
    return ""

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_xlsx(filepath):
    df = pd.read_excel(filepath)
    return df.to_string()

def extract_text_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=2)

def extract_text_from_image(filepath):
    return pytesseract.image_to_string(Image.open(filepath))

def transcribe_audio(filepath):
    result = whisper_model.transcribe(filepath)
    return result["text"]

@app.on_event("startup")
async def startup_event():
    logging.info("Application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Application shutdown complete.")
