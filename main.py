from fastapi import FastAPI, UploadFile, Form, Header, HTTPException
from fastapi.responses import FileResponse
from gtts import gTTS
import whisper
from googletrans import Translator
import tempfile
import os
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

API_KEY = os.getenv("API_KEY")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    language: str = Form(...),
    authorization: str = Header(None)
):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API key")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(await file.read())
        video_path = temp_file.name

    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    transcribed = result["text"]

    translator = Translator()
    translated = translator.translate(transcribed, dest=language).text

    audio_path = video_path.replace(".mp4", ".mp3")
    tts = gTTS(translated, lang=language)
    tts.save(audio_path)

    return FileResponse(audio_path, media_type="audio/mpeg", filename="output.mp3")
