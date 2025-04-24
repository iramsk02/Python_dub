# from fastapi import FastAPI, UploadFile, Form, Header, HTTPException
# from fastapi.responses import FileResponse
# from gtts import gTTS
# import whisper
# from googletrans import Translator
# import tempfile
# import os
# from dotenv import load_dotenv

# app = FastAPI()

# load_dotenv()

# API_KEY = os.getenv("API_KEY")

# @app.get("/")
# def root():
#     return {"message": "FastAPI with Vosk is running!"}

# @app.post("/transcribe")
# async def transcribe_audio(
#     file: UploadFile,
#     language: str = Form(...),
#     authorization: str = Header(None)
# ):
#     if authorization != f"Bearer {API_KEY}":
#         raise HTTPException(status_code=403, detail="Invalid API key")

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
#         temp_file.write(await file.read())
#         video_path = temp_file.name

#     model = whisper.load_model("base")
#     result = model.transcribe(video_path)
#     transcribed = result["text"]

#     translator = Translator()
#     translated = translator.translate(transcribed, dest=language).text

#     audio_path = video_path.replace(".mp4", ".mp3")
#     tts = gTTS(translated, lang=language)
#     tts.save(audio_path)

#     return FileResponse(audio_path, media_type="audio/mpeg", filename="output.mp3")
from fastapi import FastAPI, UploadFile, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from gtts import gTTS
from googletrans import Translator
import tempfile
import os
import subprocess
import speech_recognition as sr
from dotenv import load_dotenv
import aiohttp
import cloudinary
import cloudinary.uploader
from typing import Optional

app = FastAPI()
load_dotenv()

# Load environment variables
API_KEY = os.getenv("API_KEY")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

@app.get("/")
def root():
    return {"message": "FastAPI with CMU Sphinx, Cloudinary, and Translation is running!"}

@app.post("/transcribe")
async def transcribe_audio(
    file: Optional[UploadFile] = None,
    video_url: Optional[str] = Form(None),
    language: str = Form(...),
    authorization: str = Header(None)
):
    # Step 1: API key check
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API key")

    video_path = wav_path = audio_path = final_video_path = None

    try:
        # Step 2: Save the uploaded video or download from URL
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            if file:
                temp_file.write(await file.read())
            elif video_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(video_url) as resp:
                        if resp.status != 200:
                            raise HTTPException(status_code=400, detail="Failed to fetch video from URL.")
                        temp_file.write(await resp.read())
            else:
                raise HTTPException(status_code=400, detail="Provide either a video file or a Cloudinary URL.")
            video_path = temp_file.name

        # Step 3: Extract audio as WAV
        wav_path = video_path.replace(".mp4", ".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Step 4: Transcribe with CMU Sphinx
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_sphinx(audio_data)
        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Could not understand audio.")
        except sr.RequestError:
            raise HTTPException(status_code=500, detail="Sphinx recognition failed.")

        # Step 5: Translate text
        supported_languages = gTTS.langs()
        if language not in supported_languages:
            raise HTTPException(status_code=400, detail=f"Language '{language}' is not supported.")
        translator = Translator()
        translated_text = translator.translate(transcript, dest=language).text

        # Step 6: Convert to speech with gTTS
        audio_path = wav_path.replace(".wav", f"_{language}.mp3")
        gTTS(translated_text, lang=language).save(audio_path)

        # Step 7: Replace video audio
        final_video_path = video_path.replace(".mp4", f"_translated_{language}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", final_video_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Step 8: Upload new video to Cloudinary
        upload_result = cloudinary.uploader.upload_large(final_video_path, resource_type="video")
        cloudinary_url = upload_result.get("secure_url")

        return JSONResponse(content={
            "message": "Translated video created and uploaded successfully.",
            "transcript": transcript,
            "translated_text": translated_text,
            "cloudinary_url": cloudinary_url
        })

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr.decode()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        for path in [video_path, wav_path, audio_path, final_video_path]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Cleanup error for {path}: {e}")

# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
