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
from fastapi.responses import FileResponse
from gtts import gTTS
from googletrans import Translator
import tempfile
import os
import subprocess
import speech_recognition as sr
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("API_KEY")  # Get the API key from environment variables

@app.get("/")
def root():
    return {"message": "FastAPI with CMU Sphinx and video audio replacement is running!"}

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    language: str = Form(...),
    authorization: str = Header(None)
):
    # API key validation
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API key")

    video_path = None
    wav_path = None
    audio_path = None
    final_video_path = None

    try:
        # Step 1: Save the uploaded video file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(await file.read())
            video_path = temp_file.name

        # Step 2: Convert the video file to a WAV audio file using ffmpeg
        wav_path = video_path.replace(".mp4", ".wav")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {result.stderr.decode()}")

        # Step 3: Initialize the speech recognizer and recognize speech from the WAV file
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)  # Record the audio from the file

        # Step 4: Transcribe the audio using CMU Sphinx
        try:
            transcript = recognizer.recognize_sphinx(audio)
        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Speech could not be understood.")
        except sr.RequestError:
            raise HTTPException(status_code=500, detail="Sphinx recognition service failed.")

        # Step 5: Validate if the target language is supported by gTTS
        supported_languages = gTTS.langs()
        if language not in supported_languages:
            raise HTTPException(status_code=400, detail=f"Language '{language}' is not supported by gTTS.")

        # Step 6: Translate the transcript to the desired language
        translator = Translator()
        translated_text = translator.translate(transcript, dest=language).text

        # Step 7: Convert translated text to speech using gTTS
        audio_path = wav_path.replace(".wav", f"_{language}.mp3")
        tts = gTTS(translated_text, lang=language)
        tts.save(audio_path)

        # Step 8: Burn the translated audio into the video using ffmpeg
        final_video_path = video_path.replace(".mp4", f"_translated_{language}.mp4")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", final_video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {result.stderr.decode()}")

        # Step 9: Return the video with burned-in translated audio
        return FileResponse(final_video_path, media_type="video/mp4", filename="translated_video.mp4")

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary files
        for path in [video_path, wav_path, audio_path, final_video_path]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception as cleanup_error:
                print(f"Error cleaning up file {path}: {cleanup_error}")

# Run locally or on Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

