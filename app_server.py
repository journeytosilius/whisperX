from fastapi import FastAPI, UploadFile, File, HTTPException
import whisperx
import tempfile
import os
import torchaudio

app = FastAPI()

device = "cuda"

# Load model on startup
model = whisperx.load_model("large-v2", device=device)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp3", ".wav", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Transcribe
        audio = whisperx.load_audio(tmp_path)
        result = model.transcribe(audio)

        # Align words
        model_aligned = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], model_aligned, audio, device=device)

        return result_aligned  # Contains 'segments' and 'word_segments' with timestamps

    finally:
        os.remove(tmp_path)
