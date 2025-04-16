from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisperx
import os

app = FastAPI()

device = "cuda"
model = whisperx.load_model("large-v2", device=device)

class TranscriptionRequest(BaseModel):
    path: str

@app.post("/transcribe")
async def transcribe_audio_path(req: TranscriptionRequest):
    print("received path:", req.path)
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    try:
        audio = whisperx.load_audio(req.path)
        result = model.transcribe(audio)
        print(result["segments"]) # before alignment

        # Align words
        model_aligned, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], model=model_aligned, align_model_metadata=metadata, audio=audio, device=device)
        
        print(result_aligned["segments"]) # after alignment

        return result_aligned  # Contains 'segments' and 'word_segments'
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
