from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisperx
import os
import torch

app = FastAPI()

device = "cuda"
AUDIO_DIR = "/app/output"

class TranscriptionRequest(BaseModel):
    path: str

class AlignmentRequest(BaseModel):
    audio_path: str
    narration_text: str

@app.post("/transcribe")
async def transcribe_audio_path(req: TranscriptionRequest):
    full_path = os.path.join(AUDIO_DIR, req.path)
    print("received path:", full_path)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    try:
        audio = whisperx.load_audio(full_path)

        # 🔹 Load the model only when needed
        model = whisperx.load_model("large-v2", device=device)
        result = model.transcribe(audio)
        print(result["segments"])

        model_aligned, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result_aligned = whisperx.align(
            result["segments"],
            model=model_aligned,
            align_model_metadata=metadata,
            audio=audio,
            device=device,
        )

        del model, model_aligned, metadata
        torch.cuda.empty_cache()

        return result_aligned

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/align")
async def align_custom_script(req: AlignmentRequest):
    full_path = os.path.join(AUDIO_DIR, req.audio_path)
    print("Aligning with custom narration:", full_path)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    try:
        audio = whisperx.load_audio(full_path)

        # 🔹 Load WhisperX for language detection (if needed)
        model = whisperx.load_model("large-v2", device=device)
        result = model.transcribe(audio)
        end_time = result["segments"][-1]["end"]

        segments = [{"text": req.narration_text, "start": 0, "end": end_time}]

        model_aligned, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result_aligned = whisperx.align(
            segments,
            model=model_aligned,
            align_model_metadata=metadata,
            audio=audio,
            device=device,
        )

        del model, model_aligned, metadata
        torch.cuda.empty_cache()

        return result_aligned

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Alignment error: {str(e)}")
