from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisperx
import os
import torch  # Import torch for cleanup

app = FastAPI()

device = "cuda"
model = whisperx.load_model("large-v2", device=device)


class TranscriptionRequest(BaseModel):
    path: str


class AlignmentRequest(BaseModel):
    audio_path: str
    narration_text: str

@app.post("/transcribe")
async def transcribe_audio_path(req: TranscriptionRequest):
    print("received path:", req.path)
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    try:
        audio = whisperx.load_audio(req.path)
        result = model.transcribe(audio)
        print(result["segments"])  # before alignment

        # Load alignment model
        model_aligned, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )

        # Perform alignment
        result_aligned = whisperx.align(
            result["segments"],
            model=model_aligned,
            align_model_metadata=metadata,
            audio=audio,
            device=device,
        )

        # Clean up GPU memory after successful inference
        del model_aligned
        del metadata
        torch.cuda.empty_cache()

        return result_aligned

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")


@app.post("/align")
async def align_custom_script(req: AlignmentRequest):
    print("Aligning with custom narration:", req.audio_path)
    if not os.path.exists(req.audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    try:
        # Load and transcribe just to get end time
        audio = whisperx.load_audio(req.audio_path)
        result = model.transcribe(audio)
        end_time = result["segments"][-1]["end"]

        # Use the custom narration
        segments = [{"text": req.narration_text, "start": 0, "end": end_time}]

        model_aligned, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(segments, model=model_aligned, align_model_metadata=metadata, audio=audio, device=device)

        del model_aligned, metadata
        torch.cuda.empty_cache()

        return result_aligned

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Alignment error: {str(e)}")