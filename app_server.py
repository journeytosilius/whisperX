from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisperx
import os
import torch
import gc

# --- Configuration ---
AUDIO_DIR = "/app/output"

# Check for GPU availability
GPU_AVAILABLE = torch.cuda.is_available()

# --- Device and Compute Type Configuration ---
CPU_DEVICE = "cpu"
CPU_COMPUTE_TYPE = "int8" # Recommended for CPU

if GPU_AVAILABLE:
    GPU_DEVICE = "cuda"
    GPU_COMPUTE_TYPE = "float16" # Recommended for GPU
    print("GPU detected. Both CPU and GPU endpoints will be available.")
else:
    print("No GPU detected. Only CPU endpoints will be available.")

app = FastAPI()

# --- Model Storage ---
# We will store models in dictionaries to access them easily
models_cpu = {"align": {}}
models_gpu = {"align": {}}

# --- Model Loading (at startup) ---
print("Loading CPU models (this might take a moment)...")
# Load the main Whisper model for CPU
model_cpu = whisperx.load_model("large-v2", CPU_DEVICE, compute_type=CPU_COMPUTE_TYPE)
models_cpu["whisper"] = model_cpu

# Pre-load English alignment model for CPU
try:
    align_model_en_cpu, metadata_en_cpu = whisperx.load_align_model(language_code="en", device=CPU_DEVICE)
    models_cpu["align"]["en"] = (align_model_en_cpu, metadata_en_cpu)
    print("CPU models loaded.")
except Exception as e:
    print(f"Warning: Could not load CPU alignment model for English. Error: {e}")


# Load GPU models only if a GPU is available
if GPU_AVAILABLE:
    print("Loading GPU models (this might take a moment)...")
    # Load the main Whisper model for GPU
    model_gpu = whisperx.load_model("large-v2", GPU_DEVICE, compute_type=GPU_COMPUTE_TYPE)
    models_gpu["whisper"] = model_gpu

    # Pre-load English alignment model for GPU
    try:
        align_model_en_gpu, metadata_en_gpu = whisperx.load_align_model(language_code="en", device=GPU_DEVICE)
        models_gpu["align"]["en"] = (align_model_en_gpu, metadata_en_gpu)
        print("GPU models loaded.")
    except Exception as e:
        print(f"Warning: Could not load GPU alignment model for English. Error: {e}")

print("Server is ready.")

# --- Pydantic Models ---
class TranscriptionRequest(BaseModel):
    path: str

class AlignmentRequest(BaseModel):
    audio_path: str
    narration_text: str

# --- Helper Function for Dynamic Alignment Model Loading ---
def get_align_model(language_code: str, device: str):
    models_dict = models_cpu if device == "cpu" else models_gpu
    if language_code not in models_dict["align"]:
        print(f"Alignment model for '{language_code}' on device '{device}' not pre-loaded. Loading now...")
        try:
            model_aligned, metadata = whisperx.load_align_model(language_code=language_code, device=device)
            models_dict["align"][language_code] = (model_aligned, metadata)
            print(f"Loaded and cached alignment model for '{language_code}' on '{device}'.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load alignment model for language: {language_code}. Error: {e}")
    return models_dict["align"][language_code]


# --- CPU Endpoints ---

@app.post("/transcribe_cpu")
async def transcribe_cpu(req: TranscriptionRequest):
    full_path = os.path.join(AUDIO_DIR, req.path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    try:
        audio = whisperx.load_audio(full_path)
        result = models_cpu["whisper"].transcribe(audio, batch_size=4)
        detected_language = result["language"]
        
        model_aligned, metadata = get_align_model(detected_language, CPU_DEVICE)
        
        result_aligned = whisperx.align(result["segments"], model_aligned, metadata, audio, CPU_DEVICE)
        
        del audio, result
        gc.collect()
        return {"detected_language": detected_language, "segments": result_aligned["segments"]}
    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"CPU Transcription error: {str(e)}")

@app.post("/align_cpu")
async def align_cpu(req: AlignmentRequest):
    full_path = os.path.join(AUDIO_DIR, req.audio_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
        
    try:
        audio = whisperx.load_audio(full_path)
        # Detect language first to get the correct alignment model
        result_lang = models_cpu["whisper"].transcribe(audio, batch_size=4)
        detected_language = result_lang["language"]
        
        segments = [{"text": req.narration_text, "start": 0.0, "end": len(audio) / 16000}]
        
        model_aligned, metadata = get_align_model(detected_language, CPU_DEVICE)

        result_aligned = whisperx.align(segments, model_aligned, metadata, audio, CPU_DEVICE)
        
        del audio, result_lang
        gc.collect()
        return result_aligned
    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"CPU Alignment error: {str(e)}")


# --- GPU Endpoints ---

@app.post("/transcribe_gpu")
async def transcribe_gpu(req: TranscriptionRequest):
    if not GPU_AVAILABLE:
        raise HTTPException(status_code=503, detail="GPU not available on this server.")

    full_path = os.path.join(AUDIO_DIR, req.path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    try:
        audio = whisperx.load_audio(full_path)
        result = models_gpu["whisper"].transcribe(audio, batch_size=16)
        detected_language = result["language"]
        
        model_aligned, metadata = get_align_model(detected_language, GPU_DEVICE)
        
        result_aligned = whisperx.align(result["segments"], model_aligned, metadata, audio, GPU_DEVICE)

        del audio, result
        gc.collect()
        torch.cuda.empty_cache()
        return {"detected_language": detected_language, "segments": result_aligned["segments"]}
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"GPU Transcription error: {str(e)}")

@app.post("/align_gpu")
async def align_gpu(req: AlignmentRequest):
    if not GPU_AVAILABLE:
        raise HTTPException(status_code=503, detail="GPU not available on this server.")

    full_path = os.path.join(AUDIO_DIR, req.audio_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
        
    try:
        audio = whisperx.load_audio(full_path)
        # Detect language first to get the correct alignment model
        result_lang = models_gpu["whisper"].transcribe(audio, batch_size=16)
        detected_language = result_lang["language"]
        
        segments = [{"text": req.narration_text, "start": 0.0, "end": len(audio) / 16000}]
        
        model_aligned, metadata = get_align_model(detected_language, GPU_DEVICE)

        result_aligned = whisperx.align(segments, model_aligned, metadata, audio, GPU_DEVICE)
        
        del audio, result_lang
        gc.collect()
        torch.cuda.empty_cache()
        return result_aligned
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"GPU Alignment error: {str(e)}")