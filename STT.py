from fastapi import FastAPI, File, UploadFile, HTTPException
import torchaudio
import torch
import time
from transformers import pipeline
import uvicorn
import io

app = FastAPI()

# Load ASR model with error handling
try:
    pipe_seamless = pipeline(
        "automatic-speech-recognition",
        model="facebook/seamless-m4t-v2-large",
        trust_remote_code=True
    )
except Exception as e:
    raise RuntimeError(f"Failed to load ASR model: {e}")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Read the file into memory
        audio_bytes = await file.read()
        audio_tensor, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)

        # Convert to mono if multi-channel
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

        # Ensure tensor is in correct format
        audio_tensor = audio_tensor.squeeze(0).numpy().astype("float32")

        # Perform transcription
        start_time = time.time()
        transcription = pipe_seamless(audio_tensor, generate_kwargs={"tgt_lang": "arb"})
        end_time = time.time()

        return {
            "transcription": transcription.get("text", ""),
            "processing_time": f"{end_time - start_time:.2f} seconds"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

# Run server on port 8888 (to match RunPod settings)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
