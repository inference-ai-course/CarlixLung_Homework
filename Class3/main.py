
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import whisper
from transformers import pipeline
import pyttsx3
from huggingface_hub import login

hf_token = open("hf_token.txt").read()
login(token=hf_token)
app = FastAPI()
asr_model = whisper.load_model("small")
llm = pipeline("text-generation", model="meta-llama/Llama-3.1-8B")
conversation_history = []
tts_engine = pyttsx3.init()

def transcribe_audio(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav")
    return result["text"]

def generate_response(user_text):
    conversation_history.append({"role": "user", "text": user_text})
    # Construct prompt from history
    prompt = ""
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"
    outputs = llm(prompt, max_new_tokens=100)
    bot_response = outputs[0]["generated_text"]
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response

def synthesize_speech(text, filename="./response.wav"):
    tts_engine.save_to_file(text, filename)
    return filename

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    user_text = transcribe_audio(audio_bytes)
    bot_text = generate_response(user_text)
    audio_path = synthesize_speech(bot_text)

    return FileResponse("./response.wav", media_type="audio/wav")
    return JSONResponse({
        "message": "File received successfully!",
        "filename": file.filename,
        "size": len(audio_bytes),
        "note": "ASR → LLM → TTS not implemented yet"
    })
