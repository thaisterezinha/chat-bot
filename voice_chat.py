import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from gtts import gTTS
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

model = whisper.load_model("base")

def gravar_audio():
    fs = 44100
    segundos = 5
    
    print("🎤 Fale agora...")
    audio = sd.rec(int(segundos * fs), samplerate=fs, channels=1)
    sd.wait()
    
    write("input.wav", fs, audio)

def transcrever():
    result = model.transcribe("input.wav")
    return result["text"]

def perguntar(texto):
    resposta = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": texto}]
    )
    return resposta.choices[0].message.content

def falar(texto):
    tts = gTTS(text=texto, lang='pt')
    tts.save("resposta.mp3")
    os.system("start resposta.mp3")

while True:
    gravar_audio()
    
    texto = transcrever()
    print("🗣️ Você:", texto)
    
    resposta = perguntar(texto)
    print("🤖 ChatGPT:", resposta)
    
    falar(resposta)