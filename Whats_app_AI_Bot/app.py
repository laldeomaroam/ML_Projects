from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
import os
import requests
import tempfile
import subprocess
from faster_whisper import WhisperModel
from langdetect import detect
from google.cloud import translate_v2 as translate
from pydub import AudioSegment
from dotenv import load_dotenv
import traceback
from openai import OpenAI
from ollama import Client as OllamaClient

load_dotenv()  # This will read from your .env file

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
twilio_client = TwilioClient(TWILIO_SID, TWILIO_AUTH)
ollama_client = OllamaClient(host='http://host.docker.internal:11434')  # or 'http://localhost:11434' if not in Docker

# Choose your LLM
USE_LOCAL_LLM = True  # 👈 Toggle: True = Ollama, False = OpenAI

# Import for Local Ollama
if USE_LOCAL_LLM:
    import ollama
# Import for OpenAI
else:
    import openai
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  

from google.cloud import translate_v2 as translate


app = Flask(__name__)

translate_client = translate.Client()

SUPPORTED_LANGUAGES = {
    "fr": "french",
    "hi": "hindi",
    "bn": "bengali",
    "mfe": "mauritian creole"
}

# Load Whisper model
model_size = "small"
whisper_model = WhisperModel(model_size, compute_type="int8")


def transcribe_audio(audio_path):
    segments, _ = whisper_model.transcribe(audio_path, beam_size=5)
    return " ".join([segment.text for segment in segments])

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_text(text, target_language="en"):
    try:
        result = translate_client.translate(text, target_language=target_language)
        return result["translatedText"]
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def back_translate_text(text, target_language):
    try:
        return translate_text(text, target_language=target_language)
    except:
        return text

def generate_response(prompt):
    try:
        if USE_LOCAL_LLM:
            # 🧠 Use Ollama (local model)
            response = ollama_client.chat(
                model='llama2',  # or 'mistral', 'gemma', etc.
                messages=[{"role": "user", "content": prompt}]
            )
            print(response)

            return response.message["content"]
        else:
            # 🤖 Use OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        traceback.print_exc()  # 👈 Add this to see full error
        return "Oops! I had trouble generating a response."


@app.route("/bot", methods=["POST"])
def bot():
    incoming_msg = request.values.get('Body', '').strip()
    media_url = request.values.get('MediaUrl0', '')

    if not incoming_msg and media_url:
        audio_ext = request.values.get("MediaContentType0", "").split("/")[-1]
        audio_path = f"temp.{audio_ext}"

        # Twilio-hosted media URL requires authentication
        media_resp = requests.get(media_url, auth=(TWILIO_SID, TWILIO_AUTH))
        if media_resp.status_code != 200:
            print("Failed to fetch media:", media_resp.text)
            return "Sorry, could not process audio."


        #audio_data = request.get_data()

        with open(audio_path, 'wb') as f:
            f.write(media_resp.content)

        # Convert audio if not wav
        if audio_ext != "wav":
            sound = AudioSegment.from_file(audio_path)
            audio_path = "converted.wav"
            sound.export(audio_path, format="wav")

        incoming_msg = transcribe_audio(audio_path)
        os.remove(audio_path)

    detected_lang = detect_language(incoming_msg)
    translated_msg = translate_text(incoming_msg, target_language="en")

    print(f"User said: {incoming_msg}")
    print(f"Detected language: {detected_lang}")
    print(f"Translated: {translated_msg}")

    reply = generate_response(translated_msg)

    MAX_LENGTH = 1500  # WhatsApp message limit
    if len(reply) > MAX_LENGTH:
        reply = reply[:MAX_LENGTH] + "..."

    print("Reply to be sent to WhatsApp:", reply)
    if detected_lang in SUPPORTED_LANGUAGES and detected_lang != "en":
        reply = back_translate_text(reply, detected_lang)

    print(f"Reply: {reply}")

    twilio_response = MessagingResponse()
    twilio_response.message(reply)
    return str(twilio_response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
