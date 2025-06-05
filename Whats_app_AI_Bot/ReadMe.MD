# 🧠 WhatsApp AI Bot with Twilio, Whisper, Ollama & Translation

This is a production-ready AI-powered WhatsApp bot built using:

- 📞 [Twilio](https://www.twilio.com/) for WhatsApp messaging
- 🎤 [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) for speech-to-text transcription
- 🌐 [Google Translate API](https://cloud.google.com/translate/docs) for multi-language translation
- 💬 Either [OpenAI GPT](https://openai.com/) or a local LLM via [Ollama](https://ollama.com/)
- 🐳 Dockerized for easy deployment

---

## 🌍 Supported Languages

- 🇫🇷 French (`fr`)
- 🇮🇳 Hindi (`hi`)
- 🇧🇩 Bengali (`bn`)
- 🇲🇺 Mauritian Creole (`mfe`)

---

## 🔧 Features

- Voice-to-text transcription via Whisper
- Auto-detects the user's language
- Translates messages to English before sending to the AI
- Gets AI response using local LLM (`llama2`, `mistral`, etc.) or OpenAI
- Translates response back to the user's language
- Deployable via Docker

---

## Prerequisites

### 1. Create a Twilio Account with an API Key
### 2. Create an OpenAI account with and API Key
### 3. Register a phonenumber on WhatsApp
### 4. Create an NGROK account and generate an auth token



## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/laldeomaroam/ML_Projects/Whats_app_AI_Bot.git
cd Whats_app_AI_Bot
```

### 2. Install [Ngrok](https://ngrok.com/downloads/)

### 3. Install [Docker](https://www.docker.com/)

### 4. 
