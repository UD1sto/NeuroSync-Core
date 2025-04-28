import io
import json
import requests
import os # Added

# TODO: Consider moving this to config or a dedicated voices file
voices = {
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "Laura": "FGY2WhTYpPnrIDTdsKH5",
    "Charlie": "IKne3meq5aSn9XLyUdCD",
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Callum": "N2lVS1w4EtoT3dr4eOWO",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
    "Charlotte": "XB0fDUnXU5powFXDhCwa",
    "Alice": "Xb7hH8MSUJpSbSDYk0k2",
    "Matilda": "XrExE9yKIg1WjnnlVkGX",
    "Will": "bIHbv24MWmeRgasZH58o",
    "Jessica": "cgSgspJ2msm6clMCkdW9",
    "Eric": "cjVigY5qzO86Huf0OWal",
    "Brian": "nPczCjzI2devNBz1zQrb",
    "Daniel": "onwK4e9ZLuTAKqWW03F9",
    "Lily": "pFZP5JQG7iQjIQuC4Bku",
    "Bill": "pqHfZKP75CvOlQylNhV4",
}

XI_API_KEY = os.getenv("ELEVENLABS_API_KEY") # Read from environment

def get_voice_id_by_name(name):
    return voices.get(name)

def get_elevenlabs_audio(text, name):
    VOICE_ID = get_voice_id_by_name(name) or os.getenv("ELEVENLABS_VOICE_ID") # Fallback to env var
    if VOICE_ID is None:
        print("Error: ElevenLabs voice ID not found or specified in environment.")
        return b""

    if not XI_API_KEY:
        print("Error: ElevenLabs API key not set in environment (ELEVENLABS_API_KEY).")
        return b""

    API_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

    headers = {
        "xi-api-key": XI_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1"), # Read from env
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.5,
            "use_speaker_boost": True
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        audio_data = response.content
        return audio_data
    except requests.exceptions.RequestException as e:
        print(f"[ElevenLabs Client] Error calling API: {e}")
        return b""

def get_speech_to_speech_audio(audio_bytes, name):
    VOICE_ID = get_voice_id_by_name(name) or os.getenv("ELEVENLABS_VOICE_ID") # Fallback to env var
    if VOICE_ID is None:
        print("Error: ElevenLabs voice ID not found or specified in environment.")
        return b""

    if not XI_API_KEY:
        print("Error: ElevenLabs API key not set in environment (ELEVENLABS_API_KEY).")
        return b""

    STS_API_URL = f"https://api.elevenlabs.io/v1/speech-to-speech/{VOICE_ID}/stream"

    headers = {
        "Accept": "application/json",
        "xi-api-key": XI_API_KEY
    }

    data = {
        "model_id": "eleven_english_sts_v2",
        "voice_settings": json.dumps({
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.5,
            "use_speaker_boost": True
        })
    }

    files = {
        "audio": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")
    }

    try:
        response = requests.post(STS_API_URL, headers=headers, data=data, files=files)
        response.raise_for_status()  # Raise an error for bad responses
        audio_data = response.content
        return audio_data
    except requests.exceptions.RequestException as e:
        print(f"[ElevenLabs STS Client] Error calling API: {e}")
        return b"" 