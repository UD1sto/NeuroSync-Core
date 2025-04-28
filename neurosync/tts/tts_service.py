import os
import requests
from enum import Enum
from typing import Optional, Dict, Any, Union, Tuple, List
import base64
import json

# Corrected imports
from .local_tts import call_local_tts
try:
    from ..api_connect.multi_part_return import get_tts_with_blendshapes
except ImportError:
    # Provide a fallback implementation if the module is not available
    def get_tts_with_blendshapes(text, voice=None):
        print("Warning: neurosync.api_connect module not found. Returning empty data.")
        return b"", []

class TTSProvider(Enum):
    LOCAL = "local"
    ELEVENLABS = "elevenlabs"
    NEUROSYNC = "neurosync"  # Combined TTS + blendshapes endpoint

class TTSService:
    """Unified interface for TTS services that can switch between local TTS and ElevenLabs"""

    def __init__(self, provider: Union[str, TTSProvider] = None):
        """
        Initialize the TTS service

        Args:
            provider: The TTS provider to use (defaults to value from environment variable TTS_PROVIDER)
        """
        # Get provider from environment if not specified
        if provider is None:
            provider = os.getenv("TTS_PROVIDER", TTSProvider.ELEVENLABS.value)

        # Convert string to enum if needed
        if isinstance(provider, str):
            provider = TTSProvider(provider)

        self.provider = provider

        # Load provider-specific configuration (uses os.getenv directly)
        if self.provider == TTSProvider.ELEVENLABS:
            self.api_key = os.getenv("ELEVENLABS_API_KEY")
            self.voice_id = os.getenv("ELEVENLABS_VOICE_ID")
            self.model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1")
            self.api_url = "https://api.elevenlabs.io/v1/text-to-speech"
        elif self.provider == TTSProvider.LOCAL:
            self.endpoint = os.getenv("LOCAL_TTS_URL", "http://127.0.0.1:5000/tts")
            self.voice = os.getenv("LOCAL_TTS_VOICE")
        elif self.provider == TTSProvider.NEUROSYNC:
            self.endpoint = os.getenv("NEUROSYNC_TTS_URL", "http://127.0.0.1:5000/text_to_blendshapes")
            self.voice = os.getenv("NEUROSYNC_TTS_VOICE")

    def generate_speech(self, text: str) -> bytes:
        """
        Generate speech from text

        Args:
            text: The text to convert to speech

        Returns:
            The audio data as bytes
        """
        if self.provider == TTSProvider.ELEVENLABS:
            return self._elevenlabs_tts(text)
        elif self.provider == TTSProvider.LOCAL:
            return self._local_tts(text)
        elif self.provider == TTSProvider.NEUROSYNC:
            audio, _ = self._neurosync_tts_with_blendshapes(text)
            return audio
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_speech_with_blendshapes(self, text: str) -> Tuple[bytes, List[List[float]]]:
        """
        Generate speech and corresponding blendshapes from text

        Args:
            text: The text to convert to speech

        Returns:
            Tuple of (audio_bytes, blendshapes_data)
        """
        if self.provider == TTSProvider.NEUROSYNC:
            return self._neurosync_tts_with_blendshapes(text)
        else:
            # For other providers, we first generate audio, then get blendshapes separately
            audio = self.generate_speech(text)
            blendshapes = self._get_blendshapes_for_audio(audio)
            return audio, blendshapes

    def _elevenlabs_tts(self, text: str) -> bytes:
        """Generate speech using ElevenLabs API"""
        try:
            # Verify we have necessary configuration
            if not self.api_key:
                print("Error: ElevenLabs API key not set")
                return b""

            if not self.voice_id:
                print("Error: ElevenLabs voice ID not set")
                return b""

            # Construct URL and headers
            url = f"{self.api_url}/{self.voice_id}"

            headers = {
                # Request WAV so we can play without ffmpeg
                "Accept": "audio/wav",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }

            # Prepare the request payload
            payload = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }

            # Log request details for debugging
            print(f"Sending request to ElevenLabs: {url}")
            print(f"Using voice ID: {self.voice_id}")
            print(f"Using model ID: {self.model_id}")

            # Make the API request
            response = requests.post(url, json=payload, headers=headers)

            # Check for errors
            if response.status_code != 200:
                print(f"ElevenLabs API error: {response.status_code} {response.reason}")
                print(f"Response content: {response.text}")
                return b""

            # If everything is OK, return the audio content
            return response.content
        except Exception as e:
            print(f"ElevenLabs API error: {e}")
            return b""  # Return empty bytes on error

    def _local_tts(self, text: str) -> bytes:
        """Generate speech using local TTS endpoint"""
        try:
            return call_local_tts(text, self.voice)
        except Exception as e:
            print(f"Local TTS error: {e}")
            return b""  # Return empty bytes on error

    def _neurosync_tts_with_blendshapes(self, text: str) -> Tuple[bytes, List[List[float]]]:
        """Generate speech and blendshapes using Neurosync combined endpoint"""
        try:
            return get_tts_with_blendshapes(text, self.voice)
        except Exception as e:
            print(f"Neurosync TTS error: {e}")
            return b"", []  # Return empty data on error

    def _get_blendshapes_for_audio(self, audio_bytes: bytes) -> List[List[float]]:
        """Get blendshapes for the given audio using Neurosync API"""
        try:
            # If no audio data, return empty list
            if not audio_bytes:
                print("No audio data to generate blendshapes from")
                return []

            # Use the audio_to_blendshapes endpoint from neurosync_local_api
            url = os.getenv("NEUROSYNC_BLENDSHAPES_URL", "http://127.0.0.1:5000/audio_to_blendshapes")

            # Check if API is running before making the request
            try:
                # Simple connection test with timeout
                test_response = requests.get(
                    url.rsplit('/', 1)[0],  # Base URL without endpoint
                    timeout=1.0  # Short timeout
                )
                # If we can't connect, don't try the full request
                if test_response.status_code >= 400:
                    print(f"Warning: Blendshapes API returned status {test_response.status_code}. Skipping blendshape generation.")
                    return []
            except requests.exceptions.RequestException:
                print(f"Warning: Blendshapes API not available at {url}. Skipping blendshape generation.")
                print("If you want facial animation, make sure the NeuroSync API is running.")
                return []

            # For JSON API, encode the audio as base64
            payload = {
                "audio": base64.b64encode(audio_bytes).decode("utf-8")
            }

            headers = {
                "Content-Type": "application/json"
            }

            # Make the API request with a timeout
            print(f"Sending request to get blendshapes: {url}")
            response = requests.post(url, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()

            # Parse the response
            result = response.json()

            if "blendshapes" in result:
                return result["blendshapes"]
            else:
                print("No blendshapes in response")
                return []

        except Exception as e:
            print(f"Error getting blendshapes: {e}")
            # Provide more helpful message based on error type
            if isinstance(e, requests.exceptions.ConnectionError):
                print("Connection error: Make sure the NeuroSync API server is running at the configured URL.")
                print("You can still use speech without animation, or run neurosync_local_api.py to start the API.")
            elif isinstance(e, requests.exceptions.Timeout):
                print("Request timed out: The blendshape generation process may be taking too long.")

            return [] 