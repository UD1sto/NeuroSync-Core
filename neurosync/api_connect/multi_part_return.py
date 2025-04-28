# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import json
import requests
import os
import base64
from typing import Tuple, List, Optional

# Default URL, can be overridden by environment variable
DEFAULT_NEUROSYNC_TTS_URL = "http://127.0.0.1:5000/text_to_blendshapes"

def parse_json_response(response) -> Tuple[Optional[bytes], Optional[List[List[float]]]]:
    """
    Parses a JSON response expected to contain base64 audio and blendshapes list.
    """
    try:
        data = response.json()
        audio_b64 = data.get("audio")
        blendshapes = data.get("blendshapes")

        audio_bytes = None
        if audio_b64:
             try:
                 audio_bytes = base64.b64decode(audio_b64)
             except Exception as e:
                 print(f"Error decoding base64 audio: {e}")
                 # Keep blendshapes if they exist, even if audio fails

        if not isinstance(blendshapes, list):
             print(f"Warning: Blendshapes data is not a list: {type(blendshapes)}")
             # Decide whether to return None or empty list for blendshapes
             # blendshapes = []

        # Only return None, None if both are missing or failed
        if audio_bytes is None and blendshapes is None:
             print("Neither audio nor blendshapes found in JSON response.")
             return None, None

        return audio_bytes, blendshapes

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None, None
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return None, None


def get_tts_with_blendshapes(text, voice=None) -> Tuple[Optional[bytes], Optional[List[List[float]]]]:
    """
    Calls the combined TTS+Blendshapes endpoint (expected to return JSON).
    Returns a tuple: (audio_bytes, blendshapes) if successful, else (None, None).
    """
    endpoint = os.getenv("NEUROSYNC_TTS_URL", DEFAULT_NEUROSYNC_TTS_URL)

    payload = {"prompt": text}
    if voice is not None:
        # Note: Check if the target API actually uses the 'voice' parameter
        payload["voice"] = voice

    try:
        print(f"Calling TTS+Blendshapes endpoint: {endpoint}")
        response = requests.post(endpoint, json=payload, timeout=30.0) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Assuming the response is JSON as per neurosync/server/app.py
        return parse_json_response(response)

    except requests.exceptions.RequestException as e:
        print(f"Error calling TTS endpoint {endpoint}: {e}")
        return None, None
    except Exception as e:
        # Catch other potential errors during parsing or processing
        print(f"Unexpected error in get_tts_with_blendshapes: {e}")
        return None, None 