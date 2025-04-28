# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import requests
import json
import os # Added
from typing import Optional, List
import base64 # Added

# Configuration via environment variables
NEUROSYNC_API_KEY = os.getenv("NEUROSYNC_API_KEY")
NEUROSYNC_REMOTE_URL = os.getenv("NEUROSYNC_REMOTE_URL") # For potential future remote API
NEUROSYNC_LOCAL_URL = os.getenv("NEUROSYNC_BLENDSHAPES_URL", "http://127.0.0.1:5000/audio_to_blendshapes") # Default local endpoint

def send_audio_to_neurosync(audio_bytes: bytes, use_local: bool = True) -> Optional[List[List[float]]]:
    """ Sends audio bytes to the NeuroSync API (local or remote) and returns blendshapes. """
    if not validate_audio_bytes(audio_bytes):
         print("Error: Invalid audio bytes provided.")
         return None

    try:
        # Determine URL and headers based on local/remote flag
        url = NEUROSYNC_LOCAL_URL if use_local else NEUROSYNC_REMOTE_URL
        if not url:
            print(f"Error: NeuroSync URL not configured ({'local' if use_local else 'remote'}). Set NEUROSYNC_{'LOCAL' if use_local else 'REMOTE'}_URL.")
            return None

        headers = {"Content-Type": "application/json"} # Assume JSON input now
        if not use_local:
            if not NEUROSYNC_API_KEY:
                 print("Warning: NEUROSYNC_API_KEY not set for remote API call.")
                 # Decide if call should proceed or fail
            else:
                 headers["Authorization"] = f"Bearer {NEUROSYNC_API_KEY}" # Example using Bearer token

        # Encode audio as base64 for JSON payload
        payload = {
            "audio": base64.b64encode(audio_bytes).decode("utf-8")
        }

        print(f"Sending audio to NeuroSync API: {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=20.0) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses

        # Parse the JSON response which should contain blendshapes
        return parse_blendshapes_from_json(response.json())

    except requests.exceptions.RequestException as e:
        print(f"Error sending audio to NeuroSync API ({url}): {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response from NeuroSync API: {e}")
        return None
    except Exception as e:
         print(f"Unexpected error in send_audio_to_neurosync: {e}")
         return None

def validate_audio_bytes(audio_bytes: Optional[bytes]) -> bool:
    """ Basic validation for audio bytes. """
    return audio_bytes is not None and len(audio_bytes) > 0

def parse_blendshapes_from_json(json_response: dict) -> Optional[List[List[float]]]:
    """ Extracts and validates blendshapes list from a JSON dictionary. """
    if not isinstance(json_response, dict):
         print("Error: Invalid JSON response format (expected dict).")
         return None

    blendshapes = json_response.get("blendshapes")

    if blendshapes is None:
        print("Error: 'blendshapes' key not found in JSON response.")
        return None

    if not isinstance(blendshapes, list):
        print(f"Error: 'blendshapes' data is not a list (type: {type(blendshapes)}).")
        return None

    # Optional: Add validation for frame structure (list of lists of floats)
    facial_data = []
    try:
        for frame in blendshapes:
            if isinstance(frame, list):
                frame_data = [float(value) for value in frame]
                facial_data.append(frame_data)
            else:
                print(f"Warning: Skipping invalid frame data (not a list): {frame}")
    except (ValueError, TypeError) as e:
         print(f"Error converting blendshape frame data to float: {e}")
         return None # Or return partial data?

    return facial_data 