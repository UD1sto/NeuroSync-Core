# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import requests
# Corrected config import
from ..core.config import USE_OPENAI_EMBEDDING, EMBEDDING_LOCAL_SERVER_URL, EMBEDDING_OPENAI_MODEL, LOCAL_EMBEDDING_SIZE, OPENAI_EMBEDDING_SIZE

def get_embedding(text: str, use_openai: bool = USE_OPENAI_EMBEDDING, openai_api_key: str = None, local_server_url: str = EMBEDDING_LOCAL_SERVER_URL) -> list:
    if use_openai:
        return get_openai_embedding(text, openai_api_key)
    else:
        return get_local_embedding(text, local_server_url)

def get_local_embedding(text: str, local_server_url: str) -> list:
    """ Retrieves embedding from a local server endpoint. """
    try:
        if not local_server_url:
             raise ValueError("Local embedding server URL not configured.")
        payload = {"text": text}
        response = requests.post(local_server_url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if embedding is None:
            raise ValueError("No 'embedding' key in the local server response.")
        if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
             raise ValueError("Invalid embedding format received from local server.")
        # Optional: Validate embedding length
        # if len(embedding) != LOCAL_EMBEDDING_SIZE:
        #     print(f"Warning: Local embedding size mismatch. Expected {LOCAL_EMBEDDING_SIZE}, got {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"Error getting embedding from local provider ({local_server_url}): {e}")
        # Return None or raise exception based on desired error handling
        return None
        # return [0.0] * LOCAL_EMBEDDING_SIZE # Fallback to zero vector (less informative)

def get_openai_embedding(text: str, openai_api_key: str = None) -> list:
    """ Retrieves embedding from OpenAI API. """
    try:
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not provided or found in environment.")

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {"input": text, "model": EMBEDDING_OPENAI_MODEL}
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "data" in data and len(data["data"]) > 0:
            embedding = data["data"][0].get("embedding")
            if embedding is None:
                raise ValueError("No 'embedding' key in the OpenAI response data.")
            if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                 raise ValueError("Invalid embedding format received from OpenAI.")
            # Optional: Validate embedding length
            # if len(embedding) != OPENAI_EMBEDDING_SIZE:
            #     print(f"Warning: OpenAI embedding size mismatch. Expected {OPENAI_EMBEDDING_SIZE}, got {len(embedding)}")
            return embedding
        else:
            raise ValueError("Invalid response structure from OpenAI embeddings API.")
    except Exception as e:
        print(f"Error getting embedding from OpenAI provider: {e}")
        # Return None or raise exception based on desired error handling
        return None
        # return [0.0] * OPENAI_EMBEDDING_SIZE # Fallback to zero vector 