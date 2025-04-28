# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import requests
from threading import Thread
from queue import Queue
from openai import OpenAI

# Corrected import
from .sentence_builder import SentenceBuilder


def warm_up_llm_connection(config):
    """
    Perform a lightweight dummy request to warm up the LLM connection.
    This avoids the initial delay when the user sends the first real request.
    """
    print("Warming up LLM connection...")
    if config["USE_LOCAL_LLM"]:
        try:
            # For local LLM, use a dummy ping request with a short timeout.
            ping_url = config.get("LLM_STREAM_URL") or config.get("LLM_API_URL")
            if ping_url:
                requests.post(ping_url, json={"dummy": "ping"}, timeout=1)
                print("Local LLM connection warmed up.")
            else:
                print("Warning: No local LLM URL found for warm-up.")
        except Exception as e:
            print(f"Local LLM connection warm-up failed: {e}")
    else:
        try:
            # For OpenAI API, send a lightweight ping message.
            api_key = config.get("OPENAI_API_KEY")
            if not api_key:
                 print("Warning: OpenAI API key not provided. Cannot warm up connection.")
                 return
            client = OpenAI(api_key=api_key)
            client.chat.completions.create(
                model=config.get("OPENAI_MODEL", "gpt-4o"), # Use configured model or default
                messages=[{"role": "system", "content": "ping"}],
                max_tokens=1,
                stream=False
            )
            print("OpenAI API connection warmed up.")
        except Exception as e:
            print(f"OpenAI API connection warm-up failed: {e}")


def update_ui(token: str):
    """
    Immediately update the UI with the token.
    This version checks for newline characters and prints them so that
    line breaks and paragraphs are preserved.
    """
    # Replace Windows-style newlines with Unix-style
    token = token.replace('\r\n', '\n')
    # If the token contains newline(s), split and print accordingly.
    if '\n' in token:
        parts = token.split('\n')
        for i, part in enumerate(parts):
            print(part, end='', flush=True)
            if i < len(parts) - 1:
                print()
    else:
        print(token, end='', flush=True)


def build_llm_payload(user_input, chat_history, config):
    """
    Build the conversation messages and payload from the user input,
    chat history, and configuration.

    Returns:
        dict: The payload containing the messages and generation parameters.
    """
    system_message = config.get(
        "system_message",
        "You are Mai, speak naturally and like a human might with humour and dryness."
    )
    messages = [{"role": "system", "content": system_message}]
    # Ensure history entries are dictionaries with 'input' and 'response'
    for entry in chat_history:
        if isinstance(entry, dict) and "input" in entry and "response" in entry:
             messages.append({"role": "user", "content": entry["input"]})
             messages.append({"role": "assistant", "content": entry["response"]})
        else:
             print(f"Warning: Skipping invalid chat history entry: {entry}")

    messages.append({"role": "user", "content": user_input})

    payload = {
        "messages": messages,
        "max_new_tokens": config.get("max_new_tokens", 4000),
        "temperature": config.get("temperature", 1.0),
        "top_p": config.get("top_p", 0.9)
    }
    return payload


def local_llm_streaming(user_input, chat_history, chunk_queue, config):
    """
    Streams tokens from a local LLM using streaming.
    """
    payload = build_llm_payload(user_input, chat_history, config)
    full_response = ""
    max_chunk_length = config.get("max_chunk_length", 500)
    flush_token_count = config.get("flush_token_count", 10)

    # Create the SentenceBuilder and a dedicated token_queue.
    sentence_builder = SentenceBuilder(chunk_queue, max_chunk_length, flush_token_count)
    token_queue = Queue()
    sb_thread = Thread(target=sentence_builder.run, args=(token_queue,))
    sb_thread.daemon = True # Make daemon
    sb_thread.start()

    stream_url = config.get("LLM_STREAM_URL")
    if not stream_url:
        print("Error: LLM_STREAM_URL not configured for local LLM.")
        token_queue.put(None) # Ensure sentence builder finishes
        sb_thread.join()
        return "Error: Streaming URL not configured."

    try:
        session = requests.Session()
        with session.post(stream_url, json=payload, stream=True) as response:
            response.raise_for_status()
            print("\n\nAssistant Response (streaming - local):\n", flush=True)
            for token in response.iter_content(chunk_size=None, decode_unicode=True):
                if not token:
                    continue
                full_response += token
                update_ui(token)
                token_queue.put(token)
        session.close()

        token_queue.put(None)
        sb_thread.join()
        return full_response.strip()

    except Exception as e:
        print(f"\nError during streaming local LLM call ({stream_url}): {e}")
        token_queue.put(None) # Ensure sentence builder finishes on error
        sb_thread.join()
        return "Error: Streaming LLM call failed."


def local_llm_non_streaming(user_input, chat_history, chunk_queue, config):
    """
    Calls a local LLM non-streaming endpoint and processes the entire response.
    """
    payload = build_llm_payload(user_input, chat_history, config)
    full_response = ""
    max_chunk_length = config.get("max_chunk_length", 500)
    flush_token_count = config.get("flush_token_count", 10)

    # Set up the SentenceBuilder.
    sentence_builder = SentenceBuilder(chunk_queue, max_chunk_length, flush_token_count)
    token_queue = Queue()
    sb_thread = Thread(target=sentence_builder.run, args=(token_queue,))
    sb_thread.daemon = True # Make daemon
    sb_thread.start()

    api_url = config.get("LLM_API_URL")
    if not api_url:
        print("Error: LLM_API_URL not configured for local LLM.")
        token_queue.put(None)
        sb_thread.join()
        return "Error: API URL not configured."

    try:
        session = requests.Session()
        response = session.post(api_url, json=payload)
        session.close()
        response.raise_for_status() # Check for HTTP errors

        result = response.json()
        # Adjust based on actual local API response structure
        text = result.get('choices', [{}])[0].get('message', {}).get('content', "Error: No response content found.")

        print("Assistant Response (non-streaming - local):\n", flush=True)
        # Simulate token-by-token processing for UI and SentenceBuilder
        # Split more intelligently if needed (e.g., by sentence or word)
        # For now, just send the whole text as one chunk for simplicity
        update_ui(text)
        token_queue.put(text)
        full_response = text

        token_queue.put(None)
        sb_thread.join()
        return full_response.strip()

    except Exception as e:
        print(f"Error calling local LLM ({api_url}): {e}")
        token_queue.put(None)
        sb_thread.join()
        return "Error: Exception occurred calling local LLM."


def openai_llm_streaming(user_input, chat_history, chunk_queue, config):
    """
    Streams tokens from the OpenAI API.
    """
    payload = build_llm_payload(user_input, chat_history, config)
    full_response = ""
    max_chunk_length = config.get("max_chunk_length", 500)
    flush_token_count = config.get("flush_token_count", 10)

    # Set up the SentenceBuilder.
    sentence_builder = SentenceBuilder(chunk_queue, max_chunk_length, flush_token_count)
    token_queue = Queue()
    sb_thread = Thread(target=sentence_builder.run, args=(token_queue,))
    sb_thread.daemon = True # Make daemon
    sb_thread.start()

    api_key = config.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not configured.")
        token_queue.put(None)
        sb_thread.join()
        return "Error: OpenAI API key not configured."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=config.get("OPENAI_MODEL", "gpt-4o"),
            messages=payload["messages"],
            max_tokens=payload.get("max_new_tokens", 4000), # Use max_new_tokens from payload if exists
            temperature=payload.get("temperature", 1.0),
            top_p=payload.get("top_p", 0.9),
            stream=True
        )
        print("Assistant Response (streaming - OpenAI):\n", flush=True)
        for chunk in response:
            token = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if not token:
                continue
            full_response += token
            update_ui(token)
            token_queue.put(token)

        token_queue.put(None)
        sb_thread.join()
        return full_response.strip()

    except Exception as e:
        print(f"Error calling OpenAI API (streaming): {e}")
        token_queue.put(None)
        sb_thread.join()
        return "Error: OpenAI API streaming call failed."


def openai_llm_non_streaming(user_input, chat_history, chunk_queue, config):
    """
    Calls the OpenAI API without streaming.
    """
    payload = build_llm_payload(user_input, chat_history, config)
    full_response = ""
    max_chunk_length = config.get("max_chunk_length", 500)
    flush_token_count = config.get("flush_token_count", 10)

    # Set up the SentenceBuilder.
    sentence_builder = SentenceBuilder(chunk_queue, max_chunk_length, flush_token_count)
    token_queue = Queue()
    sb_thread = Thread(target=sentence_builder.run, args=(token_queue,))
    sb_thread.daemon = True # Make daemon
    sb_thread.start()

    api_key = config.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not configured.")
        token_queue.put(None)
        sb_thread.join()
        return "Error: OpenAI API key not configured."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=config.get("OPENAI_MODEL", "gpt-4o"),
            messages=payload["messages"],
            max_tokens=payload.get("max_new_tokens", 4000),
            temperature=payload.get("temperature", 1.0),
            top_p=payload.get("top_p", 0.9)
        )
        text = response.choices[0].message.content

        print("Assistant Response (non-streaming - OpenAI):\n", flush=True)
        # Simulate token processing for UI and SentenceBuilder
        update_ui(text)
        token_queue.put(text) # Send whole response as one chunk
        full_response = text

        token_queue.put(None)
        sb_thread.join()
        return full_response.strip()

    except Exception as e:
        print(f"Error calling OpenAI API (non-streaming): {e}")
        token_queue.put(None)
        sb_thread.join()
        return "Error: OpenAI API non-streaming call failed."


def stream_llm_chunks(user_input, chat_history, chunk_queue, config):
    """
    Dispatches the LLM call to the proper variant based on the configuration.
    """
    USE_LOCAL_LLM = config.get("USE_LOCAL_LLM", False)
    USE_STREAMING = config.get("USE_STREAMING", True)

    if USE_LOCAL_LLM:
        if USE_STREAMING:
            return local_llm_streaming(user_input, chat_history, chunk_queue, config)
        else:
            return local_llm_non_streaming(user_input, chat_history, chunk_queue, config)
    else:
        if USE_STREAMING:
            return openai_llm_streaming(user_input, chat_history, chunk_queue, config)
        else:
            return openai_llm_non_streaming(user_input, chat_history, chunk_queue, config) 