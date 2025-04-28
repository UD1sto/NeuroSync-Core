#!/usr/bin/env python3
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import sys
import argparse
import queue
import threading
import time
import sounddevice as sd
import numpy as np
from scipy.io import wavfile as wav
from io import BytesIO
import dotenv
import requests
import simpleaudio as sa

# Load environment variables
dotenv.load_dotenv()

# Import our service interfaces using relative imports
from ..llm.llm_service import LLMService, LLMProvider
from ..tts.tts_service import TTSService, TTSProvider
from ..core.bridge import BridgeCache

# Optional: import for animation if we want to use it
animation_available = False
try:
    from ..core.livelink.livelink_init import create_socket_connection, initialize_py_face
    from ..core.livelink.pylivelinkface import FaceBlendShape
    from ..core.runtime.player import Player
    animation_available = True
    print("Animation components loaded successfully!")
except ImportError as e:
    print(f"Animation components not available: {e}")
    print("Will run without animation.")

# Global variables for streaming
audio_queue = queue.Queue()
blendshape_queue = queue.Queue()
is_streaming = threading.Event()
audio_start   = None       # <-- new global
audio_sr      = 16000      # will be updated by the first chunk
audio_start_lck = threading.Lock()

def process_llm_chunk(chunk, tts_chunk_queue, text_buffer, buffer_threshold=10):
    """Process a chunk of text from the LLM and add it to the TTS queue when ready"""
    text_buffer.append(chunk)
    
    # Join buffer to check if we have enough text
    full_text = ''.join(text_buffer)
    
    # Check if we have a complete sentence or enough words
    sentence_endings = ['.', '!', '?', '\n']
    has_ending = any(ending in full_text for ending in sentence_endings)
    word_count = len(full_text.split())
    
    if has_ending or word_count >= buffer_threshold:
        # Find the last sentence ending
        last_idx = max([full_text.rfind(ending) for ending in sentence_endings if full_text.rfind(ending) != -1], default=-1)
        
        if last_idx != -1:
            # Get the text up to and including the sentence ending
            sentence = full_text[:last_idx+1]
            # Keep the rest in the buffer
            text_buffer.clear()
            text_buffer.append(full_text[last_idx+1:])
            
            # Add the complete sentence to the TTS queue
            tts_chunk_queue.put(sentence)
            return True
        elif word_count >= buffer_threshold:
            # If no sentence ending but enough words, process the whole buffer
            tts_chunk_queue.put(full_text)
            text_buffer.clear()
            return True
    
    return False

def stream_audio_player():
    global audio_start, audio_sr
    import io, soundfile as sf
    print("Audio streaming worker started")

    out_stream = None               # reuse a single OutputStream
    while True:
        wav_bytes = audio_queue.get()
        if wav_bytes is None:
            break

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype='int16')
        if out_stream is None:
            out_stream = sd.OutputStream(samplerate=sr,
                                         channels=1,
                                         dtype='int16')
            out_stream.start()

        # mark stream start only once
        with audio_start_lck:
            if audio_start is None:
                audio_start = time.monotonic()
                audio_sr    = sr

        out_stream.write(data)      # non-blocking
        audio_queue.task_done()

def stream_blendshape_player(py_face, sock):
    global audio_start
    frame_dur = 1/60.0
    frame_idx = 0

    # block until we know when audio really began
    while audio_start is None:
        time.sleep(0.001)

    while True:
        frame = blendshape_queue.get()
        if frame is None:
            break

        target = audio_start + frame_idx * frame_dur
        sleep  = target - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)

        # send frame … (set each blendshape and transmit over UDP)
        try:
            for i in range(min(len(frame), 61)):  # Ensure we don't exceed the supported blendshape count
                py_face.set_blendshape(FaceBlendShape(i), float(frame[i]))
            sock.sendall(py_face.encode())
        except Exception as e:
            print(f"Failed to send blendshape frame: {e}")

        frame_idx += 1
        blendshape_queue.task_done()

def tts_worker(tts_chunk_queue, tts_service, py_face=None, socket_connection=None):
    """Worker that processes text chunks and converts them to speech in real-time"""
    
    # Start audio streaming worker if it's supported
    audio_thread = None
    try:
        audio_thread = threading.Thread(target=stream_audio_player, daemon=True)
        audio_thread.start()
    except Exception as e:
        print(f"Failed to start audio streaming: {e}")
    
    # We no longer manage local per-frame blendshape streaming from the client.
    blendshape_thread = None
    
    while True:
        try:
            # Get text chunk from queue
            text_chunk = tts_chunk_queue.get()
            
            # None is the signal to stop
            if text_chunk is None:
                # Signal streaming threads to stop
                audio_queue.put(None)
                break
            
            print(f"\nConverting to speech: '{text_chunk}'")
            
            # Generate speech
            audio_bytes = tts_service.generate_speech(text_chunk)
            
            # Forward audio to NeuroSync local API which will handle BOTH playback and animation.
            # Only if that fails will we fall back to local playback (no animation).
            if audio_bytes:
                forwarded = False
                url = os.getenv("NEUROSYNC_BLENDSHAPES_URL", "http://127.0.0.1:5000/audio_to_blendshapes")
                try:
                    import base64
                    payload = {"audio": base64.b64encode(audio_bytes).decode("utf-8")}
                    headers = {"Content-Type": "application/json"}
                    requests.post(url, json=payload, headers=headers, timeout=10.0)
                    forwarded = True
                except Exception as e:
                    print(f"Could not forward audio to NeuroSync API (falling back to local playback): {e}")
                
                # Fallback to local playback only when forwarding failed
                if not forwarded:
                    audio_queue.put(audio_bytes)
            
        except Exception as e:
            print(f"Error in TTS worker: {e}")
        finally:
            # Mark task as done
            tts_chunk_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="NeuroSync Client for LLM and TTS")
    parser.add_argument("--llm", choices=["openai", "llama3_1", "llama3_2"], 
                      help="LLM provider to use")
    parser.add_argument("--tts", choices=["elevenlabs", "local", "neurosync"], 
                      help="TTS provider to use")
    parser.add_argument("--no-animation", action="store_true", 
                      help="Disable animation even if available")
    args = parser.parse_args()
    
    # Initialize LLM service
    llm_provider = args.llm if args.llm else None
    llm_service = LLMService(llm_provider)
    print(f"Initialized LLM service with provider: {llm_service.provider.value}")
    
    # Initialize TTS service
    tts_provider = args.tts if args.tts else None
    tts_service = TTSService(tts_provider)
    print(f"Initialized TTS service with provider: {tts_service.provider.value}")
    
    # Initialize animation if available and not disabled
    py_face = None
    socket_connection = None
    if animation_available and not args.no_animation:
        try:
            py_face = initialize_py_face()
            socket_connection = create_socket_connection()
            print("Animation initialized successfully")
        except Exception as e:
            print(f"Failed to initialize animation: {e}")
            py_face = None
            socket_connection = None
    
    # Create a queue for TTS chunks
    tts_chunk_queue = queue.Queue()
    
    # Start TTS worker thread
    tts_thread = threading.Thread(
        target=tts_worker, 
        args=(tts_chunk_queue, tts_service, py_face, socket_connection),
        daemon=True
    )
    tts_thread.start()
    
    print("\nNeuroSync Client initialized. Enter a prompt or 'exit' to quit.")
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
                
            print("\nGenerating response...\n")
            
            # Format message for LLM (include optional System-2 bridge context)
            messages = []
            bridge_txt = BridgeCache.read()
            if bridge_txt:
                messages.append({"role": "system", "content": bridge_txt})
            messages.append({"role": "user", "content": user_input})
            
            # Process in streaming mode
            text_buffer = []
            print("AI: ", end="", flush=True)
            
            try:
                # Stream response from LLM
                for chunk in llm_service.generate_stream(
                    messages,
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=0.9
                ):
                    print(chunk, end="", flush=True)
                    
                    # Process chunk for TTS
                    process_llm_chunk(chunk, tts_chunk_queue, text_buffer)
            except Exception as e:
                print(f"\nError generating response: {e}")
                continue
                
            # Process any remaining text in buffer
            if text_buffer:
                final_text = ''.join(text_buffer)
                if final_text.strip():
                    tts_chunk_queue.put(final_text)
                    
            print("\n")  # Add some space after the response
            
    except KeyboardInterrupt:
        print("\nExiting...")
    
    # Cleanup
    tts_chunk_queue.put(None)  # Signal TTS worker to stop
    tts_thread.join(timeout=5)

if __name__ == "__main__":
    main() 