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
import numpy as np
from scipy.io import wavfile as wav
from io import BytesIO
import dotenv
import requests
import hashlib
import secrets

# Load environment variables
dotenv.load_dotenv()

# Import our service interfaces using relative imports
from ..llm.llm_service import LLMService, LLMProvider
from ..tts.tts_service import TTSService, TTSProvider
from ..core.bridge import BridgeCache
from ..core.scb_store import scb_store
from neurosync.core.color_text import ColorText

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
blendshape_queue = queue.Queue()
is_streaming = threading.Event()
audio_start   = None       # <-- new global
audio_sr      = 16000      # will be updated by the first chunk
audio_start_lck = threading.Lock()

# Add after imports
# --- SCB Remote Configuration ---
SCB_API_BASE_URL = os.getenv("SCB_API_BASE_URL") or os.getenv("NEUROSYNC_URL", "http://127.0.0.1:5000") + "/scb"
SCB_API_KEY = os.getenv("NEUROSYNC_API_KEY", "")

def scb_remote_append(entry: dict):
    """Send an SCB entry to the remote NeuroSync server if configured."""
    try:
        headers = {"Content-Type": "application/json"}
        if SCB_API_KEY:
            headers["X-NeuroSync-Key"] = SCB_API_KEY

        route = "/directive" if entry.get("type") == "directive" else "/event"
        url = f"{SCB_API_BASE_URL}{route}"
        requests.post(url, json=entry, headers=headers, timeout=3)
    except Exception as e:
        print(f"[SCB Remote] Failed to send entry to {url}: {e}")

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
    
    while True:
        try:
            # Get text chunk from queue
            text_chunk = tts_chunk_queue.get()
            
            # None is the signal to stop
            if text_chunk is None:
                break
            
            print(f"\nConverting to speech: '{text_chunk}'")
            
            # Generate speech
            audio_bytes = tts_service.generate_speech(text_chunk)
            
            # Forward audio to NeuroSync local API which will handle BOTH playback (RTMP) and animation.
            # No fallback to local playback.
            if audio_bytes:
                forwarded = False
                url = os.getenv("NEUROSYNC_BLENDSHAPES_URL", "http://127.0.0.1:5000/audio_to_blendshapes")
                try:
                    import base64
                    # Send as raw bytes if server expects that, or base64 if it expects JSON
                    # Current server /audio_to_blendshapes expects raw bytes with octet-stream or wav
                    headers = {"Content-Type": "audio/wav"} # Or application/octet-stream
                    # Use requests.post(url, data=audio_bytes, headers=headers, timeout=10.0)
                    # OR if server expects JSON/base64:
                    # payload = {"audio": base64.b64encode(audio_bytes).decode("utf-8")}
                    # headers = {"Content-Type": "application/json"}
                    # requests.post(url, json=payload, headers=headers, timeout=10.0)
                    
                    # Assuming server /audio_to_blendshapes expects raw bytes based on previous adjustments
                    requests.post(url, data=audio_bytes, headers=headers, timeout=10.0)

                    forwarded = True
                except Exception as e:
                    print(f"Could not forward audio to NeuroSync API ({url}): {e}")
                
                # Removed fallback to local playback via audio_queue
            
        except Exception as e:
            print(f"Error in TTS worker: {e}")
        finally:
            # Mark task as done
            tts_chunk_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="NeuroSync Client for LLM and TTS")
    parser.add_argument("--llm", choices=["openai", "llama3_1", "llama3_2"], 
                      help="LLM provider to use")
    parser.add_argument("--tts", choices=["elevenlabs", "local", "neurosync", "neurosync_stream"], 
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
    
    # Decide execution mode based on TTS provider
    streaming_mode = tts_service.provider.value == "neurosync_stream"

    if not streaming_mode:
        # Start classic TTS worker that converts queued text chunks to speech
        tts_thread = threading.Thread(
            target=tts_worker, 
            args=(tts_chunk_queue, tts_service, py_face, socket_connection),
            daemon=True
        )
        tts_thread.start()
    else:
        tts_thread = None  # Will launch streaming threads per prompt
    
    print("\nNeuroSync Client initialized. Enter a prompt or 'exit' to quit.")
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
                
            print("\nGenerating response...\n")
            
            if not streaming_mode:
                # ------- Classic LLM + local TTS pipeline --------

                persona = "You are Mai, a witty and helpful VTuber assistant with a dry sense of humor."
                summary = scb_store.get_summary()
                recent_chat = scb_store.get_recent_chat(3)
                bridge_txt = BridgeCache.read()

                system_parts = [persona]
                if bridge_txt:
                    system_parts.append(bridge_txt)
                if summary:
                    system_parts.append(f"Current summary:\n{summary}")
                if recent_chat:
                    system_parts.append(f"Recent chat:\n{recent_chat}")

                system_msg = "\n\n".join(system_parts)

                messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_input}]

                # Log user prompt to SCB (both local and remote)
                try:
                    scb_store.append_chat(user_input, actor="user")
                    scb_remote_append({"type": "event", "actor": "user", "text": user_input})
                except Exception as e:
                    print(f"{ColorText.RED}[Client] Failed to log user input to SCB: {e}{ColorText.END}")

                # Process in streaming mode (text)
                text_buffer = []
                full_reply_text = ""
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
                        full_reply_text += chunk
                        
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

                # After generation complete, log AI speech to SCB
                try:
                    speech_entry = {"type": "speech", "actor": "vtuber", "text": full_reply_text}
                    scb_store.append(speech_entry)
                    scb_remote_append(speech_entry)
                except Exception as e:
                    print(f"{ColorText.RED}[Client] Failed to log AI speech to SCB: {e}{ColorText.END}")

            else:
                # ------- Remote streaming pipeline --------

                if animation_available and not args.no_animation and not any(t.name == "BlendshapeStreamThread" for t in threading.enumerate()):
                    blendshape_thread = threading.Thread(
                        target=stream_blendshape_player,
                        args=(py_face, socket_connection),
                        daemon=True,
                        name="BlendshapeStreamThread"
                    )
                    blendshape_thread.start()

                print("AI (speech only – streaming):")

                try:
                    for audio_bytes, blendshapes in tts_service.stream_speech_and_blendshapes(user_input):
                        if blendshapes:
                            for frame in blendshapes:
                                blendshape_queue.put(frame)
                except Exception as e:
                    print(f"Streaming TTS error: {e}")

                # Signal end of streams
                blendshape_queue.put(None)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    
    # Cleanup
    if not streaming_mode:
        tts_chunk_queue.put(None)  # Signal TTS worker to stop
        tts_thread.join(timeout=5)
    else:
        # Signal blendshape thread to stop if running
        blendshape_queue.put(None)

# -----------------------------------------------------------------------------
# Lightweight job handler for in-process BYOC adapter integration
# -----------------------------------------------------------------------------

def accept_vtuber_job(payload: dict) -> str:
    """Handle a VTuber job submitted by the BYOC worker (server_adapter).

    For now this is a placeholder that simply logs the payload and returns a
    pseudo-random mock job hash so the front-end can display a meaningful ID.

    This keeps everything in-process (no HTTP hop) while the full NeuroSync
    pipeline integration is still under construction.
    """
    print("\n[NeuroSync-Core] accept_vtuber_job called with payload:")
    try:
        import json as _json
        print(_json.dumps(payload, indent=2)[:1000])  # Truncate to avoid log spam
    except Exception:
        print(str(payload))

    mock_hash = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
    print(f"[NeuroSync-Core] Returning mock job hash: {mock_hash}\n")
    return mock_hash

if __name__ == "__main__":
    main() 