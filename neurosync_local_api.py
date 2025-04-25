# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

from flask import request, jsonify, Response, stream_with_context
import numpy as np
import torch
import flask
import json
import os
import threading
import queue
import time
import dotenv
from io import BytesIO
import base64 # Added for JSON serialization
import sounddevice as sd # Added for local audio playback
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextIteratorStreamer,
    VitsModel,
    utils
)
import scipy.io.wavfile as wav

from utils.generate_face_shapes import generate_facial_data_from_bytes
from utils.model.model import load_model
from utils.config import config
# Add imports for PyFace and socket connection
from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
from livelink.connect.pylivelinkface import FaceBlendShape # Added for real-time streaming
from livelink.animations.default_animation import default_animation_loop, stop_default_animation, default_animation_data # Added for blinking
from livelink.send_to_unreal import apply_blink_to_facial_data # Added for blinking
from utils.generated_runners import run_audio_animation # Kept for non-streaming routes

# Load environment variables
dotenv.load_dotenv()

app = flask.Flask(__name__)

# Check for CUDA availability and respect USE_CUDA environment variable
use_cuda_env = os.getenv("USE_CUDA", "auto").lower()
cuda_available = torch.cuda.is_available()

if use_cuda_env == "true":
    if cuda_available:
        device = torch.device('cuda')
        print("✅ Using CUDA GPU acceleration")
    else:
        print("⚠️ CUDA requested but not available! Make sure PyTorch is installed with CUDA support and GPU drivers are updated.")
        print("⚠️ Falling back to CPU mode")
        device = torch.device('cpu')
elif use_cuda_env == "false":
    device = torch.device('cpu')
    print("ℹ️ Using CPU mode (CUDA disabled by configuration)")
else:  # "auto" or any other value
    device = torch.device('cuda' if cuda_available else 'cpu')
    if cuda_available:
        print("✅ Using CUDA GPU acceleration (auto-detected)")
    else:
        print("ℹ️ CUDA not available, using CPU mode")

print("🔧 Activated device:", device)

model_path = 'utils/model/model.pth'
blendshape_model = load_model(model_path, config, device)

# Initialize PyFace and socket connection
py_face = initialize_py_face()
socket_connection = create_socket_connection()
default_animation_thread = threading.Thread(target=default_animation_loop, args=(py_face,))
default_animation_thread.daemon = True
default_animation_thread.start()

# Global queues for LLM-TTS pipeline
text_queue = queue.Queue()  # Queue for text chunks from LLM to TTS
audio_blendshape_queue = queue.Queue()  # Queue for (audio_bytes, blendshapes, chunk_id) for streaming
playback_audio_queue = queue.Queue() # Queue for raw audio_bytes for local playback

# LLM-TTS Configuration
class LLMTTSConfig:
    """Configuration class for LLM and TTS models"""
    
    def __init__(self):
        # LLM Configuration
        self.llm_model = os.getenv("LLM_MODEL", "distilgpt2")
        self.llm_max_length = int(os.getenv("LLM_MAX_LENGTH", "50"))
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.llm_top_p = float(os.getenv("LLM_TOP_P", "0.9"))
        self.llm_repetition_penalty = float(os.getenv("LLM_REPETITION_PENALTY", "1.0"))
        
        # TTS Configuration
        self.tts_model = os.getenv("TTS_MODEL", "facebook/mms-tts-eng")
        self.tts_speed = float(os.getenv("TTS_SPEED", "1.0"))
        
        # Pipeline Configuration
        self.chunk_word_threshold = int(os.getenv("CHUNK_WORD_THRESHOLD", "3"))
        
    def get_device(self):
        """Get the appropriate device"""
        return device

# Create LLM-TTS configuration
llm_tts_config = LLMTTSConfig()

# LLM and TTS models dictionary (lazy loaded)
pipeline_models = {}

# Color formatting for console output
class ColorText:
    """Simple class for colored text output"""
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def load_llm_tts_models():
    """Load LLM and TTS models if not already loaded"""
    if not pipeline_models:
        print(f"{ColorText.BOLD}[Pipeline]{ColorText.END} Loading LLM and TTS models...")
        
        # Enable progress bars for downloads
        utils.logging.set_verbosity_info()
        
        try:
            # Get TRUST_REMOTE_CODE value and log it
            trust_remote_code_value = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
            print(f"{ColorText.BOLD}[Config]{ColorText.END} TRUST_REMOTE_CODE set to: {trust_remote_code_value}")
            print(f"{ColorText.BOLD}[Config]{ColorText.END} LLM Model: {llm_tts_config.llm_model}")
            
            # Check for existing cache
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(cache_dir):
                print(f"{ColorText.BOLD}[Config]{ColorText.END} HuggingFace cache exists at: {cache_dir}")
            else:
                print(f"{ColorText.BOLD}[Config]{ColorText.END} HuggingFace cache does not exist yet - model will download")
            
            # Load LLM
            print(f"\n{ColorText.BOLD}[LLM]{ColorText.END} Loading LLM model: {llm_tts_config.llm_model}...")
            print(f"{ColorText.YELLOW}If downloading, progress will be displayed below:{ColorText.END}")
            
            # Explicitly enable downloading
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            os.environ["TRANSFORMERS_VERBOSITY"] = "info"
            
            # Print the configuration to verify
            print(f"{ColorText.BOLD}[Debug]{ColorText.END} trust_remote_code set to: {trust_remote_code_value}")
            
            # Load tokenizer with explicit trust_remote_code
            print(f"{ColorText.BOLD}[LLM]{ColorText.END} Loading tokenizer with trust_remote_code={trust_remote_code_value}...")
            llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_tts_config.llm_model,
                trust_remote_code=trust_remote_code_value
            )
            
            # Load model with explicit trust_remote_code
            print(f"{ColorText.BOLD}[LLM]{ColorText.END} Loading model with trust_remote_code={trust_remote_code_value}...")
            llm_model = AutoModelForCausalLM.from_pretrained(
                llm_tts_config.llm_model,
                trust_remote_code=trust_remote_code_value,
                device_map="auto"
            )
            
            # Set pad token to eos token if not set
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
            
            # Load TTS model
            print(f"\n{ColorText.BOLD}[TTS]{ColorText.END} Loading TTS model: {llm_tts_config.tts_model}...")
            print(f"{ColorText.YELLOW}If downloading, progress will be displayed below:{ColorText.END}")
            tts_tokenizer = AutoTokenizer.from_pretrained(llm_tts_config.tts_model)
            tts_model = VitsModel.from_pretrained(llm_tts_config.tts_model).to(device)
            
            # Get sample rate from model config
            sample_rate = tts_model.config.sampling_rate
            print(f"TTS sample rate: {sample_rate} Hz")
            
            # Store models in dictionary
            pipeline_models["llm_model"] = llm_model
            pipeline_models["llm_tokenizer"] = llm_tokenizer
            pipeline_models["tts_model"] = tts_model
            pipeline_models["tts_tokenizer"] = tts_tokenizer
            pipeline_models["sample_rate"] = sample_rate
            
            print(f"{ColorText.BOLD}[Pipeline]{ColorText.END} Models loaded successfully!")
        
        except Exception as e:
            import traceback
            print(f"{ColorText.RED}[ERROR] Failed to load models: {str(e)}{ColorText.END}")
            print(f"{ColorText.RED}[ERROR] Traceback: {traceback.format_exc()}{ColorText.END}")
            raise  # Re-raise to stop the application
    
    return pipeline_models

def llm_streaming_worker(model, tokenizer, prompt, device):
    """Worker function that streams text from LLM and sends it to the TTS queue"""
    print(f"\n{ColorText.BOLD}[LLM]{ColorText.END} Generating response...")
    
    try:
        # Initialize streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Create generation kwargs
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_length": len(inputs["input_ids"][0]) + llm_tts_config.llm_max_length,
            "temperature": llm_tts_config.llm_temperature,
            "top_p": llm_tts_config.llm_top_p,
            "repetition_penalty": llm_tts_config.llm_repetition_penalty,
            "do_sample": True,
            "streamer": streamer,
        }
        
        # Start generation in a separate thread
        generation_thread = threading.Thread(
            target=model.generate,
            kwargs=generation_kwargs
        )
        generation_thread.start()
        
        # Collect text for full response
        full_text = ""
        buffer = ""
        chunk_count = 0
        
        # Define natural breakpoints for chunking
        natural_breakpoints = [".", "!", "?", ";", ":", ","]
        priority_breakpoints = [".", "!", "?"]  # Higher priority breakpoints
        
        # Process streamed output
        for new_text in streamer:
            full_text += new_text
            buffer += new_text
            
            # Check if we have enough words or reached a natural breakpoint
            word_threshold_met = buffer.count(" ") >= llm_tts_config.chunk_word_threshold
            
            # Look for priority breakpoints first (end of sentences)
            if any(bp in buffer for bp in priority_breakpoints) and buffer.count(" ") >= 3:
                # Find the last occurrence of a priority breakpoint
                last_breakpoint_idx = max([buffer.rfind(bp) for bp in priority_breakpoints if bp in buffer])
                if last_breakpoint_idx > 0:
                    chunk = buffer[:last_breakpoint_idx + 1].strip()
                    buffer = buffer[last_breakpoint_idx + 1:].strip()
                    if chunk:
                        chunk_count += 1
                        text_queue.put((chunk, chunk_count))
            # Otherwise check for secondary breakpoints or word count            
            elif (any(bp in buffer for bp in natural_breakpoints) and buffer.count(" ") >= 3) or word_threshold_met:
                # If we have secondary breakpoints, find the last one
                if any(bp in buffer for bp in natural_breakpoints):
                    last_breakpoint_idx = max([buffer.rfind(bp) for bp in natural_breakpoints if bp in buffer])
                    if last_breakpoint_idx > 0:
                        chunk = buffer[:last_breakpoint_idx + 1].strip()
                        buffer = buffer[last_breakpoint_idx + 1:].strip()
                    else:
                        chunk = buffer.strip()
                        buffer = ""
                else:
                    # Just use word threshold
                    chunk = buffer.strip()
                    buffer = ""
                
                if chunk:
                    chunk_count += 1
                    text_queue.put((chunk, chunk_count))
        
        # Put any remaining buffer into the queue
        if buffer:
            chunk_count += 1
            text_queue.put((buffer, chunk_count))
            
        # Signal that LLM has finished generating
        text_queue.put((None, None))
        
        return full_text
        
    except Exception as e:
        print(f"\n{ColorText.RED}Error in LLM streaming: {e}{ColorText.END}")
        text_queue.put((None, None))  # Signal the TTS worker to stop
        return ""

def tts_blendshape_worker(model, tokenizer, sample_rate, device):
    """Worker function that converts text chunks to speech and generates blendshapes"""
    print(f"{ColorText.BOLD}[TTS+Blendshapes]{ColorText.END} Converting text to speech and generating blendshapes...")
    
    try:
        all_audio_chunks = []
        all_blendshapes = []
        
        # Get TTS speed and stream chunk size from config
        tts_speed = float(os.getenv("TTS_SPEED", "1.0"))
        chunk_size = int(os.getenv("TTS_STREAM_CHUNK_SIZE", "50"))
        
        while True:
            # Get text chunk from queue
            text_chunk, chunk_id = text_queue.get()
            
            # Check if this is the stop signal
            if text_chunk is None:
                break
                
            # Skip empty chunks
            if not text_chunk.strip():
                continue
            
            print(f"{ColorText.BOLD}[TTS]{ColorText.END} Processing chunk {chunk_id}: '{text_chunk}'")
            
            try:
                # Process text through TTS
                inputs = tokenizer(text_chunk, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    output = model(**inputs).waveform
                
                # Convert to numpy and scale
                audio = output.squeeze().detach().cpu().numpy()
                
                # Apply speed adjustment if needed
                if tts_speed != 1.0:
                    import librosa
                    audio = librosa.effects.time_stretch(audio, rate=tts_speed)
                
                # Calculate audio length in seconds
                audio_length_seconds = len(audio) / sample_rate
                print(f"{ColorText.BOLD}[TTS]{ColorText.END} Generated {audio_length_seconds:.2f}s of audio for chunk {chunk_id}")
                
                # Save for complete audio file
                all_audio_chunks.append(audio)
                
                # Convert to int16 for blendshape generation
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_bytes = BytesIO()
                wav.write(audio_bytes, rate=sample_rate, data=audio_int16)
                audio_bytes.seek(0)
                
                # Generate blendshapes from audio
                blendshapes = generate_facial_data_from_bytes(audio_bytes.read(), blendshape_model, device, config)
                blendshapes_list = blendshapes.tolist() if isinstance(blendshapes, np.ndarray) else blendshapes
                all_blendshapes.extend(blendshapes_list)
                
                # Put audio and blendshapes in queue
                audio_blendshape_queue.put((audio_int16.tobytes(), blendshapes_list, chunk_id))
                
                print(f"{ColorText.BOLD}[Blendshapes]{ColorText.END} Generated {len(blendshapes_list)} frames for chunk {chunk_id}")
            except Exception as e:
                print(f"{ColorText.RED}Error processing chunk {chunk_id}: {e}{ColorText.END}")
                import traceback
                print(f"{ColorText.RED}Traceback: {traceback.format_exc()}{ColorText.END}")
                # Still need to put something in the queue to maintain flow
                empty_audio = np.zeros(int(sample_rate * 0.2), dtype=np.int16)  # 0.2 second of silence
                audio_blendshape_queue.put((empty_audio.tobytes(), [], chunk_id))
        
        # Signal that processing has finished
        audio_blendshape_queue.put((None, None, None))
        
        # Return combined results
        if all_audio_chunks:
            full_audio = np.concatenate(all_audio_chunks)
            full_audio_int16 = (full_audio * 32767).astype(np.int16)
            return full_audio_int16, all_blendshapes, sample_rate
        else:
            return None, [], sample_rate
        
    except Exception as e:
        print(f"\n{ColorText.RED}Error in TTS and blendshape generation: {e}{ColorText.END}")
        import traceback
        print(f"{ColorText.RED}Traceback: {traceback.format_exc()}{ColorText.END}")
        audio_blendshape_queue.put((None, None, None))  # Signal any consumers to stop
        return None, [], sample_rate

@app.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    try:
        # Check if data is base64 encoded
        content_type = request.headers.get('Content-Type', '')
        
        if 'application/json' in content_type:
            # Handle base64 encoded audio
            data = request.json
            if not data or 'audio' not in data:
                return jsonify({"error": "Missing 'audio' field in JSON body"}), 400
                
            import base64
            try:
                audio_bytes = base64.b64decode(data['audio'])
            except Exception as e:
                return jsonify({"error": f"Failed to decode base64 audio: {str(e)}"}), 400
        else:
            # Handle raw audio bytes
            audio_bytes = request.data
            
        if not audio_bytes:
            return jsonify({"error": "No audio data provided"}), 400
        
        generated_facial_data = generate_facial_data_from_bytes(audio_bytes, blendshape_model, device, config)
        generated_facial_data_list = generated_facial_data.tolist() if isinstance(generated_facial_data, np.ndarray) else generated_facial_data

        # Send the generated blendshapes to the 3D model
        try:
            # Create temporary audio file to satisfy run_audio_animation requirements
            temp_audio = BytesIO()
            wav.write(temp_audio, rate=16000, data=np.zeros(1000, dtype=np.int16))  # Dummy audio
            temp_audio.seek(0)
            
            # Stop default animation
            stop_default_animation.set()
            
            # Run the animation
            run_audio_animation(temp_audio, generated_facial_data_list, py_face, socket_connection, default_animation_thread)
            
            # Restart default animation
            stop_default_animation.clear()
        except Exception as e:
            print(f"Error sending blendshapes to 3D model: {e}")

        return jsonify({'blendshapes': generated_facial_data_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/text_to_blendshapes', methods=['POST'])
def text_to_blendshapes_route():
    """
    Convert text to speech and then generate blendshapes
    Request:
        {
            "prompt": "Text to convert to speech and generate blendshapes"
        }
    Response:
        {
            "text": "Full text response",
            "audio": "Base64 encoded audio",
            "blendshapes": [[...blendshape values...], [...], ...]
        }
    """
    try:
        # Get text from request
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' field in request body"}), 400
        
        prompt = data['prompt']
        
        # Clear queues
        while not text_queue.empty():
            text_queue.get()
        while not audio_blendshape_queue.empty():
            audio_blendshape_queue.get()
        
        # Load LLM and TTS models if not already loaded
        models = load_llm_tts_models()
        
        # Start the LLM worker
        llm_thread = threading.Thread(
            target=llm_streaming_worker, 
            args=(models["llm_model"], models["llm_tokenizer"], prompt, device)
        )
        llm_thread.daemon = True
        llm_thread.start()
        
        # Start the TTS and blendshape worker
        tts_thread = threading.Thread(
            target=tts_blendshape_worker, 
            args=(models["tts_model"], models["tts_tokenizer"], models["sample_rate"], device)
        )
        tts_thread.daemon = True
        tts_thread.start()
        
        # Wait for threads to complete
        llm_thread.join()
        tts_thread.join()
        
        # Collect all audio and blendshape chunks
        full_audio_chunks = []
        all_blendshapes = []
        
        # Process audio and blendshape queue until end signal
        while True:
            audio_bytes, blendshapes, chunk_id = audio_blendshape_queue.get()
            
            if audio_bytes is None:  # End signal
                break
                
            # Store the audio chunk and blendshapes
            full_audio_chunks.append(audio_bytes)
            if blendshapes:
                all_blendshapes.extend(blendshapes)
        
        # Concatenate audio chunks
        if full_audio_chunks:
            full_audio = b''.join(full_audio_chunks)
            
            # Send blendshapes to the 3D model
            try:
                # Stop default animation
                stop_default_animation.set()
                
                # Convert audio bytes to format needed by run_audio_animation
                audio_data = np.frombuffer(full_audio, dtype=np.int16)
                audio_file = BytesIO()
                wav.write(audio_file, rate=models["sample_rate"], data=audio_data)
                audio_file.seek(0)
                
                # Run the animation
                run_audio_animation(audio_file, all_blendshapes, py_face, socket_connection, default_animation_thread)
                
                # Restart default animation
                stop_default_animation.clear()
            except Exception as e:
                print(f"Error sending blendshapes to 3D model: {e}")
        else:
            full_audio = b''
        
        # Import base64 for encoding binary data
        import base64
        
        # Encode binary audio data as base64 for JSON serialization
        encoded_audio = base64.b64encode(full_audio).decode('utf-8')
        
        # Return the results
        return jsonify({
            "audio": encoded_audio,
            "blendshapes": all_blendshapes
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream_text_to_blendshapes', methods=['POST'])
def stream_text_to_blendshapes_route():
    """
    Stream the conversion of text to speech and blendshapes
    Request:
        {
            "prompt": "Text to convert to speech and generate blendshapes"
        }
    Response: Stream of JSON chunks with audio and blendshapes
    """
    def generate():
        try:
            print(f"{ColorText.GREEN}[API] Received stream_text_to_blendshapes request{ColorText.END}")
            
            # Get text from request
            data = request.json
            if not data or 'prompt' not in data:
                error_msg = "Missing 'prompt' field in request body"
                print(f"{ColorText.RED}[API] {error_msg}{ColorText.END}")
                yield json.dumps({"error": error_msg}) + "\n"
                return
            
            prompt = data['prompt']
            print(f"{ColorText.GREEN}[API] Processing prompt: '{prompt}'{ColorText.END}")
            
            # Clear queues
            while not text_queue.empty():
                text_queue.get()
            while not audio_blendshape_queue.empty():
                audio_blendshape_queue.get()
            
            # Load LLM and TTS models if not already loaded
            try:
                print(f"{ColorText.GREEN}[API] Loading models...{ColorText.END}")
                models = load_llm_tts_models()
                print(f"{ColorText.GREEN}[API] Models loaded successfully{ColorText.END}")
            except Exception as e:
                error_msg = f"Failed to load models: {str(e)}"
                print(f"{ColorText.RED}[API] {error_msg}{ColorText.END}")
                yield json.dumps({"error": error_msg}) + "\n"
                return
            
            # Start the LLM worker
            print(f"{ColorText.GREEN}[API] Starting LLM worker...{ColorText.END}")
            llm_thread = threading.Thread(
                target=llm_streaming_worker, 
                args=(models["llm_model"], models["llm_tokenizer"], prompt, device)
            )
            llm_thread.daemon = True
            llm_thread.start()
            
            # Start the TTS and blendshape worker
            print(f"{ColorText.GREEN}[API] Starting TTS worker...{ColorText.END}")
            tts_thread = threading.Thread(
                target=tts_blendshape_worker, 
                args=(models["tts_model"], models["tts_tokenizer"], models["sample_rate"], device)
            )
            tts_thread.daemon = True
            tts_thread.start()
            
            # Stop default animation for streaming
            print(f"{ColorText.GREEN}[API] Stopping default animation...{ColorText.END}")
            stop_default_animation.set()
            
            # Collect all chunks for final playback
            # all_audio_chunks = [] # No longer needed as audio is streamed to client
            # all_blendshapes = [] # No longer needed as blendshapes are streamed directly
            
            # For real-time delivery monitoring
            last_chunk_time = time.time()
            frame_counter = 0 # To apply blinking based on default animation data
            frame_duration = 1.0 / 60.0 # Target 60 FPS
            
            # Stream each chunk as it becomes available
            print(f"{ColorText.GREEN}[API] Starting to process and stream audio/blendshape chunks...{ColorText.END}")
            
            # Remove buffering logic for immediate yield
            # buffer_chunks = [] 
            # max_buffer_size = 2  
            # max_buffer_time = 0.5  
            
            # Import base64 for encoding binary data
            import base64
            
            while True:
                try:
                    # Non-blocking queue get with timeout
                    try:
                        audio_bytes, blendshapes_chunk, chunk_id = audio_blendshape_queue.get(timeout=1.0) # Increased timeout slightly
                    except queue.Empty:
                        # Check if workers are still running
                        if not llm_thread.is_alive() and not tts_thread.is_alive() and audio_blendshape_queue.empty():
                            print(f"{ColorText.YELLOW}[API] Workers finished and queue empty. Breaking loop.{ColorText.END}")
                            break # Exit loop if workers are done and queue is empty
                        # No buffer to flush, just continue waiting
                        continue
                    
                    # Check for end signal
                    if audio_bytes is None:
                        print(f"{ColorText.GREEN}[API] Received end signal for audio chunks{ColorText.END}")
                        break
                    
                    # Log chunk info
                    current_time = time.time()
                    # time_since_last = current_time - last_chunk_time # Not relevant without buffer
                    print(f"{ColorText.GREEN}[API] Processing chunk {chunk_id}, audio size: {len(audio_bytes)} bytes, blendshapes: {len(blendshapes_chunk)}{ColorText.END}")
                    
                    # --- Yield Audio Chunk to Client IMMEDIATELY ---
                    encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
                    response_chunk = {
                        "chunk_id": chunk_id,
                        "audio": encoded_audio,  # Base64 encoded audio
                    }
                    try:
                        # Put raw bytes onto playback queue *before* yielding to client
                        playback_audio_queue.put(audio_bytes) 
                        
                        yield json.dumps(response_chunk) + "\n"
                        print(f"{ColorText.CYAN}[API] Yielded audio chunk {chunk_id} to client. Added to local playback queue.{ColorText.END}")
                    except Exception as yield_err:
                        print(f"{ColorText.RED}[API] Error yielding audio chunk {chunk_id}: {yield_err}{ColorText.END}")
                        break
                    
                    # --- Real-time Blendshape Streaming (After Audio Yielded) --- 
                    if blendshapes_chunk:
                        # Apply blinking data to the blendshape chunk
                        apply_blink_to_facial_data(blendshapes_chunk, default_animation_data)
                        
                        # Send each frame in the chunk
                        blendshape_send_start_time = time.time()
                        frames_sent_this_chunk = 0
                        for frame_data in blendshapes_chunk:
                            frame_start_time = time.time()
                            try:
                                # Set blendshapes in py_face object
                                for i in range(min(len(frame_data), 61)): # Ensure we don't exceed blendshape count
                                    py_face.set_blendshape(FaceBlendShape(i), float(frame_data[i]))
                                # Encode and send the frame
                                encoded_frame = py_face.encode()
                                socket_connection.sendall(encoded_frame)
                                frames_sent_this_chunk += 1
                            except Exception as e:
                                print(f"{ColorText.RED}[API] Error sending blendshape frame: {e}{ColorText.END}")
                            
                            # Maintain frame rate (approx 60 FPS)
                            elapsed_time = time.time() - frame_start_time
                            sleep_time = frame_duration - elapsed_time
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                            frame_counter += 1
                        blendshape_send_duration = time.time() - blendshape_send_start_time
                        print(f"{ColorText.BLUE}[API] Sent {frames_sent_this_chunk} blendshape frames for chunk {chunk_id} in {blendshape_send_duration:.2f}s.{ColorText.END}")
                    # --- End Real-time Blendshape Streaming --- 
                        
                except Exception as e:
                    print(f"{ColorText.RED}[API] Error processing chunk loop: {str(e)}{ColorText.END}")
                    import traceback
                    print(f"{ColorText.RED}[API] Traceback: {traceback.format_exc()}{ColorText.END}")
                    continue
            
            # Ensure workers are joined before restarting default animation
            llm_thread.join(timeout=5)
            tts_thread.join(timeout=5)
            
            # After streaming is complete, NO LONGER play the entire animation
            # The animation was streamed frame-by-frame in the loop above
            print(f"{ColorText.GREEN}[API] Finished processing all chunks.{ColorText.END}")
            
            # Restart default animation
            print(f"{ColorText.GREEN}[API] Restarting default animation{ColorText.END}")
            stop_default_animation.clear()
            # Need to restart the thread if it was joined/stopped
            global default_animation_thread # Declare as global to modify it
            if not default_animation_thread or not default_animation_thread.is_alive():
                default_animation_thread = threading.Thread(target=default_animation_loop, args=(py_face,))
                default_animation_thread.daemon = True
                default_animation_thread.start()
            
            # Send final message to client
            yield json.dumps({"status": "completed"}) + "\n"
            
            # Signal playback queue if loop exited unexpectedly before end signal
            playback_audio_queue.put(None) 
            
        except Exception as e:
            error_msg = f"Unexpected error in generate(): {str(e)}"
            print(f"{ColorText.RED}[API] {error_msg}{ColorText.END}")
            import traceback
            print(f"{ColorText.RED}[API] Traceback: {traceback.format_exc()}{ColorText.END}")
            try:
                yield json.dumps({"error": error_msg}) + "\n"
            except Exception as yield_err:
                 print(f"{ColorText.RED}[API] Error yielding error message: {yield_err}{ColorText.END}")
            
            # Make sure default animation is restarted in case of error
            stop_default_animation.clear()
            if not default_animation_thread or not default_animation_thread.is_alive():
                default_animation_thread = threading.Thread(target=default_animation_loop, args=(py_face,))
                default_animation_thread.daemon = True
                default_animation_thread.start()
            
            # Signal playback queue in case of error
            playback_audio_queue.put(None)
    
    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

# --- Audio Playback Worker ---

def audio_playback_worker(playback_queue, sample_rate, device_index=None):
    """Worker function to play audio chunks from a queue using sounddevice."""
    stream = None
    try:
        # Query devices if no index provided
        if device_index is None:
            print("Available audio output devices:")
            print(sd.query_devices())
            # You might want to set a default device index here or prompt the user
            # For now, let's try the default device
            device_index = sd.default.device[1] # Index 1 is for output device
            print(f"Using default output device index: {device_index}")
        
        # Initialize the output stream
        stream = sd.OutputStream(
            samplerate=sample_rate, 
            channels=1, 
            dtype='int16', # TTS generates int16 bytes
            device=device_index
        )
        stream.start()
        print(f"{ColorText.PURPLE}[PlaybackWorker] Started. Sample Rate: {sample_rate}, Device: {device_index}{ColorText.END}")
        
        while True:
            audio_chunk_bytes = playback_queue.get()
            if audio_chunk_bytes is None:
                print(f"{ColorText.PURPLE}[PlaybackWorker] Received stop signal.{ColorText.END}")
                break
            
            if len(audio_chunk_bytes) > 0:
                try:
                    # Convert bytes back to numpy array for sounddevice
                    # Ensure correct dtype (int16)
                    audio_data = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
                    stream.write(audio_data)
                    # print(f"{ColorText.PURPLE}[PlaybackWorker] Played chunk of size {len(audio_data)} samples.{ColorText.END}") # Verbose
                except Exception as e:
                    print(f"{ColorText.RED}[PlaybackWorker] Error playing audio chunk: {e}{ColorText.END}")
            else:
                 print(f"{ColorText.YELLOW}[PlaybackWorker] Received empty audio chunk. Skipping.{ColorText.END}")
            
            playback_queue.task_done() # Mark task as done

    except Exception as e:
        print(f"{ColorText.RED}[PlaybackWorker] Error: {e}{ColorText.END}")
        import traceback
        print(f"{ColorText.RED}Traceback: {traceback.format_exc()}{ColorText.END}")
    finally:
        if stream:
            try:
                stream.stop()
                stream.close()
                print(f"{ColorText.PURPLE}[PlaybackWorker] Stream stopped and closed.{ColorText.END}")
            except Exception as e:
                print(f"{ColorText.RED}[PlaybackWorker] Error closing stream: {e}{ColorText.END}")
        # Ensure the final None is marked done if loop exited via exception
        if 'playback_queue' in locals() and not playback_queue.empty():
             try:
                 if playback_queue.get(block=False) is None:
                     playback_queue.task_done()
             except queue.Empty:
                 pass
             except Exception as e:
                 print(f"{ColorText.RED}[PlaybackWorker] Error clearing queue: {e}{ColorText.END}")


if __name__ == '__main__':
    playback_thread = None # Initialize playback_thread
    try:
        # --- Get Sample Rate --- 
        # Need to ensure models are loaded or get rate from config
        # For now, hardcoding based on the default TTS model
        # A better way would be to load models first or read from config
        tts_model_name = os.getenv("TTS_MODEL", "facebook/mms-tts-eng")
        sample_rate = 16000 # Default for facebook/mms-tts-eng
        print(f"{ColorText.YELLOW}[Main] Assuming sample rate {sample_rate} for TTS model {tts_model_name}. Ensure this is correct.{ColorText.END}")
        # TODO: Dynamically determine sample rate after model loading if possible
        
        # --- Start Default Animation --- 
        default_animation_thread = threading.Thread(target=default_animation_loop, args=(py_face,))
        default_animation_thread.daemon = True
        default_animation_thread.start()
        
        # --- Start Playback Worker ---
        print(f"{ColorText.GREEN}[Main] Starting local audio playback worker...{ColorText.END}")
        playback_thread = threading.Thread(target=audio_playback_worker, args=(playback_audio_queue, sample_rate))
        playback_thread.daemon = True
        playback_thread.start()
        
        # --- Run Flask App --- 
        app.run(host='127.0.0.1', port=5000)
        
    finally:
        # Clean up resources
        print("Shutting down... stopping threads and closing socket.")
        
        # Stop default animation
        stop_default_animation.set()
        if default_animation_thread and default_animation_thread.is_alive():
            try:
                default_animation_thread.join(timeout=2)
            except Exception as e:
                print(f"Error joining default animation thread: {e}")
                
        # Stop playback worker (by putting None on its queue)
        # This should ideally happen when the generate() function finishes or errors out,
        # but we add a safeguard here.
        playback_audio_queue.put(None)
        if playback_thread and playback_thread.is_alive():
            try:
                print("Waiting for playback worker to finish...")
                playback_thread.join(timeout=5) # Give it time to process the None signal
                if playback_thread.is_alive():
                    print(f"{ColorText.YELLOW}[Main] Playback worker did not exit cleanly.{ColorText.END}")
                else:
                    print("Playback worker finished.")
            except Exception as e:
                print(f"Error joining playback worker thread: {e}")
                
        # Close socket
        if socket_connection:
             try:
                socket_connection.close()
                print("Socket connection closed.")
             except Exception as e:
                print(f"Error closing socket connection: {e}")
        print("Shutdown complete.")
