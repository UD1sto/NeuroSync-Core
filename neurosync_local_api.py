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
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from utils.generated_runners import run_audio_animation

# Load environment variables
dotenv.load_dotenv()

app = flask.Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Activated device:", device)

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
audio_blendshape_queue = queue.Queue()  # Queue for audio chunks and blendshapes

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
        
        # Process streamed output
        for new_text in streamer:
            full_text += new_text
            buffer += new_text
            
            # Every few words, send to TTS queue
            if buffer.count(" ") >= llm_tts_config.chunk_word_threshold or "." in buffer or "," in buffer or "!" in buffer or "?" in buffer:
                chunk_count += 1
                
                # Put the chunk and its ID in the queue
                text_queue.put((buffer, chunk_count))
                
                # Reset for next chunk
                buffer = ""
        
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
            
            # Process text through TTS
            inputs = tokenizer(text_chunk, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model(**inputs).waveform
            
            # Convert to numpy and scale
            audio = output.squeeze().detach().cpu().numpy()
            
            # Calculate audio length in seconds
            audio_length_seconds = len(audio) / sample_rate
            
            # Save for complete audio file
            all_audio_chunks.append(audio)
            
            # Convert to int16 for blendshape generation
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = BytesIO()
            wav.write(audio_bytes, rate=sample_rate, data=audio_int16)
            audio_bytes.seek(0)
            
            # Generate blendshapes from audio
            try:
                blendshapes = generate_facial_data_from_bytes(audio_bytes.read(), blendshape_model, device, config)
                blendshapes_list = blendshapes.tolist() if isinstance(blendshapes, np.ndarray) else blendshapes
                all_blendshapes.extend(blendshapes_list)
                
                # Put audio and blendshapes in queue
                audio_blendshape_queue.put((audio_int16.tobytes(), blendshapes_list, chunk_id))
                
                print(f"{ColorText.BOLD}[Blendshapes]{ColorText.END} Generated {len(blendshapes_list)} frames for chunk {chunk_id}")
            except Exception as e:
                print(f"{ColorText.RED}Error generating blendshapes: {e}{ColorText.END}")
                audio_blendshape_queue.put((audio_int16.tobytes(), [], chunk_id))
        
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
        audio_blendshape_queue.put((None, None, None))  # Signal any consumers to stop
        return None, [], sample_rate

@app.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    audio_bytes = request.data
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
        
        # Return the results
        return jsonify({
            "audio": full_audio,
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
            all_audio_chunks = []
            all_blendshapes = []
            
            # Stream each chunk as it becomes available
            print(f"{ColorText.GREEN}[API] Starting to process audio chunks...{ColorText.END}")
            while True:
                audio_bytes, blendshapes, chunk_id = audio_blendshape_queue.get()
                
                if audio_bytes is None:  # End signal
                    print(f"{ColorText.GREEN}[API] Received end signal for audio chunks{ColorText.END}")
                    break
                
                print(f"{ColorText.GREEN}[API] Processing chunk {chunk_id}, audio size: {len(audio_bytes)} bytes, blendshapes: {len(blendshapes)}{ColorText.END}")
                
                # Store for final playback
                all_audio_chunks.append(audio_bytes)
                if blendshapes:
                    all_blendshapes.extend(blendshapes)
                
                # Create response chunk
                response_chunk = {
                    "chunk_id": chunk_id,
                    "audio": audio_bytes,
                    "blendshapes": blendshapes
                }
                
                # Send the chunk
                yield json.dumps(response_chunk) + "\n"
            
            # After streaming is complete, play the entire animation
            if all_audio_chunks:
                try:
                    print(f"{ColorText.GREEN}[API] Playing complete animation with {len(all_blendshapes)} blendshapes{ColorText.END}")
                    
                    # Convert audio bytes to format needed by run_audio_animation
                    full_audio = b''.join(all_audio_chunks)
                    audio_data = np.frombuffer(full_audio, dtype=np.int16)
                    audio_file = BytesIO()
                    wav.write(audio_file, rate=models["sample_rate"], data=audio_data)
                    audio_file.seek(0)
                    
                    # Run the animation
                    run_audio_animation(audio_file, all_blendshapes, py_face, socket_connection, default_animation_thread)
                    print(f"{ColorText.GREEN}[API] Animation completed successfully{ColorText.END}")
                except Exception as e:
                    error_msg = f"Error sending blendshapes to 3D model: {str(e)}"
                    print(f"{ColorText.RED}[API] {error_msg}{ColorText.END}")
                    import traceback
                    print(f"{ColorText.RED}[API] Traceback: {traceback.format_exc()}{ColorText.END}")
                    yield json.dumps({"error": error_msg}) + "\n"
            else:
                print(f"{ColorText.YELLOW}[API] No audio chunks generated{ColorText.END}")
            
            # Restart default animation
            print(f"{ColorText.GREEN}[API] Restarting default animation{ColorText.END}")
            stop_default_animation.clear()
            
            # Send final message
            yield json.dumps({"status": "completed"}) + "\n"
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"{ColorText.RED}[API] {error_msg}{ColorText.END}")
            import traceback
            print(f"{ColorText.RED}[API] Traceback: {traceback.format_exc()}{ColorText.END}")
            yield json.dumps({"error": error_msg}) + "\n"
            
            # Make sure default animation is restarted
            stop_default_animation.clear()
    
    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=5000)
    finally:
        # Clean up resources
        stop_default_animation.set()
        if default_animation_thread and default_animation_thread.is_alive():
            default_animation_thread.join()
        socket_connection.close()
