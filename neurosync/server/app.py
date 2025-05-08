"""
NeuroSync Local API Server

This Flask server provides endpoints for:
- /audio_to_blendshapes (WAV bytes → animation)
- /text_to_blendshapes (prompt → speech + animation)
- /stream_text_to_blendshapes (prompt → streaming speech + animation)

It should be started with CUDA available if possible.
"""
import base64 # Added for JSON serialization
import importlib
import json
import os
import queue
import sys
import threading
import time
from io import BytesIO

import flask
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd # Added for local audio playback
import soundfile as sf # Added for reading WAV bytes in playback worker
import torch
import dotenv
from flask import Response, jsonify, request, stream_with_context, Blueprint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    VitsModel,
    utils as hf_utils
)
from flask_cors import CORS  # NEW: CORS support
import openai # Added for OpenAI API access

# Add the project root to sys.path to allow absolute imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Use absolute imports based on the new structure
from neurosync.core.livelink.livelink_init import (
    create_socket_connection, initialize_py_face
)
from neurosync.core.livelink.pylivelinkface import FaceBlendShape
from neurosync.core.livelink.animations.default_animation import (
    default_animation_data, default_animation_loop, stop_default_animation
)
from neurosync.core.livelink.send_to_unreal import apply_blink_to_facial_data
from neurosync.core.runtime.player import Player
from neurosync.core.model.blendshape_sequence import BlendshapeSequence
from neurosync.core.model.model import load_model
from neurosync.core.config import config
from neurosync.core.generate_face_shapes import generate_facial_data_from_bytes
from neurosync.core.bridge import BridgeCache # Assuming utils is still at the root
from neurosync.core.color_text import ColorText # Assuming a utility class
from neurosync.core.scb_store import scb_store
from neurosync.core.summarizer import SummarizerThread

# Load environment variables
dotenv.load_dotenv()

app = flask.Flask(__name__)

# Check for CUDA availability and respect USE_CUDA environment variable
use_cuda_env = os.getenv("USE_CUDA", "auto").lower()
cuda_available = torch.cuda.is_available()

if use_cuda_env == "true":
    if cuda_available:
        device = torch.device('cuda')
        print(f"{ColorText.GREEN}✅ Using CUDA GPU acceleration{ColorText.END}")
    else:
        print(f"{ColorText.YELLOW}⚠️ CUDA requested but not available! Make sure PyTorch is installed with CUDA support and GPU drivers are updated.{ColorText.END}")
        print(f"{ColorText.YELLOW}⚠️ Falling back to CPU mode{ColorText.END}")
        device = torch.device('cpu')
elif use_cuda_env == "false":
    device = torch.device('cpu')
    print(f"{ColorText.CYAN}ℹ️ Using CPU mode (CUDA disabled by configuration){ColorText.END}")
else:  # "auto" or any other value
    device = torch.device('cuda' if cuda_available else 'cpu')
    if cuda_available:
        print(f"{ColorText.GREEN}✅ Using CUDA GPU acceleration (auto-detected){ColorText.END}")
    else:
        print(f"{ColorText.CYAN}ℹ️ CUDA not available, using CPU mode{ColorText.END}")

print(f"{ColorText.BOLD}🔧 Activated device:{ColorText.END}", device)

# --- Model Loading ---
# Construct the absolute path for the model
model_path = os.path.join(root_dir, 'neurosync', 'core', 'model', 'model.pth')
# Use the imported config object
blendshape_model = load_model(model_path, config, device)

# --- LiveLink Initialization ---
print(f"{ColorText.BOLD}[LiveLink]{ColorText.END} Initializing PyFace and Socket Connection...")
py_face = initialize_py_face()
socket_connection = create_socket_connection()
default_animation_thread = None # Initialize here
stop_default_animation_event = stop_default_animation # Use the imported event
print(f"{ColorText.BOLD}[LiveLink]{ColorText.END} Initialization complete.")

# --- Global Queues ---
text_queue = queue.Queue()
audio_blendshape_queue = queue.Queue()
playback_audio_queue = queue.Queue()

# --- LLM-TTS Configuration ---
class LLMTTSConfig:
    """Configuration class for LLM and TTS models"""
    def __init__(self):
        self.llm_model = os.getenv("LLM_MODEL", "distilgpt2")
        self.llm_max_length = int(os.getenv("LLM_MAX_LENGTH", "50"))
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.llm_top_p = float(os.getenv("LLM_TOP_P", "0.9"))
        self.llm_repetition_penalty = float(os.getenv("LLM_REPETITION_PENALTY", "1.0"))
        self.tts_model = os.getenv("TTS_MODEL", "facebook/mms-tts-eng")
        self.tts_speed = float(os.getenv("TTS_SPEED", "1.0"))
        self.chunk_word_threshold = int(os.getenv("CHUNK_WORD_THRESHOLD", "3"))
        self.quantization = os.getenv("QUANTIZATION", "").lower()
        self.trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
        self.hf_token = os.getenv("HF_TOKEN", None)
        self.tts_stream_chunk_size = int(os.getenv("TTS_STREAM_CHUNK_SIZE", "50"))

    def get_device(self):
        """Get the appropriate device"""
        return device

llm_tts_config = LLMTTSConfig()

# --- Model Cache ---
pipeline_models = {} # Lazy loaded models

def load_llm_tts_models():
    """Load LLM and TTS models if not already loaded"""
    global pipeline_models
    if not pipeline_models:
        print(f"{ColorText.BOLD}[Pipeline]{ColorText.END} Loading LLM and TTS models...")
        # Determine LLM provider strategy for the server's internal pipeline
        server_llm_provider = os.getenv("LLM_PROVIDER", "huggingface").lower()
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Initialize pipeline_models with the provider information early
        pipeline_models = {
            "llm_provider": server_llm_provider if server_llm_provider == "openai" and openai_api_key else "huggingface",
            "llm_model": None,
            "llm_tokenizer": None,
            "tts_model": None,
            "tts_tokenizer": None,
            "sample_rate": 16000 # Default, will be updated by TTS model
        }

        hf_utils.logging.set_verbosity_info()

        try:
            print(f"{ColorText.BOLD}[Config]{ColorText.END} TRUST_REMOTE_CODE: {llm_tts_config.trust_remote_code}")
            print(f"{ColorText.BOLD}[Config]{ColorText.END} LLM Model: {llm_tts_config.llm_model}")
            print(f"{ColorText.BOLD}[Config]{ColorText.END} Quantization: {llm_tts_config.quantization or 'None'}")
            print(f"{ColorText.BOLD}[Config]{ColorText.END} TTS Model: {llm_tts_config.tts_model}")

            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            print(f"{ColorText.BOLD}[Config]{ColorText.END} HuggingFace cache directory: {cache_dir}")

            # Load LLM
            print(f"\n{ColorText.BOLD}[LLM]{ColorText.END} Loading LLM model: {llm_tts_config.llm_model}...")
            print(f"{ColorText.YELLOW}If downloading, progress will be displayed below:{ColorText.END}")

            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            os.environ["TRANSFORMERS_VERBOSITY"] = "info"

            if pipeline_models["llm_provider"] == "huggingface":
                print(f"{ColorText.BOLD}[LLM-Server]{ColorText.END} Using HuggingFace provider for internal LLM pipeline.")
                token_arg = {"token": llm_tts_config.hf_token} if llm_tts_config.hf_token and llm_tts_config.hf_token != "your_huggingface_token_here" else {}

                print(f"{ColorText.BOLD}[LLM]{ColorText.END} Loading tokenizer (trust_remote_code={llm_tts_config.trust_remote_code}, use_fast=False)... for {llm_tts_config.llm_model}")
                llm_tokenizer = AutoTokenizer.from_pretrained(
                    llm_tts_config.llm_model, # This is the HF model ID from env (e.g. mistral)
                    trust_remote_code=llm_tts_config.trust_remote_code,
                    use_fast=False, # Avoid sentencepiece dependency if possible
                    **token_arg
                )

                model_kwargs = {
                    "trust_remote_code": llm_tts_config.trust_remote_code,
                    "device_map": "auto",
                    **token_arg
                }

                if llm_tts_config.quantization:
                    # Dynamically import bitsandbytes only if needed
                    try:
                        import bitsandbytes as bnb # type: ignore
                        print(f"{ColorText.BOLD}[LLM]{ColorText.END} Enabling {llm_tts_config.quantization} quantization...")

                        if llm_tts_config.quantization == "int8":
                            model_kwargs["load_in_8bit"] = True
                            print(f"{ColorText.BOLD}[LLM]{ColorText.END} Using INT8 quantization.")
                        elif llm_tts_config.quantization == "fp4": # Common name for 4bit
                            quant_type = os.getenv("BNB_4BIT_QUANT_TYPE", "nf4")
                            use_double_quant = os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "false").lower() == "true"
                            compute_dtype_str = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "float16")
                            compute_dtype = torch.float16 if compute_dtype_str == "float16" else torch.bfloat16

                            from packaging import version # type: ignore
                            if version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.41.0"):
                                from transformers import BitsAndBytesConfig
                                bnb_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_quant_type=quant_type,
                                    bnb_4bit_use_double_quant=use_double_quant,
                                    bnb_4bit_compute_dtype=compute_dtype
                                )
                                model_kwargs["quantization_config"] = bnb_config
                                print(f"{ColorText.BOLD}[LLM]{ColorText.END} Using 4-bit quantization (type={quant_type}, double_quant={use_double_quant}, compute_dtype={compute_dtype_str}).")
                            else:
                                model_kwargs["load_in_4bit"] = True
                                print(f"{ColorText.YELLOW}⚠️ Using legacy 4-bit quantization (update bitsandbytes for full config options).{ColorText.END}")
                        else:
                            print(f"{ColorText.YELLOW}⚠️ Unsupported quantization type '{llm_tts_config.quantization}'. Loading model without quantization.{ColorText.END}")
                    except ImportError:
                        print(f"{ColorText.YELLOW}⚠️ bitsandbytes not installed. Cannot apply quantization. Install with 'pip install bitsandbytes'.{ColorText.END}")
                        llm_tts_config.quantization = "" # Disable quantization if library not found

                print(f"{ColorText.BOLD}[LLM]{ColorText.END} Loading model '{llm_tts_config.llm_model}' with quantization='{llm_tts_config.quantization or 'None'}'...")
                llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_tts_config.llm_model, # This is the HF model ID
                    **model_kwargs
                )

                if llm_tokenizer.pad_token is None:
                    llm_tokenizer.pad_token = llm_tokenizer.eos_token

                pipeline_models["llm_model"] = llm_model
                pipeline_models["llm_tokenizer"] = llm_tokenizer
            elif pipeline_models["llm_provider"] == "openai":
                print(f"{ColorText.BOLD}[LLM-Server]{ColorText.END} Using OpenAI provider for internal LLM pipeline. No local LLM model will be loaded by the server.")
                # No HuggingFace LLM model or tokenizer to load in this case.
                # OPENAI_API_KEY and OPENAI_MODEL will be used by the worker.
                pass # llm_model and llm_tokenizer remain None in pipeline_models

            # Load TTS
            print(f"\n{ColorText.BOLD}[TTS]{ColorText.END} Loading TTS model: {llm_tts_config.tts_model}...")
            print(f"{ColorText.YELLOW}If downloading, progress will be displayed below:{ColorText.END}")
            # Use a fresh token_arg for TTS, as it might be different from LLM's needs (e.g. if LLM is openai)
            tts_token_arg = {"token": llm_tts_config.hf_token} if llm_tts_config.hf_token and llm_tts_config.hf_token != "your_huggingface_token_here" else {}
            tts_tokenizer = AutoTokenizer.from_pretrained(llm_tts_config.tts_model, **tts_token_arg)
            tts_model = VitsModel.from_pretrained(llm_tts_config.tts_model, **tts_token_arg).to(device)

            sample_rate = tts_model.config.sampling_rate
            print(f"{ColorText.BOLD}[TTS]{ColorText.END} TTS sample rate: {sample_rate} Hz")

            # Update TTS parts in the existing pipeline_models dictionary
            pipeline_models["tts_model"] = tts_model
            pipeline_models["tts_tokenizer"] = tts_tokenizer
            pipeline_models["sample_rate"] = sample_rate

            print(f"{ColorText.BOLD}[Pipeline]{ColorText.END} Models loaded successfully!")

        except Exception as e:
            import traceback
            print(f"{ColorText.RED}[ERROR] Failed to load models: {str(e)}{ColorText.END}")
            print(f"{ColorText.RED}Traceback:\n{traceback.format_exc()}{ColorText.END}")
            # Clear partially loaded models to avoid issues
            pipeline_models = {} # Reset on failure
            raise # Re-raise to stop the application if models fail to load

    return pipeline_models

# --- Register SCB Blueprint -------------------------------------------------
scb_bp = Blueprint('scb_api', __name__)

# --- Security & Debug for SCB API -------------------------------------------
API_KEY = os.getenv("NEUROSYNC_API_KEY", "")
SCB_API_DEBUG = os.getenv("SCB_API_DEBUG", "false").lower() == "true"

@scb_bp.before_request
def _scb_auth_and_log():
    if SCB_API_DEBUG:
        log_prefix = f"{ColorText.YELLOW}[SCB API]{ColorText.END}"
        log_message = f"{request.method} {request.path} from {request.remote_addr}"
        if request.method in ['POST', 'PUT'] and request.data:
            try:
                # Attempt to parse JSON for pretty printing, fallback to raw bytes
                request_body = request.get_json() if request.is_json else request.get_data(as_text=True)
                log_message += f"\\n  Request Body: {request_body}"
            except Exception:
                log_message += f"\\n  Request Body (raw): {request.data[:200]}{'...' if len(request.data) > 200 else ''}" # Log first 200 bytes
        print(f"{log_prefix} {log_message}")

    if API_KEY:
        provided = request.headers.get("X-NeuroSync-Key", "")
        if provided != API_KEY:
            if SCB_API_DEBUG:
                print(f"{ColorText.RED}[SCB API] Unauthorized request – invalid key{ColorText.END}")
            return jsonify({"error": "unauthorized"}), 401

@scb_bp.route('/event', methods=['POST'])
def scb_event_route():
    data = request.json or {}
    if not all(k in data for k in ['type', 'actor', 'text']):
        return jsonify({'error': 'Fields "type", "actor", "text" required'}), 400
    scb_store.append(data)
    return jsonify({'status': 'ok'})

@scb_bp.route('/directive', methods=['POST'])
def scb_directive_route():
    data = request.json or {}
    if not all(k in data for k in ['actor', 'text']):
        return jsonify({'error': 'Fields "actor", "text" required'}), 400
    data['type'] = 'directive'
    if 'ttl' not in data:
        data['ttl'] = 15
    scb_store.append(data)
    response_data = {'status': 'ok'}
    if SCB_API_DEBUG:
        print(f"{ColorText.YELLOW}[SCB API]{ColorText.END} Response for POST /scb/directive: {response_data}")
    return jsonify(response_data)

@scb_bp.route('/slice', methods=['GET'])
def scb_slice_route():
    # Return the full SCB whiteboard (summary + full window).
    # The previous token-budgeted behaviour is now deprecated. We keep the
    #   optional `tokens` query param for backward-compatibility but ignore it.
    response_data = scb_store.get_full()
    if SCB_API_DEBUG:
        # Log summary and window size for brevity
        summary = response_data.get('summary', '')
        window_size = len(response_data.get('window', []))
        print(f"{ColorText.YELLOW}[SCB API]{ColorText.END} Response for GET /scb/slice: Summary Length={len(summary)}, Window Entries={window_size}")
    return jsonify(response_data)

@scb_bp.route('/ping', methods=['GET'])  # NEW health-check endpoint
def scb_ping_route():
    response_data = {'status': 'ok'}
    if SCB_API_DEBUG:
        print(f"{ColorText.YELLOW}[SCB API]{ColorText.END} Response for GET /scb/ping: {response_data}")
    return jsonify(response_data)

app.register_blueprint(scb_bp, url_prefix='/scb')

# --- NEW CORS CONFIG ---------------------------------------------------------
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
CORS(app, resources={r"/scb/*": {"origins": allowed_origins}})

# --- Worker Functions ---

def llm_streaming_worker(pipeline_models_dict, prompt, device):
    """Streams text from LLM, logs to SCB and sends chunks to TTS queue"""
    # Log user prompt as event
    scb_store.append_chat(prompt, actor='user')
    print(f"\n{ColorText.BOLD}[LLM]{ColorText.END} Generating response...")
    try:
        server_llm_provider = pipeline_models_dict.get("llm_provider", "huggingface")

        # Construct the base prompt / messages structure
        # This structure is used by both OpenAI and the full_prompt for HuggingFace
        persona = "You are Mai, a witty and helpful VTuber assistant with a dry sense of humor."
        summary = scb_store.get_summary()
        recent_chat = scb_store.get_recent_chat(3)
        bridge_txt = BridgeCache.read()
        prompt_parts = [persona]
        if bridge_txt:
            prompt_parts.append(bridge_txt)
        if summary:
            prompt_parts.append(f"Current summary:\n{summary}")
        if recent_chat:
            prompt_parts.append(f"Recent chat:\n{recent_chat}")
        prompt_parts.append(f"User: {prompt}\nAI:")

        full_text = ""
        buffer = ""
        natural_breakpoints = [".", "!", "?", ";", ":", ","]
        priority_breakpoints = [".", "!", "?"]
        chunk_id_counter = 0 # Use a simple counter for chunk IDs

        if server_llm_provider == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            if not openai_api_key:
                print(f"{ColorText.RED}[LLM-Server] OpenAI provider selected, but OPENAI_API_KEY is not set. Cannot proceed.{ColorText.END}")
                text_queue.put((None, None))
                return ""

            print(f"{ColorText.BOLD}[LLM-Server]{ColorText.END} Using OpenAI ({openai_model_name}) for text generation.")

            # Prepare messages for OpenAI
            system_message_for_openai = "\n\n".join(prompt_parts[:-1]) # All parts except the "User: ... AI:" part
            openai_messages = [
                {"role": "system", "content": system_message_for_openai},
                {"role": "user", "content": prompt} # Just the user's prompt
            ]

            try:
                oai_client = openai.OpenAI(api_key=openai_api_key)
                response_stream = oai_client.chat.completions.create(
                    model=openai_model_name,
                    messages=openai_messages,
                    max_tokens=llm_tts_config.llm_max_length,
                    temperature=llm_tts_config.llm_temperature,
                    top_p=llm_tts_config.llm_top_p,
                    stream=True
                )

                for chunk in response_stream:
                    new_text = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content else ""
                    if new_text:
                        full_text += new_text
                        buffer += new_text
                        # Chunking logic (same as below for HF)
                        send = False
                        if any(bp in buffer for bp in priority_breakpoints) and buffer.count(" ") >= 3:
                            idx = max(buffer.rfind(bp) for bp in priority_breakpoints if bp in buffer) # Ensure bp is in buffer
                            send = True
                            cut = buffer[:idx+1].strip()
                            buffer = buffer[idx+1:].strip()
                        elif buffer.count(" ") >= llm_tts_config.chunk_word_threshold:
                            idx = buffer.rfind(" ")
                            cut = buffer[:idx].strip() if idx != -1 else buffer.strip()
                            buffer = buffer[idx:].strip() if idx != -1 else ""
                            send = True
                        if send and cut:
                            chunk_id_counter += 1
                            text_queue.put((cut, chunk_id_counter))
            except Exception as e_openai:
                print(f"{ColorText.RED}[LLM-Server] OpenAI API error: {e_openai}{ColorText.END}")
                text_queue.put((None, None))
                return ""
        else: # HuggingFace provider
            print(f"{ColorText.BOLD}[LLM-Server]{ColorText.END} Using HuggingFace model for text generation: {llm_tts_config.llm_model}")
            hf_model = pipeline_models_dict.get("llm_model")
            hf_tokenizer = pipeline_models_dict.get("llm_tokenizer")

            if not hf_model or not hf_tokenizer:
                print(f"{ColorText.RED}[LLM-Server] HuggingFace model or tokenizer not loaded. Cannot proceed.{ColorText.END}")
                text_queue.put((None, None))
                return ""

            full_prompt_for_hf = "\n\n".join(prompt_parts)
            print(f"{ColorText.BOLD}[LLM]{ColorText.END} Prompt (trunc): {full_prompt_for_hf[:300]}...")
            streamer = TextIteratorStreamer(hf_tokenizer, skip_prompt=True, skip_special_tokens=True)
            inputs = hf_tokenizer(full_prompt_for_hf, return_tensors="pt").to(device)
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": llm_tts_config.llm_max_length,
                "temperature": llm_tts_config.llm_temperature,
                "top_p": llm_tts_config.llm_top_p,
                "repetition_penalty": llm_tts_config.llm_repetition_penalty,
                "do_sample": True,
                "streamer": streamer,
                "pad_token_id": hf_tokenizer.eos_token_id
            }
            thread = threading.Thread(target=hf_model.generate, kwargs=generation_kwargs)
            thread.start()

            # Common chunk processing logic for both providers
            for new_text in streamer: # For HF, streamer yields. For OpenAI, we iterate above.
                full_text += new_text
                buffer += new_text
                # decide split
                send = False
                if any(bp in buffer for bp in priority_breakpoints) and buffer.count(" ") >= 3:
                    idx = max(buffer.rfind(bp) for bp in priority_breakpoints if bp in buffer) # Ensure bp is in buffer
                    send = True
                    cut = buffer[:idx+1].strip()
                    buffer = buffer[idx+1:].strip()
                elif buffer.count(" ") >= llm_tts_config.chunk_word_threshold:
                    idx = buffer.rfind(" ")
                    cut = buffer[:idx].strip() if idx != -1 else buffer.strip()
                    buffer = buffer[idx:].strip() if idx != -1 else ""
                    send = True
                if send and cut:
                    chunk_id_counter += 1
                    text_queue.put((cut, chunk_id_counter))
            if server_llm_provider != "openai": # Join thread for HF
                thread.join()

        # After loop (either OpenAI or HF)
        full_text = ""
        buffer = ""
        if buffer.strip():
            chunk_id_counter += 1
            text_queue.put((buffer.strip(), chunk_id_counter))
        text_queue.put((None, None))
        
        # Log AI speech
        scb_store.append({"type": "speech", "actor": "vtuber", "text": full_text})
        return full_text
    except Exception as e:
        print(f"\n{ColorText.RED}[ERROR] Error in LLM streaming: {e}{ColorText.END}")
        import traceback; print(traceback.format_exc())
        text_queue.put((None, None))
        return ""

def tts_blendshape_worker(model, tokenizer, sample_rate, device):
    """Worker function that converts text chunks to speech and generates blendshapes"""
    print(f"{ColorText.BOLD}[TTS+Blendshapes]{ColorText.END} Worker started...")
    try:
        all_audio_chunks = []
        all_blendshapes = []
        librosa = None # Lazy load librosa

        while True:
            text_chunk, chunk_id = text_queue.get()
            if text_chunk is None: # End signal
                break
            if not text_chunk.strip():
                continue

            print(f"{ColorText.BOLD}[TTS]{ColorText.END} Processing chunk {chunk_id}: '{text_chunk}'")
            try:
                inputs = tokenizer(text_chunk, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model(**inputs).waveform

                audio = output.squeeze().detach().cpu().numpy()

                # Apply speed adjustment if needed
                if llm_tts_config.tts_speed != 1.0:
                    if librosa is None: # Lazy load librosa
                        import librosa
                    audio = librosa.effects.time_stretch(audio, rate=llm_tts_config.tts_speed)

                audio_length_seconds = len(audio) / sample_rate
                print(f"{ColorText.BOLD}[TTS]{ColorText.END} Generated {audio_length_seconds:.2f}s audio for chunk {chunk_id}")

                all_audio_chunks.append(audio) # Store float audio for potential full file later

                # Convert to int16 for blendshape generation and playback
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_bytes_io = BytesIO()
                wav.write(audio_bytes_io, rate=sample_rate, data=audio_int16)
                wav_formatted_bytes = audio_bytes_io.getvalue() # Get the WAV formatted bytes

                # Generate blendshapes
                # Use the imported function directly
                blendshapes = generate_facial_data_from_bytes(
                    wav_formatted_bytes, blendshape_model, device, config # Pass WAV bytes
                )
                blendshapes_list = blendshapes.tolist() if isinstance(blendshapes, np.ndarray) else blendshapes
                all_blendshapes.extend(blendshapes_list)

                # Put WAV formatted audio (bytes) and blendshapes (list) in queue
                audio_blendshape_queue.put((wav_formatted_bytes, blendshapes_list, chunk_id))
                print(f"{ColorText.BOLD}[Blendshapes]{ColorText.END} Generated {len(blendshapes_list)} frames for chunk {chunk_id}")

            except Exception as e:
                print(f"{ColorText.RED}[ERROR] Error processing chunk {chunk_id}: {e}{ColorText.END}")
                import traceback
                print(f"{ColorText.RED}Traceback:\n{traceback.format_exc()}{ColorText.END}")
                # Put empty data to maintain stream flow if needed by client
                empty_audio = np.zeros(int(sample_rate * 0.1), dtype=np.int16) # Short silence
                audio_blendshape_queue.put((empty_audio.tobytes(), [], chunk_id))

        # Signal end of processing
        audio_blendshape_queue.put((None, None, None))
        print(f"{ColorText.BOLD}[TTS+Blendshapes]{ColorText.END} Worker finished.")

        # Prepare final combined results (though maybe not used directly by streaming endpoint)
        fps = 60
        complete_sequence = BlendshapeSequence(fps=fps, sr=sample_rate, frames=all_blendshapes)
        if all_audio_chunks:
            full_audio = np.concatenate(all_audio_chunks)
            full_audio_int16 = (full_audio * 32767).astype(np.int16)
            return full_audio_int16, complete_sequence, sample_rate
        else:
            return None, BlendshapeSequence(fps=fps, sr=sample_rate, frames=[]), sample_rate

    except Exception as e:
        print(f"\n{ColorText.RED}[ERROR] Error in TTS/Blendshape worker: {e}{ColorText.END}")
        import traceback
        print(f"{ColorText.RED}Traceback:\n{traceback.format_exc()}{ColorText.END}")
        audio_blendshape_queue.put((None, None, None)) # Ensure stop signal is sent
        # Determine sample rate even on error if possible
        sr = pipeline_models.get("sample_rate", 16000) # Default if model didn't load
        return None, BlendshapeSequence(fps=60, sr=sr, frames=[]), sr


def audio_playback_worker(playback_queue, sample_rate, device_index=None):
    """Worker function to play audio chunks from a queue using sounddevice."""
    stream = None
    try:
        if device_index is None or device_index == "auto":
             try:
                 print(f"{ColorText.PURPLE}[PlaybackWorker]{ColorText.END} Available audio output devices:")
                 print(sd.query_devices())
                 default_output_device = sd.default.device[1] # Index 1 = output
                 if default_output_device == -1:
                      print(f"{ColorText.YELLOW}⚠️ No default audio output device found. Playback disabled.{ColorText.END}")
                      device_index = None # Disable playback
                 else:
                      device_index = default_output_device
                      print(f"{ColorText.PURPLE}[PlaybackWorker]{ColorText.END} Using default output device index: {device_index}")
             except Exception as e:
                 print(f"{ColorText.RED}[PlaybackWorker] Error querying audio devices: {e}. Playback disabled.{ColorText.END}")
                 device_index = None # Disable playback

        if device_index is not None:
            print(f"{ColorText.PURPLE}[PlaybackWorker]{ColorText.END} Initializing stream. Sample Rate: {sample_rate}, Device: {device_index}")
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='int16', # Expecting int16 bytes
                device=device_index
            )
            stream.start()
            print(f"{ColorText.PURPLE}[PlaybackWorker]{ColorText.END} Stream started.")
        else:
             print(f"{ColorText.YELLOW}[PlaybackWorker] No output device selected. Audio playback will be skipped.{ColorText.END}")


        while True:
            audio_chunk_wav_bytes = playback_queue.get() # Now expects WAV formatted bytes
            if audio_chunk_wav_bytes is None: # End signal
                print(f"{ColorText.PURPLE}[PlaybackWorker]{ColorText.END} Received stop signal.")
                break

            if stream and len(audio_chunk_wav_bytes) > 0:
                try:
                    # Convert WAV bytes back to numpy array for sounddevice
                    # The 'sf' module (soundfile) should be imported at the top of the file
                    audio_data, read_samplerate = sf.read(BytesIO(audio_chunk_wav_bytes), dtype='int16')
                    
                    # Basic check for samplerate consistency, though ideally they should always match
                    if read_samplerate != sample_rate:
                        print(f"{ColorText.YELLOW}[PlaybackWorker] Warning: Samplerate mismatch. Expected {sample_rate}, got {read_samplerate}. Playback might be affected.{ColorText.END}")

                    # Ensure audio_data is 1D if stream is mono (sounddevice expects this)
                    if audio_data.ndim > 1 and stream.channels == 1:
                        audio_data = audio_data[:, 0] # Take first channel
                    elif audio_data.ndim == 1 and stream.channels > 1:
                        # If stream is stereo but data is mono, duplicate mono to stereo (optional, depends on desired behavior)
                        # For now, we assume if stream.channels > 1, sounddevice handles mono data correctly or audio_data is already stereo.
                        pass 

                    stream.write(audio_data)
                    # print(f"Played chunk: {len(audio_data)} samples") # Verbose
                except Exception as e:
                    print(f"{ColorText.RED}[PlaybackWorker] Error playing audio chunk: {e}{ColorText.END}")
            elif not stream:
                # If stream isn't initialized, just consume the queue items
                pass # Skip playback
            # else: # Empty chunk, do nothing
            #     print(f"{ColorText.YELLOW}[PlaybackWorker] Received empty audio chunk. Skipping.{ColorText.END}") # Debug

            playback_queue.task_done()

    except Exception as e:
        print(f"{ColorText.RED}[PlaybackWorker] Error: {e}{ColorText.END}")
        import traceback
        print(f"{ColorText.RED}Traceback:\n{traceback.format_exc()}{ColorText.END}")
    finally:
        if stream:
            try:
                print(f"{ColorText.PURPLE}[PlaybackWorker]{ColorText.END} Stopping and closing audio stream...")
                stream.stop()
                stream.close()
                print(f"{ColorText.PURPLE}[PlaybackWorker]{ColorText.END} Stream stopped and closed.")
            except Exception as e:
                print(f"{ColorText.RED}[PlaybackWorker] Error closing stream: {e}{ColorText.END}")
        # Ensure the final None is marked done if loop exited unexpectedly
        try:
            if not playback_queue.empty() and playback_queue.get(block=False) is None:
                playback_queue.task_done()
        except queue.Empty:
            pass
        except Exception as qe:
             print(f"{ColorText.RED}[PlaybackWorker] Error clearing queue on exit: {qe}{ColorText.END}")
        print(f"{ColorText.PURPLE}[PlaybackWorker]{ColorText.END} Worker thread finished.")


# --- Flask Routes ---

@app.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    """
    Generates blendshapes from raw WAV audio bytes or base64 encoded audio.
    Plays the resulting animation via LiveLink.
    """
    try:
        content_type = request.headers.get('Content-Type', '').lower()
        audio_bytes = None

        if 'application/json' in content_type:
            data = request.json
            if not data or 'audio' not in data:
                return jsonify({"error": "Missing 'audio' field in JSON body"}), 400
            try:
                audio_bytes = base64.b64decode(data['audio'])
            except Exception as e:
                return jsonify({"error": f"Failed to decode base64 audio: {str(e)}"}), 400
        elif 'audio/wav' in content_type or 'application/octet-stream' in content_type:
            audio_bytes = request.data
        else:
             return jsonify({"error": "Unsupported Content-Type. Use application/json (with base64 'audio') or audio/wav."}), 415


        if not audio_bytes:
            return jsonify({"error": "No audio data provided"}), 400

        # Use the imported function
        # Assume default sample rate (16000) and fps (60) for now, as the model expects this
        # The function should ideally return these if they can vary
        sr = 16000
        fps = 60
        generated_facial_data = generate_facial_data_from_bytes(
            audio_bytes, blendshape_model, device, config
        )
        generated_facial_data_list = generated_facial_data.tolist() if isinstance(generated_facial_data, np.ndarray) else generated_facial_data

        sequence = BlendshapeSequence(fps=fps, sr=sr, frames=generated_facial_data_list)

        # Send to LiveLink using the Player
        try:
            print(f"{ColorText.BLUE}[API /audio_to_blendshapes]{ColorText.END} Stopping default animation...")
            stop_default_animation_event.set() # Use the imported event
            player = Player(py_face, socket_connection, stop_default_animation_event) # Pass event
            print(f"{ColorText.BLUE}[API /audio_to_blendshapes]{ColorText.END} Playing animation...")
            player.play(audio_bytes, sequence) # Player handles restarting default animation
            print(f"{ColorText.BLUE}[API /audio_to_blendshapes]{ColorText.END} Playback finished.")
        except Exception as e:
            print(f"{ColorText.RED}[API /audio_to_blendshapes] Error sending blendshapes to LiveLink: {e}{ColorText.END}")
            # Ensure default animation restarts even on error
            stop_default_animation_event.clear()

        return jsonify({
            'sr': sequence.sr,
            'fps': sequence.fps,
            'blendshapes': sequence.frames
        })

    except Exception as e:
        print(f"{ColorText.RED}[API /audio_to_blendshapes] Error: {e}{ColorText.END}")
        import traceback
        print(f"{ColorText.RED}Traceback:\n{traceback.format_exc()}{ColorText.END}")
        return jsonify({"error": str(e)}), 500

@app.route('/text_to_blendshapes', methods=['POST'])
def text_to_blendshapes_route():
    """
    Generates speech audio and corresponding blendshapes from text.
    Plays the resulting animation via LiveLink.
    Returns the full audio (base64) and blendshapes.
    """
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' field in request body"}), 400
        prompt = data['prompt']
        print(f"{ColorText.BLUE}[API /text_to_blendshapes]{ColorText.END} Received request: '{prompt[:50]}...'")

        # Clear queues
        while not text_queue.empty(): text_queue.get()
        while not audio_blendshape_queue.empty(): audio_blendshape_queue.get()
        while not playback_audio_queue.empty(): playback_audio_queue.get() # Clear playback queue too

        # Load models (ensures they are ready)
        models = load_llm_tts_models()
        if not models: # Handle loading failure
             return jsonify({"error": "Failed to load necessary models."}), 500
        sample_rate = models["sample_rate"]

        # Start workers
        llm_thread = threading.Thread(
            target=llm_streaming_worker,
            args=(models, prompt, device), # Pass the whole models dictionary
            daemon=True
        )
        tts_thread = threading.Thread(
            target=tts_blendshape_worker,
            args=(models["tts_model"], models["tts_tokenizer"], sample_rate, device),
            daemon=True
        )
        llm_thread.start()
        tts_thread.start()

        # Wait for workers and collect results
        llm_thread.join()
        full_text_response = "" # Capture full text if needed, though not returned here
        # Result of tts_worker captures combined audio/sequence
        # Use the values returned by the tts_blendshape_worker function
        # The worker now returns (full_audio_int16, complete_sequence, sample_rate)
        tts_thread.join() # Wait for TTS worker to finish
        # Instead of getting from queue, get the return value from the worker function
        # This requires modifying how tts_blendshape_worker returns values
        # Let's assume tts_blendshape_worker is modified to return its results
        # For now, we'll need to rethink how to get the final combined result
        # Let's collect from the queue as it was originally designed for non-streaming

        full_audio_list = []
        all_blendshapes = []
        while True:
            try:
                 audio_chunk_bytes, blendshapes_list, chunk_id = audio_blendshape_queue.get(timeout=0.1) # Short timeout
                 if audio_chunk_bytes is None: # End signal
                      break
                 full_audio_list.append(audio_chunk_bytes)
                 if blendshapes_list:
                      all_blendshapes.extend(blendshapes_list)
            except queue.Empty:
                 # If the queue is empty and the worker is done, we break
                 if not tts_thread.is_alive() and audio_blendshape_queue.empty():
                      break
                 # Otherwise, keep waiting briefly
                 continue

        if not full_audio_list:
            print(f"{ColorText.YELLOW}[API /text_to_blendshapes] No audio generated.{ColorText.END}")
            return jsonify({"audio": "", "sr": sample_rate, "fps": 60, "blendshapes": []})

        full_audio_bytes = b''.join(full_audio_list)
        complete_sequence = BlendshapeSequence(fps=60, sr=sample_rate, frames=all_blendshapes)

        print(f"{ColorText.BLUE}[API /text_to_blendshapes]{ColorText.END} Generated {len(full_audio_bytes)} audio bytes, {len(complete_sequence.frames)} blendshape frames.")

        # Send to LiveLink using Player
        try:
            print(f"{ColorText.BLUE}[API /text_to_blendshapes]{ColorText.END} Stopping default animation...")
            stop_default_animation_event.set()
            player = Player(py_face, socket_connection, stop_default_animation_event)
            print(f"{ColorText.BLUE}[API /text_to_blendshapes]{ColorText.END} Playing animation...")
            player.play(full_audio_bytes, complete_sequence)
            print(f"{ColorText.BLUE}[API /text_to_blendshapes]{ColorText.END} Playback finished.")
        except Exception as e:
            print(f"{ColorText.RED}[API /text_to_blendshapes] Error sending to LiveLink: {e}{ColorText.END}")
            stop_default_animation_event.clear() # Ensure restart on error

        # Encode audio for JSON response
        encoded_audio = base64.b64encode(full_audio_bytes).decode('utf-8')

        return jsonify({
            # "text": full_text_response, # Optionally return full LLM text
            "audio": encoded_audio,
            "sr": complete_sequence.sr,
            "fps": complete_sequence.fps,
            "blendshapes": complete_sequence.frames
        })

    except Exception as e:
        print(f"{ColorText.RED}[API /text_to_blendshapes] Error: {e}{ColorText.END}")
        import traceback
        print(f"{ColorText.RED}Traceback:\n{traceback.format_exc()}{ColorText.END}")
        # Ensure default animation is restarted on error
        if 'stop_default_animation_event' in locals() and stop_default_animation_event:
             stop_default_animation_event.clear()
        return jsonify({"error": str(e)}), 500


@app.route('/stream_text_to_blendshapes', methods=['POST'])
def stream_text_to_blendshapes_route():
    """
    Streams text-to-speech and blendshapes generation.
    Each chunk contains base64 audio and blendshape frames.
    Simultaneously streams blendshapes to LiveLink and plays audio locally.
    """
    def generate():
        stream_start_time = time.time()
        print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Received request.")
        player = None # Initialize player for potential use
        llm_thread = None
        tts_thread = None

        try:
            data = request.json
            if not data or 'prompt' not in data:
                error_msg = "Missing 'prompt' field in request body"
                print(f"{ColorText.RED}[API /stream] {error_msg}{ColorText.END}")
                yield json.dumps({"error": error_msg}) + "\n"
                return
            prompt = data['prompt']
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Processing prompt: '{prompt[:50]}...'")

            # Clear queues before starting
            while not text_queue.empty(): text_queue.get()
            while not audio_blendshape_queue.empty(): audio_blendshape_queue.get()
            while not playback_audio_queue.empty(): playback_audio_queue.get()

            # Load models
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Loading models...")
            models = load_llm_tts_models()
            if not models:
                 error_msg = "Failed to load necessary models."
                 print(f"{ColorText.RED}[API /stream] {error_msg}{ColorText.END}")
                 yield json.dumps({"error": error_msg}) + "\n"
                 return
            sample_rate = models["sample_rate"]
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Models loaded. Sample rate: {sample_rate}")

            # Start workers
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Starting LLM worker...")
            llm_thread = threading.Thread(
                target=llm_streaming_worker,
                args=(models, prompt, device), # Pass the whole models dictionary
                daemon=True
            )
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Starting TTS+Blendshape worker...")
            tts_thread = threading.Thread(
                target=tts_blendshape_worker,
                args=(models["tts_model"], models["tts_tokenizer"], sample_rate, device),
                daemon=True
            )
            llm_thread.start()
            tts_thread.start()

            # Prepare for LiveLink streaming
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Stopping default animation for streaming...")
            stop_default_animation_event.set()
            fps = 60
            frame_duration = 1.0 / fps
            player = Player(py_face, socket_connection, stop_default_animation_event) # Create player instance

            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Starting to process and stream chunks...")
            total_frames_streamed = 0
            total_audio_bytes = 0

            while True:
                try:
                    # Get data from the TTS worker queue
                    audio_bytes, blendshapes_chunk, chunk_id = audio_blendshape_queue.get(timeout=1.0) # Timeout to prevent blocking forever

                    if audio_bytes is None: # End signal from TTS worker
                        print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Received end signal from TTS worker.")
                        break

                    total_audio_bytes += len(audio_bytes)
                    total_frames_streamed += len(blendshapes_chunk)
                    print(f"{ColorText.CYAN}[API /stream Chunk {chunk_id}]{ColorText.END} Received: {len(audio_bytes)} audio bytes, {len(blendshapes_chunk)} frames.")

                    # --- 1. Yield JSON chunk to client ---
                    encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
                    response_chunk = {
                        "chunk_id": chunk_id,
                        "audio": encoded_audio,
                        "sr": sample_rate,
                        "fps": fps,
                        "blendshapes": blendshapes_chunk
                    }
                    yield json.dumps(response_chunk) + "\n"
                    # print(f"[API /stream Chunk {chunk_id}] Yielded JSON to client.") # Debug

                    # --- 2. Queue audio for local playback ---
                    playback_audio_queue.put(audio_bytes)
                    # print(f"[API /stream Chunk {chunk_id}] Queued audio for playback.") # Debug

                    # --- 3. Stream blendshapes to LiveLink ---
                    if blendshapes_chunk:
                        # Apply blinking
                        apply_blink_to_facial_data(blendshapes_chunk, default_animation_data)
                        # Use Player's stream_frames method
                        player.stream_frames(blendshapes_chunk, frame_duration)
                        # print(f"[API /stream Chunk {chunk_id}] Streamed {len(blendshapes_chunk)} frames to LiveLink.") # Debug

                except queue.Empty:
                    # Timeout occurred. Check if workers are done.
                    if not llm_thread.is_alive() and not tts_thread.is_alive() and audio_blendshape_queue.empty():
                        print(f"{ColorText.YELLOW}[API /stream]{ColorText.END} Workers finished and queue empty. Exiting loop.")
                        break
                    else:
                        # print("[API /stream] Queue empty, workers still active. Continuing...") # Debug
                        continue # Continue waiting for chunks
                except Exception as loop_err:
                     print(f"{ColorText.RED}[API /stream] Error in chunk processing loop: {loop_err}{ColorText.END}")
                     import traceback
                     print(f"{ColorText.RED}Traceback:\n{traceback.format_exc()}{ColorText.END}")
                     # Maybe yield an error chunk? For now, continue if possible.
                     continue

            # --- Stream finished ---
            stream_duration = time.time() - stream_start_time
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Finished processing all chunks.")
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Total audio bytes: {total_audio_bytes}, Total frames streamed: {total_frames_streamed}")
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Stream duration: {stream_duration:.2f}s")


            # Ensure workers are joined
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Waiting for worker threads to finish...")
            if llm_thread: llm_thread.join(timeout=5)
            if tts_thread: tts_thread.join(timeout=5)
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Worker threads joined.")

            # Signal playback worker to finish *after* all audio chunks have been queued
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Signaling playback worker to stop.")
            playback_audio_queue.put(None)

            # Restart default animation (Player handles this internally now)
            # player.ensure_default_animation_restarted() # Let player handle this
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Default animation restart triggered by Player.")


            # Send final status to client
            yield json.dumps({"status": "completed"}) + "\n"
            print(f"{ColorText.GREEN}[API /stream]{ColorText.END} Sent completion status to client.")


        except Exception as e:
            error_msg = f"Unexpected error in generate(): {str(e)}"
            print(f"{ColorText.RED}[API /stream] {error_msg}{ColorText.END}")
            import traceback
            print(f"{ColorText.RED}Traceback:\n{traceback.format_exc()}{ColorText.END}")
            try:
                yield json.dumps({"error": error_msg, "status": "failed"}) + "\n"
            except Exception as yield_err:
                 print(f"{ColorText.RED}[API /stream] Error yielding final error message: {yield_err}{ColorText.END}")

            # Ensure cleanup even on error
            if player:
                player.ensure_default_animation_restarted() # Try to restart default anim
            # Signal playback queue to stop if it hasn't already
            playback_audio_queue.put(None)
             # Attempt to join threads if they were started
            if llm_thread and llm_thread.is_alive(): llm_thread.join(timeout=1)
            if tts_thread and tts_thread.is_alive(): tts_thread.join(timeout=1)


    # Return the streaming response
    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')


# --- Main Execution ---

def main():
    """Starts the Flask server and background threads."""
    global default_animation_thread # Allow modification
    playback_thread = None
    try:
        # --- Determine Sample Rate ---
        # Try loading models briefly to get sample rate, or use default
        sample_rate = 16000 # Default
        try:
            print(f"{ColorText.CYAN}[Main]{ColorText.END} Attempting to determine TTS sample rate...")
            # Temporarily load just the TTS model config if possible, otherwise load full models
            if not pipeline_models:
                 models = load_llm_tts_models() # Load models now if not loaded
                 if models:
                     sample_rate = models.get("sample_rate", sample_rate)
                 else:
                     print(f"{ColorText.YELLOW}⚠️ Could not load models to determine sample rate. Using default: {sample_rate} Hz.{ColorText.END}")
            else:
                 sample_rate = pipeline_models.get("sample_rate", sample_rate)
            print(f"{ColorText.CYAN}[Main]{ColorText.END} Using sample rate: {sample_rate} Hz")
        except Exception as e:
             print(f"{ColorText.YELLOW}⚠️ Error determining sample rate: {e}. Using default: {sample_rate} Hz.{ColorText.END}")


        # --- Start Default Animation ---
        print(f"{ColorText.GREEN}[Main]{ColorText.END} Starting default animation thread...")
        if not default_animation_thread or not default_animation_thread.is_alive():
            default_animation_thread = threading.Thread(
                target=default_animation_loop,
                args=(py_face, socket_connection), # Pass event
                daemon=True
            )
            default_animation_thread.start()
        else:
             print(f"{ColorText.YELLOW}[Main] Default animation thread already running.{ColorText.END}")

        # --- Start Playback Worker ---
        playback_device = os.getenv("AUDIO_PLAYBACK_DEVICE", "auto") # Allow selecting device via env var
        print(f"{ColorText.GREEN}[Main]{ColorText.END} Starting local audio playback worker (Device: {playback_device})...")
        playback_thread = threading.Thread(
            target=audio_playback_worker,
            args=(playback_audio_queue, sample_rate, playback_device),
            daemon=True
        )
        playback_thread.start()

        # --- Start Summarizer Thread ---
        summarizer_stop = threading.Event()
        summarizer_thread = SummarizerThread(stop_event=summarizer_stop)
        summarizer_thread.start()

        # --- Run Flask App ---
        host = os.getenv("FLASK_HOST", '127.0.0.1')
        port = int(os.getenv("FLASK_PORT", 5000))
        print(f"{ColorText.GREEN}[Main]{ColorText.END} Starting Flask server on {host}:{port}...")
        app.run(host=host, port=port, threaded=True) # Use threaded=True for handling concurrent requests

    except Exception as e:
         print(f"{ColorText.RED}[Main] Unhandled exception during startup or runtime: {e}{ColorText.END}")
         import traceback
         print(f"{ColorText.RED}Traceback:\n{traceback.format_exc()}{ColorText.END}")

    finally:
        # --- Cleanup ---
        print(f"\n{ColorText.YELLOW}[Main] Shutting down server...{ColorText.END}")

        # Stop default animation
        print(f"{ColorText.YELLOW}[Main] Stopping default animation...{ColorText.END}")
        stop_default_animation_event.set()
        if default_animation_thread and default_animation_thread.is_alive():
            try:
                default_animation_thread.join(timeout=2)
                if default_animation_thread.is_alive(): print(f"{ColorText.YELLOW}⚠️ Default animation thread did not exit cleanly.{ColorText.END}")
            except Exception as e:
                print(f"{ColorText.RED}Error joining default animation thread: {e}{ColorText.END}")

        # Stop playback worker
        print(f"{ColorText.YELLOW}[Main] Signaling playback worker to stop...{ColorText.END}")
        playback_audio_queue.put(None)
        if playback_thread and playback_thread.is_alive():
            try:
                playback_thread.join(timeout=5)
                if playback_thread.is_alive(): print(f"{ColorText.YELLOW}⚠️ Playback worker thread did not exit cleanly.{ColorText.END}")
            except Exception as e:
                print(f"{ColorText.RED}Error joining playback worker thread: {e}{ColorText.END}")

        # Clear queues (as a final measure)
        while not text_queue.empty(): text_queue.get()
        while not audio_blendshape_queue.empty(): audio_blendshape_queue.get()
        while not playback_audio_queue.empty(): playback_audio_queue.get()


        # Close socket
        if socket_connection:
             try:
                print(f"{ColorText.YELLOW}[Main] Closing LiveLink socket connection...{ColorText.END}")
                socket_connection.close()
                print(f"{ColorText.GREEN}Socket connection closed.{ColorText.END}")
             except Exception as e:
                print(f"{ColorText.RED}Error closing socket connection: {e}{ColorText.END}")

        # Stop summarizer
        summarizer_stop.set()
        if summarizer_thread.is_alive():
            summarizer_thread.join(timeout=3)

        print(f"{ColorText.YELLOW}[Main] Shutdown complete.{ColorText.END}")


if __name__ == '__main__':
    main() 