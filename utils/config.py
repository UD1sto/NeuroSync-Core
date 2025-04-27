import os
import warnings
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Model configuration for the facial animation model
config = {
    'sr': 88200,
    'frame_rate': 60,
    'hidden_dim': 1024,
    'n_layers': 8, 
    'num_heads': 16, 
    'dropout': 0.0,
    'output_dim': 68, # if you trained your own, this should also be 61
    'input_dim': 256,
    'frame_size': 128, 
    'use_half_precision': False
}

# Default configuration values
DEFAULT_VOICE_NAME = os.getenv("DEFAULT_VOICE_NAME", "Rachel")
USE_LOCAL_AUDIO = os.getenv("USE_LOCAL_AUDIO", "True").lower() == "true"
USE_COMBINED_ENDPOINT = os.getenv("USE_COMBINED_ENDPOINT", "True").lower() == "true"
ENABLE_EMOTE_CALLS = os.getenv("ENABLE_EMOTE_CALLS", "True").lower() == "true"

# LLM Configuration
BASE_SYSTEM_MESSAGE = """You are Mai, a witty and helpful assistant with a dry sense of humor."""

# TTS Configuration
LOCAL_TTS_URL = os.getenv("LOCAL_TTS_URL", "http://127.0.0.1:5000/tts")

# LLM and TTS Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")

# Bridge / System-2 integration
ENABLE_SYSTEM2_BRIDGE = os.getenv("ENABLE_SYSTEM2_BRIDGE", "False").lower() == "true"
BRIDGE_FILE_PATH = os.getenv("BRIDGE_FILE_PATH", "bridge.txt")
MOCK_SYSTEM2 = os.getenv("MOCK_SYSTEM2", "False").lower() == "true"
BRIDGE_DEBUG = os.getenv("BRIDGE_DEBUG", "False").lower() == "true"

def setup_warnings():
    """Configure warning behavior for the application."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
    warnings.filterwarnings("ignore", message=".*The dataloader.*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python integer might cause.*")

def get_llm_config(system_message=BASE_SYSTEM_MESSAGE):
    """
    Build the LLM configuration dictionary based on environment variables.
    
    Args:
        system_message: The system message to use for the LLM.
        
    Returns:
        dict: Configuration dictionary for LLM services.
    """
    # Default to True unless explicitly set to false
    use_local_llm = os.getenv("USE_LOCAL_LLM", "False").lower() == "true"
    use_streaming = os.getenv("USE_STREAMING", "True").lower() == "true"
    
    config = {
        "USE_LOCAL_LLM": use_local_llm,
        "USE_STREAMING": use_streaming,
        "system_message": system_message
    }
    
    # OpenAI configuration
    if not use_local_llm:
        config.update({
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        })
    # Local LLM configuration
    else:
        llm_provider = os.getenv("LLM_PROVIDER", "llama3_2")
        
        if llm_provider == "llama3_1":
            config.update({
                "LLM_URL": os.getenv("LLAMA_3_1_ENDPOINT", "http://127.0.0.1:5050/generate_llama"),
                "LLM_STREAM_URL": os.getenv("LLAMA_3_1_STREAM_ENDPOINT", "http://127.0.0.1:5050/generate_stream")
            })
        elif llm_provider == "llama3_2":
            config.update({
                "LLM_URL": os.getenv("LLAMA_3_2_ENDPOINT", "http://127.0.0.1:5050/generate_llama"),
                "LLM_STREAM_URL": os.getenv("LLAMA_3_2_STREAM_ENDPOINT", "http://127.0.0.1:5050/generate_stream")
            })
        else:
            # Default URLs if provider not recognized
            config.update({
                "LLM_URL": os.getenv("LLM_URL", "http://127.0.0.1:5050/generate_llama"),
                "LLM_STREAM_URL": os.getenv("LLM_STREAM_URL", "http://127.0.0.1:5050/generate_stream")
            })
    
    return config

def get_tts_config():
    """
    Build the TTS configuration dictionary based on environment variables.
    
    Returns:
        dict: Configuration dictionary for TTS services.
    """
    tts_provider = os.getenv("TTS_PROVIDER", "elevenlabs")
    
    config = {
        "TTS_PROVIDER": tts_provider
    }
    
    # ElevenLabs configuration
    if tts_provider == "elevenlabs":
        config.update({
            "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY", ""),
            "ELEVENLABS_VOICE_ID": os.getenv("ELEVENLABS_VOICE_ID", ""),
            "ELEVENLABS_MODEL_ID": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1")
        })
    # Local TTS configuration
    elif tts_provider == "local":
        config.update({
            "LOCAL_TTS_URL": os.getenv("LOCAL_TTS_URL", "http://127.0.0.1:5000/tts"),
            "LOCAL_TTS_VOICE": os.getenv("LOCAL_TTS_VOICE", "default")
        })
    # Neurosync TTS+Blendshapes configuration
    elif tts_provider == "neurosync":
        config.update({
            "NEUROSYNC_TTS_URL": os.getenv("NEUROSYNC_TTS_URL", "http://127.0.0.1:5000/text_to_blendshapes"),
            "NEUROSYNC_BLENDSHAPES_URL": os.getenv("NEUROSYNC_BLENDSHAPES_URL", "http://127.0.0.1:5000/audio_to_blendshapes"),
            "NEUROSYNC_TTS_VOICE": os.getenv("NEUROSYNC_TTS_VOICE", "default")
        })
    
    return config
