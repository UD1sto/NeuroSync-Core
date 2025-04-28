# NeuroSync-Core

> **Note:** This is an extended fork of the original NeuroSync Player + NeuroSync Local API which can be found at [https://github.com/AnimaVR](https://github.com/AnimaVR). All code in this repository falls under the original NeuroSync license.

A platform for generating 3D facial animation blendshapes from text or audio input, featuring a modular architecture for flexible integration with various LLM and TTS providers.

## Features

- Modular integration with different LLM providers (OpenAI, Local Llama 3.1/3.2)
- Modular integration with different TTS providers (ElevenLabs, Local TTS, NeuroSync combined service)
- Generate facial blendshapes from audio
- Stream text-to-speech and blendshape generation
- Direct animation of 3D face models via LiveLink connection (requires NeuroSync Player)

## Installation

NeuroSync-Core supports a wide range of hardware configurations, from CPU-only setups to the latest NVIDIA GPUs.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/NeuroSync-Core.git
cd NeuroSync-Core
```

### 2. Create Environment & Install Dependencies
```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Run the automatic hardware detection and PyTorch installation script
# This is useful if you haven't installed PyTorch with the correct CUDA support yet.
# python install.py
```
The `install.py` script can help ensure the correct PyTorch version is installed for your hardware, but installing via `requirements.txt` is the standard method.

### Manual Installation Steps (Alternative)

If you prefer to install manually or encounter issues:

#### a. Install PyTorch with CUDA support

Choose the appropriate command based on your CUDA version (check with `nvidia-smi`):

**For CUDA 12.8 (latest GPUs)**:
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**For CUDA 12.1**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU-only systems**:
```bash
pip install torch torchvision
```

#### b. Install other dependencies

```bash
# Install all other requirements
pip install -r requirements.txt
```

## Core Functionality

NeuroSync-Core provides the underlying services for:
- Text processing via selected LLM.
- Speech synthesis via selected TTS.
- Audio-to-blendshape generation using the NeuroSync model.
- Sending blendshape data over LiveLink.

Specific API endpoints for direct interaction are under development. The primary way to use the system currently is via the client application.

## Testing

Use the provided client application for testing the end-to-end workflow.

## Running the Core Services

To start the backend NeuroSync API service (required for `neurosync` TTS provider and potentially other future local processing):
```bash
python -m neurosync.server.app
```
This service typically runs on http://127.0.0.1:5000 and is needed if you select `neurosync` as the `TTS_PROVIDER` in your `.env` or command-line arguments. For cloud providers like OpenAI and ElevenLabs, running this specific server is not strictly necessary unless you use the `neurosync` TTS option.

## 29/03/2025 Update to model.pth and model.py

- Increased accuracy (timing and overall face shows more natural movement overall, brows, squint, cheeks + mouth shapes)
- More smoothness during playback (flappy mouth be gone in most cases, even when speaking quickly)
- Works better with more voices and styles of speaking.
- This preview of the new model is a modest increase in capability that requires both model.pth and model.py to be replace with the new versions.

[Download the model from Hugging Face](https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape)

These increases in quality come from better data and removal of "global" positional encoding from the model and staying with ropes positional encoding within the MHA block.

## Step-by-Step Guide (OpenAI + ElevenLabs)

This guide walks through the most common setup using cloud services for LLM and TTS, sending animations to the NeuroSync Player.

**Prerequisites:**
- Installation completed (see "Installation" section).
- **NeuroSync Player running:** You *must* have the NeuroSync Player application (.exe) open and running for the facial animations (blendshapes) to be received and displayed on your character.

**Steps:**

1.  **Configure API Keys in `.env`**: 
    *   Copy `example.env` to `.env` if you haven't already.
        ```bash
        # Use 'copy' on Windows
        copy example.env .env
        # Use 'cp' on macOS/Linux
        # cp example.env .env
        ```
    *   Edit the `.env` file and fill in *at least* the following:
        ```dotenv
        OPENAI_API_KEY=your_openai_api_key_here
        ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
        ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id_here # Find this using the utility script below
        
        # Ensure providers are set for this guide
        LLM_PROVIDER=openai
        TTS_PROVIDER=elevenlabs
        ```
    *   *(Optional)* To find your ElevenLabs Voice ID, run the following command after adding your `ELEVENLABS_API_KEY` to `.env`:
        ```bash
        python -m neurosync.utils.tts.getVoicesElevenLabs
        ```
        Copy the desired ID into your `.env` file.

2.  **Ensure NeuroSync Player is Running**: Double-check that the NeuroSync Player application is open.

3.  **(Optional) Start NeuroSync Server**: If you intend to use the `neurosync` TTS provider *instead* of ElevenLabs, you need to start the local server. For *this* guide (using ElevenLabs), this step is **not** required.
    ```bash
    # Only needed if TTS_PROVIDER=neurosync
    # python -m neurosync.server.app 
    ```

4.  **Run the Client Application**: Open your terminal or command prompt, navigate to the `NeuroSync-Core` directory, and run:
    ```bash
    python -m neurosync.cli.client
    ```
    *(Note: Since `LLM_PROVIDER` and `TTS_PROVIDER` are set to `openai` and `elevenlabs` in the `.env` file, you don't need to specify them with `--llm` or `--tts` here, but you could override the `.env` settings if desired.)*

5.  **Send Text via Client**: The client will prompt you to enter text. Type a sentence and press Enter.
    ```
Enter text to process (or type 'quit' to exit):
> Hello world!
    ```

6.  **Observe Animation**: You should hear the audio synthesized by ElevenLabs, and simultaneously see the facial animation on your character in the NeuroSync Player.

That's it! You have successfully processed text through OpenAI, generated speech with ElevenLabs, and animated a character via NeuroSync Core and the NeuroSync Player.

## Running with Docker Compose

The easiest way to run NeuroSync with all its potential local services (like Llama APIs) is with Docker Compose:

```bash
docker-compose up -d
```

## Running the Client

To run the sample command-line client:

```bash
python -m neurosync.cli.client
```

You can specify LLM and TTS providers on the command line, overriding `.env` settings if needed:

```bash
python -m neurosync.cli.client --llm openai --tts elevenlabs
```

Available options:
- `--llm`: `openai`, `llama3_1`, or `llama3_2`
- `--tts`: `elevenlabs`, `local`, or `neurosync`
- `--no-animation`: Disable sending animation data via LiveLink

## Configuration

The system can be configured through environment variables. Key settings include:

### LLM Configuration
- `LLM_PROVIDER`: Which LLM provider to use (`openai`, `llama3_1`, or `llama3_2`). Default: `openai`.
  **Note:** Local LLM options (`llama3_1`, `llama3_2`) are included but have not been fully tested with the latest refactoring as of this update.
- `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI).
- `LLAMA_3_1_ENDPOINT`/`LLAMA_3_2_ENDPOINT`: Endpoints for local Llama models (e.g., `http://localhost:5050/v1`).

### TTS Configuration
- `TTS_PROVIDER`: Which TTS provider to use (`elevenlabs`, `local`, or `neurosync`). Default: `elevenlabs`.
- `ELEVENLABS_API_KEY`: Your ElevenLabs API key (if using ElevenLabs).
- `ELEVENLABS_VOICE_ID`: Voice ID to use with ElevenLabs.
- `LOCAL_TTS_ENDPOINT`: Endpoint for the local TTS service (if using `local`).
- `NEUROSYNC_API_ENDPOINT`: Endpoint for the combined NeuroSync TTS+Blendshapes service (if using `neurosync`). Default: `http://127.0.0.1:5000`.

### Other Configuration
- `LIVELINK_TARGET_IP`: IP address for LiveLink connection. Default: `127.0.0.1`.
- `LIVELINK_TARGET_PORT`: Port for LiveLink connection. Default: `11111`.
- `LIVELINK_SUBJECT_NAME`: Subject name for LiveLink. Default: `NeuroSyncSubject`.

See `example.env` for all available configuration options.

## License

This software is licensed under a dual-license model:
- For individuals and businesses earning under $1M per year, this software is licensed under the MIT License
- Businesses or organizations with annual revenue of $1,000,000 or more must obtain permission to use this software commercially.
