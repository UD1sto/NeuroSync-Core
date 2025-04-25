# NeuroSync Local API

> **Note:** This is an extended fork of the original NeuroSync Player + NeuroSync Local API which can be found at [https://github.com/AnimaVR](https://github.com/AnimaVR). All code in this repository falls under the original NeuroSync license.

A Flask-based API to generate blendshapes for 3D facial animation from text or audio input.

## Features

- Convert text to speech with LLM augmentation
- Generate facial blendshapes from audio
- Stream text-to-speech and blendshape generation
- Direct animation of 3D face models via LiveLink connection

## Installation

NeuroSync-Core supports a wide range of hardware configurations, from CPU-only setups to the latest NVIDIA GPUs.

### Automatic Installation (Recommended)

The easiest way to install NeuroSync-Core is to use our automatic installation script:

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuroSync-Core.git
cd NeuroSync-Core

# Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Run the installation script
python install.py
```

The installation script will:
1. Detect your hardware configuration
2. Install the appropriate version of PyTorch based on your GPU/CUDA capabilities
3. Install all other required dependencies
4. Verify the installation

### Manual Installation

If you prefer to install manually or encounter issues with the automatic installation:

#### 1. Install PyTorch with CUDA support

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

#### 2. Install other dependencies

```bash
pip install numpy scipy flask pydub sounddevice transformers librosa
```

## API Endpoints

### `/text_to_blendshapes` (POST)

Unified endpoint that converts text to speech via LLM, generates blendshapes, and automatically sends the animation to the connected 3D model via LiveLink.

Request:
```json
{
    "prompt": "Text to convert to speech and generate blendshapes"
}
```

Response:
```json
{
    "audio": "binary audio data",
    "blendshapes": [[...blendshape values...], [...], ...]
}
```

**Note:** This endpoint now automatically sends the generated blendshapes to the 3D model through LiveLink, so you can see the animation in real-time without additional steps.

### `/audio_to_blendshapes` (POST)

Generate blendshapes from audio and automatically send them to the connected 3D model.

Request: Raw audio bytes (WAV format)

Response:
```json
{
    "blendshapes": [[...blendshape values...], [...], ...]
}
```

### `/stream_text_to_blendshapes` (POST)

Stream the conversion of text to speech and blendshapes. Sends the animation to the 3D model in real-time as chunks are processed.

Request:
```json
{
    "prompt": "Text to convert to speech and generate blendshapes"
}
```

Response: Stream of JSON chunks with audio and blendshapes

## Testing from PowerShell

1. Text-to-blendshapes:
```powershell
$body = @{
    prompt = "Hello, this is a test of the 3D animation system."
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/text_to_blendshapes" -Method Post -Body $body -ContentType "application/json"
```

2. Audio-to-blendshapes:
```powershell
$audioBytes = [System.IO.File]::ReadAllBytes("C:\path\to\sample.wav")
Invoke-RestMethod -Uri "http://127.0.0.1:5000/audio_to_blendshapes" -Method Post -Body $audioBytes -ContentType "audio/wav"
```

## Unified Processing Workflow

The NeuroSync Local API now features a streamlined processing pipeline:

1. **Text Input** → When you send text to the `/text_to_blendshapes` endpoint, the system:
   - Processes your text through a language model (LLM)
   - Converts the processed text to natural speech
   - Generates facial blendshapes from the audio
   - Automatically sends the animation to your 3D model via LiveLink

2. **Real-time Animation** → All endpoints now automatically animate the connected 3D model:
   - No additional steps required to visualize the animation
   - Works with Unreal Engine through the NeuroSync Player and LiveLink
   - Provides smooth, natural facial movements directly from text or audio input

This unified workflow makes it easier than ever to create expressive 3D character animations from simple text prompts or audio files.

## Running the API

```bash
python neurosync_local_api.py
```

This will start the local API server on http://127.0.0.1:5000

## 29/03/2025 Update to model.pth and model.py

- Increased accuracy (timing and overall face shows more natural movement overall, brows, squint, cheeks + mouth shapes)
- More smoothness during playback (flappy mouth be gone in most cases, even when speaking quickly)
- Works better with more voices and styles of speaking.
- This preview of the new model is a modest increase in capability that requires both model.pth and model.py to be replace with the new versions.

[Download the model from Hugging Face](https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape)

These increases in quality come from better data and removal of "global" positional encoding from the model and staying with ropes positional encoding within the MHA block.

## Overview

The **NeuroSync Local API** allows you to host the audio-to-face blendshape transformer model locally. This API processes audio data and outputs facial blendshape coefficients, which can be streamed directly to Unreal Engine using the **NeuroSync Player** and LiveLink.

### Features:
- Host the model locally for full control
- Process audio files and generate facial blendshapes

## NeuroSync Model

To generate the blendshapes, you can:

- [Download the model from Hugging Face](https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape)

## Player Requirement

To stream the generated blendshapes into Unreal Engine, you will need the **NeuroSync Player**. The Player allows for real-time integration with Unreal Engine via LiveLink. 

You can find the NeuroSync Player and instructions on setting it up here:

- [NeuroSync Player GitHub Repository](https://github.com/AnimaVR/NeuroSync_Player)

Visit [neurosync.info](https://neurosync.info)

## Talk to a NeuroSync prototype live on Twitch : [Visit Mai](https://www.twitch.tv/mai_anima_ai)

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related issues:

1. Verify your CUDA installation with `nvidia-smi`
2. Check if PyTorch can detect CUDA:
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. Make sure your GPU drivers are up to date
4. Try installing a different CUDA version of PyTorch

For detailed diagnostics, run:
```bash
python cuda_check.py
```

## Hardware Requirements

- **CPU-only**: Any modern CPU (slower performance)
- **GPU (recommended)**: NVIDIA GPU with CUDA support
- **Memory**: At least 8GB RAM (16GB+ recommended)
- **Storage**: 5GB+ free space for models and application
