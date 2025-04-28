## NeuroSync-Core – Runtime Architecture Overview

### 1. High-level data-flow

```
+-------------------+     text prompt      +--------------------+
|   neurosync_client|  ----------------->  |   LLM (remote or   |
|   (CLI, desktop)  |                      |     local API)     |
+-------------------+  <-----------------  +--------------------+
        |  audio + BS request                         |
        |                                             |
        | REST /audio_to_blendshapes (JSON)           |
        v                                             v
+-------------------+   audio & blendshape   +------------------+
| neurosync_local_  |  ------------------->  |  Unreal LiveLink |
| api (Flask)       | (Player: audio + UDP) |  (UDP 11111)     |
+-------------------+                       +------------------+
```

### 2. File-by-file responsibilities

| File | Purpose |
|------|---------|
| **`neurosync_client.py`** | CLI that talks to LLM and TTS (ElevenLabs/local).  Generates WAV, then **forwards** the bytes to `/audio_to_blendshapes`.  If the Flask server is down it falls back to local audio-only playback.  No longer streams blend-shapes itself. |
| **`neurosync_local_api.py`** | Flask server that offers three endpoints: `/audio_to_blendshapes` (wav→blendshape), `/text_to_blendshapes` (prompt→full audio), and `/stream_text_to_blendshapes` (NDJSON streaming).  Internally: 1) LLM stream → text_queue; 2) TTS → audio + blendshapes; 3) `Player` sends sample-accurate audio+UDP frames and local loud-speaker playback. |
| **`livelink/connect/livelink_init.py`** | Utility to make a UDP socket to Unreal LiveLink and to instantiate a zeroed `PyLiveLinkFace`. |
| **`livelink/connect/pylivelinkface.py`** | Minimal re-implementation of Apple ARKit LiveLink packet encoder. Holds 61 blendshape slots, supports per-section scaling via `dimension_scalars.scale_blendshapes_by_section`. |
| **`livelink/connect/faceblendshapes.py`** | Enum mapping ARKit blendshape names to indices (0-67). |
| **`livelink/connect/dimension_scalars.py`** | Region-based scaling helper applied in `PyLiveLinkFace.encode`. Allows dampening/boosting certain face areas. |
| **`livelink/send_to_unreal.py`** | Helpers to pre-blend, blink-apply, pre-encode facial frames and to stream them with sample-accurate timing (`send_sequence_to_unreal`). |
| **`utils/generated_runners.py`** | `Player` class that coordinates audio playback thread (`play_audio_from_memory`) + UDP streaming thread (via functions above) while pausing default-idle animation. Also legacy wrapper `run_audio_animation`. |
| **`utils/audio_face_workers.py`** | Background queue worker used by other scripts (not by client/api) for batch WAV processing + facial animation. Probably legacy. |
| **`utils/generate_face_shapes.py`** | Thin wrapper: bytes → feature extraction → model inference → np.array of 68/61 blendshape frames. |
| **`utils/config.py`** | Centralised env-driven configuration (model parameters, provider URLs, feature flags). |

### 3. Currently used vs. unused

Used at runtime:
* neurosync_client.py (desktop)
* neurosync_local_api.py (server)
* livelink/connect/** (all four helper modules)
* livelink/send_to_unreal.py
* utils/generated_runners.py (via Flask)
* utils/generate_face_shapes.py
* utils/config.py

Potentially unused/legacy paths:
* utils/audio_face_workers.py – only referenced by some batch scripts, not by current client/api. Could be removed or merged.
* Legacy functions inside `generated_runners.py` (e.g. `run_audio_animation`) still called by legacy worker above; if we delete the worker, we can drop the wrapper.
* Some blink-blend helpers duplicated across `default_animation.py` and `send_to_unreal.py`.

### 4. Refactor roadmap

1. **Package structure** – turn scatter of utils/ livelink/ scripts/ into installable package `neurosync` with clear sub-modules.
2. **Configuration**
   * Consolidate env+defaults in `utils/config.py`, inject via `pydantic.BaseSettings` or similar.
3. **Playback API**
   * `Player` is the canonical runtime; move it to `neurosync.runtime.player` and make both client & server import it.
   * Remove `audio_face_workers.py` and the legacy wrapper once confirmed unused.
4. **Networking**
   * REST endpoints and client hops could be replaced by a WebSocket or gRPC stream; would remove file forwarding latency.
5. **Testing**
   * Add unit tests for `dimension_scalars` scaling logic and timing accuracy in `send_sequence_to_unreal` (mock socket).
6. **Documentation**
   * Auto-generate API docs (e.g. with FastAPI instead of Flask) for easier client consumption.

### 5. Supporting layers (Audio ⇄ ML)

| Category | Core modules | Purpose |
|----------|--------------|---------|
| **Audio I/O & DSP** | `utils/audio/extraction/*`, `utils/audio/processing/*`, `utils/audio/convert_audio.py`, `utils/audio/play_audio.py`, `utils/audio/save_audio.py` | Load, resample, feature-extract (MFCC + autocorr) and play/convert WAV/PCM.  Extraction pipeline feeds the face-shape model.  Playback helpers are used by `Player`. |
| **Face-shape ML model** | `utils/model/*`, `utils/generate_face_shapes.py` | Custom Transformer seq-to-seq that maps audio features → 68/61 blendshape frames.  Loaded once in Flask. |
| **TTS layer** | `utils/tts/*` | Abstracts ElevenLabs, local Kokoro, or combined Neurosync endpoint.  `TTSService` is used by both client & server. |
| **LLM layer** | `utils/llm/*` | Provides: provider-agnostic `LLMService`, OpenAI/local-llama runner, chat-history helpers, sentence chunker, etc.  Only a subset (service + utils/chat_utils.py) is needed by current runtime. |
| **Vector DB** | `utils/vector_db/*` | Optional semantic memory for chat context.  Not referenced by the client/server paths we reviewed, so **legacy/optional**. |

### 6. Active vs. legacy (extended)

Active at runtime (Happy-path):
```
neurosync_client.py
neurosync_local_api.py
livelink/connect/**
livelink/send_to_unreal.py
utils/model/*, utils/generate_face_shapes.py
utils/tts/tts_service.py (+ provider impls actually chosen)
utils/llm/llm_service.py
utils/audio/* (play_audio.py used by Player; extraction/processing used by face-model)
```

Optional / rarely used:
* `utils/llm/local_api/**` – stand-alone llama server scripts.
* `utils/vector_db/**` – semantic memory; only used by higher-level chat flows, not the realtime pipeline.
* `utils/audio_face_workers.py`, `utils/llm/llm_initialiser.py`, `utils/llm/turn_processing.py` – legacy batch/chat framework.
* Kokoro TTS (`utils/tts/kokoro/*`) – experimental local TTS.

Safe to remove/relocate after confirming no external scripts rely on them.

### 7. Refactor roadmap (extended)

1. **Package split**
   * `neurosync.core`   (runtime Player, model, LiveLink helpers)
   * `neurosync.server` (Flask API, face-model loader)
   * `neurosync.cli`    (client CLI + maybe GUI)
   * `neurosync.audio`, `neurosync.llm`, `neurosync.tts` as sub-packages.
2. **Dependency inversion**
   * Make `Player` accept pluggable audio-player & frame-sender so we can swap Pygame/SoundDevice and UDP.
3. **Config unification** – single `pyproject.toml` or `settings.toml` parsed by pydantic-settings.
4. **Remove legacy** – drop `audio_face_workers.py`, old llama api folders, vector-db unless needed.
5. **Switch to FastAPI** – provides both REST and WebSocket streaming, auto-docs.  Client can then connect over WS (lower latency).
6. **Testing & CI** – pytest with mocks for audio & UDP, GitHub actions.
7. **Packaging** – build wheels / Docker images for `server` and `client`.

### 8. Animation assets & helpers

| Asset / Module | Description | Runtime usage |
|----------------|-------------|---------------|
| `livelink/animations/default_anim/default.csv` | 2.4 MB idle-blink loop recorded on iPhone.  Loaded once by `default_animation.py`, blended into seamless loop, streamed when avatar is idle. | **Active** (idle state) |
| `livelink/animations/<Emotion>/*.csv` (`Angry`, `Disgusted`, `Fearful`, `Happy`, `Neutral`, `Sad`, `Surprised`) | Short emotion-specific mouth/eye performances captured on iPhone (≈1.4 MB each).  Loaded by `animation_loader.py` into `emotion_animations`.  Overlaid by `Player` via `animation_emotion.merge_emotion_data_into_facial_data_wrapper`. | **Active** (overlay when emotion detected) |
| `animation_loader.py` | Reads CSV → numpy, blends start/end, builds the `emotion_animations` dict at import-time. | Active |
| `animation_emotion.py` | Detects dominant emotion vector (last 7 dims of model output) and merges chosen emotion CSV onto facial frames. | Active |
| `blending_anims.py` | Low-level helpers to blend start/end frames and to blend-in/out animations; also contains per-dim looping helpers. | Active |
| `default_animation.py` | Manages idle-loop thread and `stop_default_animation` flag used throughout the runtime. | Active |

CSV schema: first two columns are `Timecode`, `BlendshapeCount` then 61 blendshape floats; some emotion files include extra 7 columns (68 total).

### Animation refactor ideas
1. **Convert CSV → NPZ** to cut disk read time & memory (load with `numpy.load`).
2. **Lazy load** only the emotion folders that are actually requested to speed up cold start.
3. **Move data to S3 or git-lfs**; keep small NPZ stubs in repo.
4. **Pre-encode** emotion sequences using `pre_encode_facial_data` so Player can send them directly without per-frame encode at runtime.
5. **Emotion routing**: expose config to enable/disable automatic overlay; could be off for performance-critical demos.

### 9. System-1 / System-2 bridge (new)
System-1 = fast, reactive chain (current repo).  System-2 = optional, slower reasoning service.
Communication is a simple shared text/JSON file (env `BRIDGE_FILE_PATH`).
Flags in config:
* `ENABLE_SYSTEM2_BRIDGE` – turn feature on/off.
* `MOCK_SYSTEM2` – if true, background thread appends mock insights every 10 s.
* `BRIDGE_DEBUG` – verbose logging when bridge contents change.

Injection points:
* neurosync_client → prepends bridge text to every LLM prompt.
* neurosync_local_api → same inside `llm_streaming_worker`.

When OFF, runtime identical to pre-bridge behaviour.

---
_Last updated by AI assistant – 2025-04-27 (bridge section)_
