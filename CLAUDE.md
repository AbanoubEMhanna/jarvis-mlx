# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Jarvis-MLX is an offline voice assistant for Apple Silicon Macs. It runs a full speech-to-text → LLM → text-to-speech pipeline entirely on-device using Apple's MLX framework. No internet connection is required at runtime.

The pipeline:
```
Microphone → VAD → Whisper STT → Llama/Phi LLM → MeloTTS → Speaker
```

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- Native arm64 Python — **not** Rosetta. Verify with:
  ```bash
  python -c "import platform; print(platform.processor())"
  # must print: arm
  ```
- Create a native conda environment:
  ```bash
  CONDA_SUBDIR=osx-arm64 conda create -n jarvis python numpy -c conda-forge
  conda activate jarvis
  pip install -r requirements.txt
  ```

## Running

```bash
python main.py
```

On startup the app:
1. Loads MeloTTS (`EN_NEWEST`) and downloads model weights if missing
2. Loads Whisper (`mlx-community/whisper-large-v3-mlx-4bit`)
3. Loads the LLM (`mlx-community/Meta-Llama-3-8B-Instruct-4bit`)
4. Plays `beep.mp3` and starts listening

Speak into the mic. After a pause (~0.3 s silence threshold), the utterance is transcribed, sent to the LLM, and the response is spoken aloud before listening resumes.

## Testing

```bash
pytest test/          # run all tests
pytest test/test_tts.py   # TTS pipeline only
pytest test/test_llm.py   # LLM inference only
```

Tests are minimal integration smoke tests — they load the actual models and require an Apple Silicon machine.

## Repository structure

```
.
├── main.py                      # Entry point — Client class, pipeline orchestration
├── requirements.txt             # Python dependencies
├── beep.mp3                     # Audio played when listening mode activates
├── stt/                         # Speech-to-text module
│   ├── VoiceActivityDetection.py  # VAD using webrtcvad
│   └── whisper/                 # Whisper MLX inference
│       ├── transcribe.py        # FastTranscriber class (main interface)
│       ├── whisper.py           # MLX Whisper model definition
│       ├── decoding.py          # Beam search / greedy decoding
│       ├── audio.py             # Audio preprocessing (mel spectrogram)
│       ├── tokenizer.py         # Whisper tokenizer
│       ├── timing.py            # Word-level timestamp alignment
│       ├── torch_whisper.py     # Torch reference model (for weight loading)
│       └── load_models.py       # Model download + loading helpers
├── melo/                        # MeloTTS inference (English-only strip-down)
│   ├── api.py                   # TTS class — main interface used in main.py
│   ├── models.py                # SynthesizerTrn model architecture
│   ├── modules.py               # Core neural network modules
│   ├── attentions.py            # Attention layers
│   ├── transforms.py            # Flow transforms
│   ├── commons.py               # Shared utilities
│   ├── utils.py                 # Config loading, checkpoint handling
│   ├── split_utils.py           # Text splitting for long inputs
│   ├── download_utils.py        # Model weight downloading
│   └── text/                    # Text processing (G2P, phonemization)
└── test/
    ├── test_llm.py              # LLM inference smoke test
    └── test_tts.py              # TTS synthesis smoke test
```

## Architecture

### `Client` class (`main.py`)

Orchestrates everything. Constructed once at startup. Key state:
- `self.vad` — `VADDetector` instance, runs on a dedicated thread
- `self.vad_data` — `Queue` bridging VAD thread → transcription loop
- `self.tts` — `TTS` (MeloTTS)
- `self.stt` — `FastTranscriber` (Whisper)
- `self.model`, `self.tokenizer` — MLX LLM
- `self.history` — list of `ChatMLMessage` (conversation context)
- `self.listening` — bool toggled by `toggleListening()`

**Threading model:**
1. Main thread: initialisation, blocks on `t.start()` for VAD
2. VAD thread (`startListening`): continuous mic capture via PyAudio at 16 kHz, 10 ms frames; calls `onSpeechEnd` when speech ends
3. Transcription loop thread: drains `vad_data` queue → Whisper → LLM → MeloTTS → sounddevice playback

### Speech detection (`stt/VoiceActivityDetection.py`)

`VADDetector` uses `webrtcvad` in mode 3 (most aggressive). Parameters:
- Sample rate: 16 kHz
- Frame size: 10 ms (160 samples)
- Silence threshold: `sensitivity` seconds (default 0.3 s) after last voiced frame triggers `onSpeechEnd`
- Max frame history: 1000 frames (~10 s)

### Transcription (`stt/whisper/transcribe.py`)

`FastTranscriber` wraps the MLX Whisper model. Takes raw int16 numpy audio → returns `{"text": str, ...}`. Import order matters — `FastTranscriber` must be imported after other modules (see comment in `main.py`).

### LLM generation

Uses `mlx_lm.generate`. Conversation is formatted as a flat string with `<|role|>content<|end|>` tokens (ChatML-style). The system prompt is prepended to every user turn:
> "Answer in no more than three sentences. Address them as Sir at all times. Only respond with the dialogue, nothing else."

Response is truncated at `<|assistant|>` and `<|end|>` tokens before speaking.

### TTS (`melo/api.py`)

`TTS(language="EN_NEWEST", device="mps")` — uses MPS (Apple GPU). `tts_to_file()` returns raw audio data. The audio is trimmed with `librosa.effects.trim(top_db=20)` to remove silence, then played via `sounddevice` at 44100 Hz.

## Default models

| Component | Model | Notes |
|-----------|-------|-------|
| LLM | `mlx-community/Meta-Llama-3-8B-Instruct-4bit` | ~4 GB; swap to `mlx-community/Phi-3-mini-4k-instruct-8bit` for lower latency |
| STT | `mlx-community/whisper-large-v3-mlx-4bit` | Downloaded automatically on first run |
| TTS | MeloTTS `EN_NEWEST` | Default voice is female; finetune MeloTTS to change |

All models are downloaded from Hugging Face Hub on first use and cached locally.

## Swapping models

To change the LLM, edit `main.py`:
```python
self.model, self.tokenizer = load("mlx-community/YOUR-MODEL-HERE")
```
Any MLX-compatible model from `mlx-community` on Hugging Face works.

To change the Whisper model, edit the `FastTranscriber` call:
```python
self.stt = FastTranscriber("mlx-community/whisper-large-v3-mlx-4bit")
```

To change TTS voice you must finetune MeloTTS — the pretrained checkpoint only contains the EN-Newest speaker.

## Key conventions

- **arm64 only** — do not run under Rosetta; MLX will fail or silently fall back to CPU.
- **MPS device** — TTS uses `device="mps"`. LLM uses MLX native (not PyTorch).
- **Import order** — `FastTranscriber` must be imported last in `main.py` due to a module initialisation side effect.
- **No `.env` file** — configuration is done by editing constants in `main.py` directly (model paths, system prompt, sensitivity).
- **Audio format** — VAD operates at 16 kHz int16 mono; TTS outputs at 44100 Hz float32.
- **Minimum audio length** — transcription is skipped for utterances shorter than 12 000 samples (~0.75 s at 16 kHz) to avoid hallucinations.
