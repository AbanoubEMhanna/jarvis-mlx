# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Jarvis-MLX is an offline voice assistant running entirely on Apple Silicon (M-series Macs). It chains three AI models in a real-time audio pipeline:

1. **VAD** → detects when the user is speaking (`stt/VoiceActivityDetection.py`)
2. **STT** → transcribes speech to text using Whisper via MLX (`stt/whisper/`)
3. **LLM** → generates a response via `mlx_lm` with a HuggingFace quantized model
4. **TTS** → synthesises the response using MeloTTS via PyTorch/MPS (`melo/`)

The `Client` class in `main.py` wires these together with threading and a `Queue` for audio chunks.

## Platform Requirement

This project **only runs on Apple Silicon**. MLX is Apple's ML framework for arm64. Confirm the environment before running:

```bash
python -c "import platform; print(platform.processor())"  # must print "arm"
```

Create the correct conda environment if needed:

```bash
CONDA_SUBDIR=osx-arm64 conda create -n native numpy -c conda-forge
conda activate native
```

## Setup and Running

```bash
pip install -r requirements.txt
python main.py
```

Models are downloaded from HuggingFace automatically on first run:
- **Whisper** (STT): `mlx-community/whisper-large-v3-mlx-4bit`
- **LLM**: `mlx-community/Meta-Llama-3-8B-Instruct-4bit` (swap to `mlx-community/Phi-3-mini-4k-instruct-8bit` for lower latency)
- **MeloTTS** (TTS): `myshell-ai/MeloTTS-English-v3`

## Running Tests

```bash
# All tests
pytest

# A single test
pytest test/test_tts.py
pytest test/test_llm.py
```

Tests require network access to download models. `test_tts.py` creates and deletes `en-newest.wav` in the working directory. `.wav` files are gitignored.

## Architecture Notes

### Threading model
`Client.__init__` starts two background threads:
- `vad.startListening()` — reads the mic via PyAudio, calls `onSpeechEnd` with raw `int16` numpy data when speech stops
- `transcription_loop()` — drains `vad_data` Queue, runs STT → LLM → TTS sequentially per utterance

`toggleListening()` pauses mic ingestion and flushes the queue between turns so the assistant doesn't hear itself.

### STT layer (`stt/`)
- `VADDetector` uses `webrtcvad` (mode 3 = most aggressive) at 16 kHz, 10 ms frames. Sensitivity is in seconds of silence before triggering `onSpeechEnd`.
- `FastTranscriber` in `stt/whisper/transcribe.py` wraps `ModelHolder` (singleton pattern) and the `transcribe()` function from `stt/whisper/whisper.py` (Apple's MLX Whisper port).
- Whisper models are downloaded from HuggingFace via `snapshot_download` and loaded as `.npz` weights.

### TTS layer (`melo/`)
This is a stripped-down inference-only copy of [MeloTTS](https://github.com/myshell-ai/MeloTTS). It runs on PyTorch with MPS acceleration. The key entry point is `TTS.tts_to_file()` in `melo/api.py`, which:
1. Splits text into sentences (`melo/split_utils.py`)
2. Converts text → phonemes → BERT embeddings (`melo/text/`, `melo/utils.py`)
3. Runs `SynthesizerTrn` (`melo/models.py`) to produce a float32 numpy audio array
4. Returns the array directly when `output_path=None` (used in `main.py`)

`main.py` trims silence from the output with `librosa.effects.trim` before playing via `sounddevice`.

### LLM layer
Uses `mlx_lm.load` / `mlx_lm.generate` directly. The conversation history is serialised as a ChatML string (`<|user|>...<|end|>`) and the system prompt (`master`) is prepended to every user turn inside `addToHistory()`.

### Changing the LLM model
Swap the model string in `Client.__init__`:
```python
self.model, self.tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-8bit")
```
Any `mlx_lm`-compatible HuggingFace repo works.

### Changing the voice
The default voice is `EN-Newest` (female). Changing it requires training a custom MeloTTS model using the full [MeloTTS repo](https://github.com/myshell-ai/MeloTTS) and pointing `load_or_download_model` to the custom checkpoint.
