# Real-Time Transcription v2.0

Version 2.0 of [real-time-transcription](https://github.com/Davide-Bonn/real-time-transcription) -- a complete rewrite of the speech recognition and transcription pipeline with significantly improved accuracy and robustness.

## What's New in v2.0

This version builds on the original real-time transcription system with 18 key improvements:

**Audio Preprocessing Pipeline**
- High-pass filter (80Hz cutoff) to remove rumble, HVAC noise, and vibrations
- Spectral gating noise reduction
- Pre-emphasis filter to boost consonant clarity
- Dynamic range compression
- VAD-precise trimming (extracts speech-only portions)
- Peak normalization (95%) and silence padding (300ms)

**Improved Transcription**
- Explicit language setting
- Context chaining via `initial_prompt` from previous segments
- Hallucination silence threshold + word timestamps
- Repetition penalty and n-gram suppression
- Post-processing: hallucination filtering, capitalization, whitespace cleanup
- Minimum segment duration filter (skips segments < 0.5s)
- Segment overlap (0.5s from previous segment prepended for continuity)
- Model warm-up at startup for consistent first-segment performance

**Better Speech Detection**
- Dynamic silence threshold (grows with speech duration: 0.3s - 1.5s)
- Tuned VAD parameters: `speech_pad_ms=150`, `min_speech_duration_ms=250`, `min_silence_duration_ms=100`
- Maximum speech duration cap (30s) to prevent runaway segments
- Single ordered transcription worker queue (context-aware)

**Live Terminal Display**
- Color-coded state indicator (IDLE / LISTENING / RECORDING / PROCESSING)
- Real-time transcript rendering with word wrap
- Segment count and audio duration stats

## Requirements

- Python 3.9+
- CUDA 12.1 + cuDNN 8.9.7 (recommended for GPU acceleration)
- FFmpeg

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Davide-Bonn/real-time-transcription-v2.git
   cd real-time-transcription-v2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg (Windows via Chocolatey):
   ```bash
   choco install ffmpeg
   ```

## Usage

```bash
python rec_speak_trans.py
```

Press `Ctrl+C` to stop. On exit, the system:
- Waits for pending transcriptions to complete
- Combines all audio segments into `combined_recording.wav`
- Prints the final transcript
- Saves the transcript to `transcripts/transcript_N.txt`

## Directory Structure

```
recordings/          # Individual audio segments (.wav)
transcripts/         # Generated transcript files (.txt)
combined_recording.wav  # Merged audio on exit
```

## Credits

- [Silero VAD](https://github.com/snakers4/silero-vad) -- Voice activity detection
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) -- Real-time transcription
- [noisereduce](https://github.com/timsainb/noisereduce) -- Spectral gating noise reduction
- [pydub](https://github.com/jiaaro/pydub) -- Audio file processing

## License

Creative Commons NonCommercial (CC BY-NC)
