"""
Improved real-time speech recognition and transcription pipeline (v2).

All improvements:
 1. Explicit language setting
 2. initial_prompt context chaining
 3. hallucination_silence_threshold + word_timestamps
 4. Audio normalization (peak to 95%)
 5. Silence padding (300ms before/after)
 6. VAD speech_pad_ms=150, min_speech_duration_ms=250, min_silence_duration_ms=100
 7. repetition_penalty=1.2, no_repeat_ngram_size=3
 8. Post-processing (hallucination filter, capitalization, whitespace)
 9. Minimum segment duration filter (skip < 0.5s)
10. Single transcription worker queue (ordered, context-aware)
11. beam_size=5
12. Model warm-up at startup
13. Noise reduction (spectral gating)
14. High-pass filter (80Hz cutoff)
15. Pre-emphasis filter (boost consonants)
16. Dynamic range compression
17. VAD-precise trimming (extract speech-only portions)
18. Segment overlap (0.5s from previous segment prepended)
"""

import os
import sys
import re
import shutil
import textwrap
import signal
import numpy as np
import sounddevice as sd
import wave
from silero_vad import load_silero_vad, get_speech_timestamps
import torch
import time
from faster_whisper import WhisperModel
import threading
import queue
from pydub import AudioSegment
from scipy.signal import butter, sosfilt
import noisereduce as nr


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

class C:
    """ANSI color codes."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    BCYAN   = "\033[1;36m"
    BGREEN  = "\033[1;32m"
    BYELLOW = "\033[1;33m"


# ---------------------------------------------------------------------------
# Live terminal display
# ---------------------------------------------------------------------------

class LiveTranscriptDisplay:
    """Clean colored live transcript in the terminal."""

    def __init__(self):
        self._lock = threading.Lock()
        self.full_transcript = ""
        self.segment_count = 0
        self.total_duration = 0.0
        self.state = "IDLE"      # IDLE, LISTENING, RECORDING, PROCESSING
        self._last_line_count = 0

    def set_state(self, state):
        with self._lock:
            self.state = state
            self._redraw()

    def add_transcript(self, text, duration):
        with self._lock:
            if text:
                self.full_transcript += " " + text if self.full_transcript else text
            self.segment_count += 1
            self.total_duration += duration
            self._redraw()

    def _redraw(self):
        term_width = shutil.get_terminal_size((100, 24)).columns

        # State indicator
        if self.state == "IDLE":
            state_str = f"{C.DIM}IDLE - waiting for speech{C.RESET}"
        elif self.state == "LISTENING":
            state_str = f"{C.BYELLOW}LISTENING - speech detected{C.RESET}"
        elif self.state == "RECORDING":
            state_str = f"{C.RED}{C.BOLD}RECORDING{C.RESET}"
        elif self.state == "PROCESSING":
            state_str = f"{C.MAGENTA}{C.BOLD}PROCESSING - transcribing...{C.RESET}"
        else:
            state_str = self.state

        # Wrap transcript text
        if self.full_transcript.strip():
            wrapped = textwrap.wrap(self.full_transcript.strip(), width=term_width - 4)
        else:
            wrapped = [f"{C.DIM}(waiting for first transcription...){C.RESET}"]

        # Build output
        lines = []
        lines.append("")
        lines.append(f"{C.BCYAN}{'=' * term_width}{C.RESET}")
        lines.append(
            f"  {state_str}    "
            f"{C.DIM}Segments: {self.segment_count} | "
            f"Audio: {self.total_duration:.1f}s{C.RESET}"
        )
        lines.append(f"{C.CYAN}{'-' * term_width}{C.RESET}")
        for line in wrapped:
            lines.append(f"  {C.BGREEN}{line}{C.RESET}")
        lines.append(f"{C.BCYAN}{'=' * term_width}{C.RESET}")
        lines.append("")

        # Move cursor up to overwrite previous
        if self._last_line_count > 0:
            sys.stdout.write(f"\033[{self._last_line_count}A\033[J")

        output = "\n".join(lines)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
        self._last_line_count = len(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"{C.GREEN}GPU detected: {name}{C.RESET}")
        return "cuda"
    print(f"{C.YELLOW}No CUDA GPU detected, falling back to CPU.{C.RESET}")
    return "cpu"


# ---------------------------------------------------------------------------
# Audio preprocessing pipeline
# ---------------------------------------------------------------------------

def _build_highpass(cutoff=80, sr=16000, order=5):
    nyq = sr / 2.0
    sos = butter(order, cutoff / nyq, btype="high", output="sos")
    return sos

_HIGHPASS_SOS = _build_highpass(cutoff=80, sr=16000, order=5)


def highpass_filter(audio, sos=_HIGHPASS_SOS):
    return sosfilt(sos, audio).astype(np.float32)


def reduce_noise(audio, sr=16000):
    reduced = nr.reduce_noise(
        y=audio, sr=sr, stationary=True,
        prop_decrease=0.75, n_fft=1024, hop_length=256,
    )
    return np.asarray(reduced, dtype=np.float32)


def pre_emphasis(audio, coeff=0.97):
    return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)


def dynamic_range_compress(audio, sr=16000, target_rms=0.1, window_ms=50):
    window_size = int(sr * window_ms / 1000)
    if len(audio) < window_size:
        return audio
    output = np.copy(audio)
    for start in range(0, len(audio), window_size):
        end = min(start + window_size, len(audio))
        chunk = audio[start:end]
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms > 0.001:
            gain = min(target_rms / rms, 5.0)
            output[start:end] = chunk * gain
    return np.clip(output, -1.0, 1.0).astype(np.float32)


def vad_trim(audio, sr=16000):
    vad_model = load_silero_vad()
    tensor = torch.tensor(audio, dtype=torch.float32)
    timestamps = get_speech_timestamps(
        tensor, vad_model, threshold=0.3,
        speech_pad_ms=200, min_speech_duration_ms=100,
    )
    if not timestamps:
        return audio
    gap = np.zeros(int(sr * 0.05), dtype=np.float32)
    parts = []
    for ts in timestamps:
        parts.append(audio[ts["start"]:ts["end"]])
        parts.append(gap)
    if parts:
        parts.pop()
    return np.concatenate(parts).astype(np.float32)


def preprocess_audio(audio, sr=16000):
    audio = highpass_filter(audio)
    audio = reduce_noise(audio, sr=sr)
    audio = pre_emphasis(audio)
    audio = dynamic_range_compress(audio, sr=sr)
    audio = vad_trim(audio, sr=sr)
    return audio


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

HALLUCINATION_PATTERNS = [
    r"(?i)thank you (for watching|for listening|so much)",
    r"(?i)subscribe to (my|our|the) channel",
    r"(?i)please (like|share) (and|&) subscribe",
    r"(?i)see you (in the|next) (next|video)",
    r"(?i)^\s*\.+\s*$",
    r"(?i)^\s*,+\s*$",
    r"(?i)^\s*\*+\s*$",
]
_HALLUCINATION_RES = [re.compile(p) for p in HALLUCINATION_PATTERNS]


def clean_transcript(text):
    text = text.strip()
    if not text:
        return ""
    for pattern in _HALLUCINATION_RES:
        text = pattern.sub("", text)
    text = re.sub(r'\b(\w+(?:\s+\w+){1,3})\s+(?:\1\s*){2,}', r'\1', text)
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text


# ---------------------------------------------------------------------------
# Speech Recognizer
# ---------------------------------------------------------------------------

class SpeechRecognizer:
    def __init__(
        self,
        transcriber,
        display,
        sample_rate=16000,
        buffer_size=8000,
        min_silence_duration=0.3,
        max_silence_duration=1.5,
        silence_grow_rate=0.1,
    ):
        self.device = get_device()
        self.model = load_silero_vad()
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        self.min_silence_duration = min_silence_duration
        self.max_silence_duration = max_silence_duration
        self.silence_grow_rate = silence_grow_rate
        self.current_silence_threshold = min_silence_duration

        self.max_speech_duration = 30.0

        self.audio_queue = queue.Queue()
        self.pending_buffer = []
        self.sentence_buffer = []
        self.recording_speech = False
        self.silence_start = None
        self.speech_start = None
        self.file_count = 1
        self.directory_name = "recordings"
        self.transcriber = transcriber
        self.display = display
        self.running = False

    def create_directories(self):
        for directory in ["recordings", "transcripts"]:
            os.makedirs(directory, exist_ok=True)

    def callback(self, indata, frames, time_info, status):
        if status:
            sys.stderr.write(f"Audio status: {status}\n")
        self.audio_queue.put(indata[:, 0].copy())

    def start_recording(self):
        """Starts the audio stream and begins recording."""
        self.create_directories()
        self.running = True

        mic_name = sd.query_devices(kind='input')['name']
        print(f"\n{C.BCYAN}Microphone:{C.RESET} {mic_name}")
        print(f"{C.BCYAN}Sample rate:{C.RESET} {self.sample_rate}Hz")
        print(f"{C.BCYAN}Silence threshold:{C.RESET} {self.min_silence_duration}-{self.max_silence_duration}s (dynamic)")
        print(f"\n{C.BGREEN}Recording... Speak in sentences. Press Ctrl+C to stop.{C.RESET}\n")

        self.display.set_state("IDLE")

        with sd.InputStream(
            callback=self.callback, channels=1, samplerate=self.sample_rate
        ):
            while self.running:
                self._drain_queue()
                time.sleep(0.01)

    def stop(self):
        self.running = False

    def _drain_queue(self):
        while True:
            try:
                chunk = self.audio_queue.get_nowait()
                self.pending_buffer.extend(chunk)
            except queue.Empty:
                break

        if len(self.pending_buffer) < self.buffer_size:
            return

        data = list(self.pending_buffer)
        self.pending_buffer.clear()
        self._process_audio(data)

    def _process_audio(self, audio_data_list):
        audio_tensor = torch.tensor(audio_data_list, dtype=torch.float32)

        speech_timestamps = get_speech_timestamps(
            audio_tensor, self.model,
            threshold=0.4,
            speech_pad_ms=150,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )

        if speech_timestamps:
            if not self.recording_speech:
                self.speech_start = time.time()
                self.display.set_state("RECORDING")
            self.recording_speech = True
            self.silence_start = None
            int_audio = (np.array(audio_data_list) * 32767).astype(np.int16)
            self.sentence_buffer.extend(int_audio)

            if self.speech_start and (time.time() - self.speech_start > self.max_speech_duration):
                self._save_audio()
                self._reset_buffers()
        else:
            if self.recording_speech:
                int_audio = (np.array(audio_data_list) * 32767).astype(np.int16)
                self.sentence_buffer.extend(int_audio)
                self._handle_silence()
            else:
                self.display.set_state("IDLE")

    def _compute_dynamic_threshold(self):
        if self.speech_start is None:
            return self.min_silence_duration
        speech_duration = time.time() - self.speech_start
        threshold = self.min_silence_duration + speech_duration * self.silence_grow_rate
        return min(threshold, self.max_silence_duration)

    def _handle_silence(self):
        if self.silence_start is None:
            self.silence_start = time.time()
            self.current_silence_threshold = self._compute_dynamic_threshold()
            self.display.set_state("LISTENING")
        elif time.time() - self.silence_start > self.current_silence_threshold:
            if self.sentence_buffer:
                self._save_audio()
            self._reset_buffers()

    def _save_audio(self):
        output_file = os.path.join(
            self.directory_name, f"output_sentence_audio_{self.file_count}.wav"
        )
        audio_array = np.array(self.sentence_buffer, dtype=np.int16)
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_array.tobytes())

        self.display.set_state("PROCESSING")
        self.transcriber.enqueue(output_file)
        self.file_count += 1

    def _reset_buffers(self):
        self.sentence_buffer.clear()
        self.recording_speech = False
        self.silence_start = None
        self.speech_start = None
        self.current_silence_threshold = self.min_silence_duration


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    MIN_SEGMENT_DURATION = 0.5
    SILENCE_PAD_SAMPLES = 4800
    OVERLAP_SAMPLES = 8000

    def __init__(
        self,
        display,
        model_name="large-v3",
        sample_rate=16000,
        transcript_directory="./transcripts",
        language="en",
    ):
        self.device = get_device()
        compute = "float16" if self.device == "cuda" else "int8"
        print(f"{C.CYAN}Loading Whisper {model_name} on {self.device} ({compute})...{C.RESET}")
        self.model = WhisperModel(model_name, device=self.device, compute_type=compute)
        self.sample_rate = sample_rate
        self.language = language
        self.chunk_duration = 30
        self.transcript_file = self._generate_transcript_file(transcript_directory)
        self.current_transcription = ""
        self._file_lock = threading.Lock()
        self.display = display

        self._last_context = ""
        self._last_audio = None

        self._work_queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._warm_up()

    def _warm_up(self):
        print(f"{C.CYAN}Warming up model...{C.RESET}")
        dummy = np.zeros(self.sample_rate, dtype=np.float32)
        segments, _ = self.model.transcribe(dummy, beam_size=1, language=self.language)
        for _ in segments:
            pass
        print(f"{C.GREEN}Warm-up complete.{C.RESET}")

    def _generate_transcript_file(self, transcript_directory):
        os.makedirs(transcript_directory, exist_ok=True)
        file_count = 1
        while os.path.exists(
            os.path.join(transcript_directory, f"transcript_{file_count}.txt")
        ):
            file_count += 1
        path = os.path.join(transcript_directory, f"transcript_{file_count}.txt")
        print(f"{C.CYAN}Transcript file:{C.RESET} {path}")
        return path

    def enqueue(self, audio_file):
        self._work_queue.put(audio_file)

    def _worker_loop(self):
        while True:
            try:
                audio_file = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if audio_file is None:
                break
            self._transcribe_audio_file(audio_file)
            self._work_queue.task_done()

    def _transcribe_audio_file(self, audio_file):
        audio_data, _ = self._load_audio(audio_file)
        duration = len(audio_data) / self.sample_rate

        if duration < self.MIN_SEGMENT_DURATION:
            return

        # Overlap from previous segment
        if self._last_audio is not None:
            audio_data = np.concatenate([self._last_audio, audio_data])
        self._last_audio = audio_data[-self.OVERLAP_SAMPLES:].copy()

        # Full preprocessing pipeline
        audio_data = preprocess_audio(audio_data, sr=self.sample_rate)

        # Peak normalization
        peak = np.max(np.abs(audio_data))
        if peak > 0.01:
            audio_data = audio_data / peak * 0.95

        # Silence padding
        pad = np.zeros(self.SILENCE_PAD_SAMPLES, dtype=np.float32)
        audio_data = np.concatenate([pad, audio_data, pad])

        self._process_audio_chunks(audio_data, duration)

    def _load_audio(self, audio_file):
        with wave.open(audio_file, "rb") as wf:
            frame_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            audio_array = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            )
        return audio_array, frame_rate

    def _process_audio_chunks(self, audio_array, original_duration):
        chunk_size = self.sample_rate * self.chunk_duration
        full_text = ""
        for i in range(0, len(audio_array), chunk_size):
            chunk = audio_array[i : i + chunk_size]
            if len(chunk) > 0:
                full_text += self._transcribe_chunk(chunk)

        full_text = clean_transcript(full_text)

        if full_text:
            self._last_context = full_text
            self.current_transcription += " " + full_text
            self._update_transcript_file(full_text)
            self.display.add_transcript(full_text, original_duration)

    def _transcribe_chunk(self, audio_chunk):
        prompt = self._last_context[-200:] if self._last_context else None

        segments, info = self.model.transcribe(
            audio_chunk,
            beam_size=5,
            task="transcribe",
            language=self.language,
            initial_prompt=prompt,
            word_timestamps=True,
            hallucination_silence_threshold=0.1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            condition_on_previous_text=True,
        )

        text = ""
        for segment in segments:
            text += segment.text
        return text

    def _update_transcript_file(self, transcript):
        max_length = 80
        with self._file_lock:
            lines = []
            current_line = ""
            try:
                with open(self.transcript_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        current_line = lines[-1].strip()
            except FileNotFoundError:
                pass

            current_line += " " + transcript if current_line else transcript

            lines_to_write = []
            while len(current_line) > max_length:
                lines_to_write.append(current_line[:max_length])
                current_line = current_line[max_length:]

            with open(self.transcript_file, "w") as f:
                if lines:
                    f.writelines(lines[:-1])
                for line in lines_to_write:
                    f.write(line + "\n")
                if current_line:
                    f.write(current_line)

    def combine_and_cleanup_recordings(
        self, recordings_directory, output_file="combined_recording.wav"
    ):
        audio_files = sorted(
            f for f in os.listdir(recordings_directory) if f.endswith((".wav", ".mp3"))
        )
        if not audio_files:
            print(f"{C.YELLOW}No recordings to combine.{C.RESET}")
            return

        combined_audio = AudioSegment.empty()
        for file_name in audio_files:
            file_path = os.path.join(recordings_directory, file_name)
            combined_audio += AudioSegment.from_file(file_path)

        combined_audio.export(output_file, format="wav")
        print(f"{C.GREEN}Combined audio saved as '{output_file}'{C.RESET}")

        for file_name in audio_files:
            os.remove(os.path.join(recordings_directory, file_name))

        print(f"{C.GREEN}All recordings combined and originals deleted.{C.RESET}")

    def shutdown(self):
        self._work_queue.join()
        self._work_queue.put(None)
        self._worker.join(timeout=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def handle_exit(sig, frame, transcriber, recognizer):
    print(f"\n{C.YELLOW}Shutting down...{C.RESET}")
    recognizer.stop()
    transcriber.shutdown()
    transcriber.combine_and_cleanup_recordings("recordings")

    # Print final transcript
    print(f"\n{C.BCYAN}{'=' * 70}{C.RESET}")
    print(f"{C.BOLD}FINAL TRANSCRIPT{C.RESET}")
    print(f"{C.BCYAN}{'=' * 70}{C.RESET}")
    if transcriber.current_transcription.strip():
        wrapped = textwrap.wrap(transcriber.current_transcription.strip(), width=70)
        for line in wrapped:
            print(f"  {C.BGREEN}{line}{C.RESET}")
    else:
        print(f"  {C.DIM}(no transcription){C.RESET}")
    print(f"{C.BCYAN}{'=' * 70}{C.RESET}")
    print(f"{C.CYAN}Transcript saved to: {transcriber.transcript_file}{C.RESET}\n")

    sys.exit(0)


if __name__ == "__main__":
    display = LiveTranscriptDisplay()
    transcriber = Transcriber(display, language="en")
    recognizer = SpeechRecognizer(transcriber, display)

    signal.signal(
        signal.SIGINT,
        lambda sig, frame: handle_exit(sig, frame, transcriber, recognizer),
    )

    recognizer.start_recording()
