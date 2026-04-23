#!/usr/bin/env python3
"""Batch MP3 transcription with timestamps and optional speaker diarization.

Uses NVIDIA Parakeet CTC 1.1B for ASR with word-level timestamps extracted
from CTC frame alignments, and pyannote.audio 3.1 for speaker diarization.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

DEFAULT_INPUT_DIR = Path(__file__).parent / "input"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"
LOG_DIR = Path(__file__).parent / "logs"
MODEL_NAME = "nvidia/parakeet-ctc-1.1b"
SAMPLE_RATE = 16000
# Parakeet CTC has max_position_embeddings=5000 at ~10ms per frame
# 25 seconds of audio keeps us safely under the 5000 limit
CHUNK_SECONDS = 25
# CTC blank token ID (also pad_token_id for Parakeet)
BLANK_TOKEN_ID = 1024
# Each CTC output frame = 0.08 seconds (hop_length=160 @ 16kHz, 8x subsampling)
SEC_PER_FRAME = 0.08


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"transcribe_{timestamp}.log"

    logger = logging.getLogger("transcriber")
    logger.setLevel(logging.DEBUG)

    # File handler — DEBUG level (everything)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Console handler — INFO level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


def log_system_info(logger: logging.Logger):
    """Log system and environment details."""
    logger.debug("=" * 60)
    logger.debug("SYSTEM INFO")
    logger.debug("=" * 60)
    logger.debug(f"Python: {sys.version}")
    logger.debug(f"PyTorch: {torch.__version__}")
    logger.debug(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.debug(f"CUDA device: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.debug(f"GPU memory: {mem:.1f} GB")
    logger.debug(f"Model: {MODEL_NAME}")
    logger.debug(f"Chunk size: {CHUNK_SECONDS}s")
    logger.debug(f"Working dir: {os.getcwd()}")
    logger.debug(f"ffmpeg: {os.popen('ffmpeg -version 2>/dev/null | head -1').read().strip()}")
    logger.debug("=" * 60)


def get_file_info(mp3_path: Path) -> dict:
    """Get file metadata for logging."""
    stat = mp3_path.stat()
    return {
        "name": mp3_path.name,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "path": str(mp3_path),
    }


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Phase 1: CTC timestamp extraction
# ---------------------------------------------------------------------------

def extract_word_timestamps(logits, processor, chunk_start_sec):
    """Extract word-level timestamps from CTC logits using frame alignment.

    Args:
        logits: Tensor (1, num_frames, vocab_size) — raw CTC output
        processor: ParakeetProcessor with tokenizer
        chunk_start_sec: absolute start time of this chunk

    Returns:
        list of {"word": str, "start_sec": float, "end_sec": float}
    """
    predicted_ids = torch.argmax(logits, dim=-1)[0]  # (num_frames,)

    # Collapse CTC: remove consecutive duplicates, track frame positions
    collapsed = []
    prev = None
    for frame_idx, tid in enumerate(predicted_ids.tolist()):
        if tid != prev:
            if tid != BLANK_TOKEN_ID:
                collapsed.append({"token_id": tid, "frame": frame_idx})
            prev = tid

    if not collapsed:
        return []

    # Progressively decode tokens to detect word boundaries (spaces)
    words = []
    current_tokens = []
    current_start_frame = None

    for t in collapsed:
        if current_start_frame is None:
            current_start_frame = t["frame"]
        current_tokens.append(t["token_id"])

        partial = processor.tokenizer.decode(current_tokens)

        # Word boundary: space appears in decoded text
        if " " in partial and len(partial.strip()) > 0:
            parts = partial.split()
            if len(parts) > 1 or partial.endswith(" "):
                word = parts[0]
                if word:
                    words.append({
                        "word": word,
                        "start_sec": chunk_start_sec + current_start_frame * SEC_PER_FRAME,
                        "end_sec": chunk_start_sec + t["frame"] * SEC_PER_FRAME,
                    })
                # If there's a remainder after the space, start next word from this frame
                remainder_text = " ".join(parts[1:]) if len(parts) > 1 else ""
                if remainder_text:
                    current_tokens = list(processor.tokenizer.encode(remainder_text))
                    current_start_frame = t["frame"]
                else:
                    current_tokens = []
                    current_start_frame = None

    # Last word
    if current_tokens:
        word = processor.tokenizer.decode(current_tokens).strip()
        if word:
            words.append({
                "word": word,
                "start_sec": chunk_start_sec + current_start_frame * SEC_PER_FRAME,
                "end_sec": chunk_start_sec + collapsed[-1]["frame"] * SEC_PER_FRAME,
            })

    return words


def transcribe_audio(audio, processor, model, device, logger):
    """Transcribe audio with word-level timestamps via manual chunking.

    Returns:
        list of {"word": str, "start_sec": float, "end_sec": float}
    """
    total_samples = len(audio)
    chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
    num_chunks = int(np.ceil(total_samples / chunk_samples))
    duration = total_samples / SAMPLE_RATE

    logger.debug(f"  Audio duration: {duration:.1f}s, chunks: {num_chunks}, samples: {total_samples}")

    all_words = []

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_samples
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        chunk_start_sec = start / SAMPLE_RATE

        if len(chunk) < SAMPLE_RATE * 0.5:
            logger.debug(f"  Chunk {chunk_idx+1}/{num_chunks}: skipped (too short)")
            continue

        inputs = processor.feature_extractor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
        )
        input_features = inputs["input_features"].to(device=device, dtype=torch.float16)

        with torch.no_grad():
            logits = model(input_features).logits

        chunk_words = extract_word_timestamps(logits, processor, chunk_start_sec)
        word_count = len(chunk_words)
        all_words.extend(chunk_words)

        if word_count > 0:
            logger.debug(f"  Chunk {chunk_idx+1}/{num_chunks}: {word_count} words")
        else:
            logger.debug(f"  Chunk {chunk_idx+1}/{num_chunks}: (empty/silence)")

    return all_words


# ---------------------------------------------------------------------------
# Phase 2: Speaker diarization (requires pyannote.audio)
# ---------------------------------------------------------------------------

class DiarizationEngine:
    """Lazy-loaded speaker diarization via pyannote.audio 3.1."""

    def __init__(self, hf_token, device, logger):
        self._pipeline = None
        self._hf_token = hf_token
        self._device = device
        self._logger = logger

    def _load(self):
        if self._pipeline is not None:
            return
        self._logger.info("Loading diarization model (pyannote/speaker-diarization-3.1)...")
        load_start = time.time()
        try:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._hf_token,
            )
            # Run on CPU — Blackwell (sm_121) FFT kernels not yet compiled in
            # PyTorch nightly, causing RuntimeError in speaker embedding fbank.
            # CPU diarization is slower (~2x realtime) but fully functional.
            self._logger.info("Diarization running on CPU (Blackwell FFT kernels not yet available)")
            self._logger.info(f"Diarization model loaded in {time.time() - load_start:.1f}s")
        except ImportError:
            raise RuntimeError(
                "pyannote.audio not installed. Install with:\n"
                "  pip install pyannote-audio==3.1.1 --no-deps\n"
                "  pip install asteroid-filterbanks einops lightning speechbrain rich semver torch-audiomentations\n"
                "Or use --no-diarize to skip speaker diarization."
            )

    def diarize(self, audio_path, num_speakers=None):
        """Run diarization on an audio file.

        Returns:
            list of (start_sec, end_sec, speaker_label) sorted by start time
        """
        self._load()
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        diarization = self._pipeline(str(audio_path), **kwargs)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        return sorted(segments, key=lambda x: x[0])


def assign_speakers_to_words(words, speaker_segments):
    """Assign a speaker label to each word based on diarization segments.

    Uses the midpoint of each word to find the active speaker.
    """
    if not speaker_segments:
        for w in words:
            w["speaker"] = "SPEAKER_00"
        return words

    for w in words:
        midpoint = (w["start_sec"] + w["end_sec"]) / 2.0
        w["speaker"] = _find_speaker_at_time(midpoint, speaker_segments)

    return words


def _find_speaker_at_time(time_sec, segments):
    """Find which speaker is active at a given time."""
    for start, end, speaker in segments:
        if start <= time_sec <= end:
            return speaker
    return "SPEAKER_00"


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_plain(words):
    """Plain text output (legacy format, no timestamps)."""
    return " ".join(w["word"] for w in words)


def format_timestamps_only(words, interval=30):
    """Timestamps every N seconds, no speaker labels."""
    if not words:
        return ""

    lines = []
    next_boundary = 0.0

    for w in words:
        if w["start_sec"] >= next_boundary:
            if lines and not lines[-1] == "":
                lines.append("")
            lines.append(f"[{format_time(w['start_sec'])}]")
            next_boundary = (int(w["start_sec"] / interval) + 1) * interval

        # Append word to the last line or start a new content line
        if lines and not lines[-1].startswith("["):
            lines[-1] += " " + w["word"]
        else:
            lines.append(w["word"])

    return "\n".join(lines)


def format_diarized(words, interval=30):
    """Full diarized output: [HH:MM:SS] Speaker N: text..."""
    if not words:
        return ""

    # Map raw speaker IDs to "Speaker 1", "Speaker 2", etc.
    speaker_map = {}
    speaker_counter = 0

    lines = []
    current_speaker = None
    current_words = []
    next_boundary = 0.0

    for w in words:
        speaker_raw = w.get("speaker", "SPEAKER_00")
        if speaker_raw not in speaker_map:
            speaker_counter += 1
            speaker_map[speaker_raw] = f"Speaker {speaker_counter}"
        speaker_label = speaker_map[speaker_raw]

        need_break = (
            speaker_label != current_speaker
            or w["start_sec"] >= next_boundary
        )

        if need_break and current_words:
            ts = format_time(current_words[0]["start_sec"])
            text = " ".join(cw["word"] for cw in current_words)
            lines.append(f"[{ts}] {current_speaker}: {text}")
            lines.append("")
            current_words = []

        if need_break or current_speaker is None:
            current_speaker = speaker_label
            if w["start_sec"] >= next_boundary:
                next_boundary = (int(w["start_sec"] / interval) + 1) * interval

        current_words.append(w)

    # Final segment
    if current_words:
        ts = format_time(current_words[0]["start_sec"])
        text = " ".join(cw["word"] for cw in current_words)
        lines.append(f"[{ts}] {current_speaker}: {text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe MP3 files with Parakeet CTC 1.1B (timestamps + optional diarization)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Directory containing MP3 files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for transcript output",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only N files (0 = all)",
    )
    parser.add_argument(
        "--output-format", choices=["diarized", "timestamps", "plain"],
        default="timestamps",
        help="Output format (default: timestamps)",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="Skip speaker diarization (timestamps only)",
    )
    parser.add_argument(
        "--timestamp-interval", type=int, default=30,
        help="Insert timestamp markers every N seconds (default: 30)",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="Force number of speakers for diarization (auto-detect if omitted)",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace token for pyannote models (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    # Resolve diarization flags
    use_diarization = (args.output_format == "diarized" and not args.no_diarize)

    logger = setup_logging(LOG_DIR)
    log_system_info(logger)

    logger.info(f"Output format: {args.output_format}")
    logger.info(f"Diarization: {'enabled' if use_diarization else 'disabled'}")
    logger.info(f"Timestamp interval: {args.timestamp_interval}s")
    logger.debug(f"Input dir: {args.input_dir}")
    logger.debug(f"Output dir: {args.output_dir}")
    logger.debug(f"Limit: {args.limit}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(args.input_dir.glob("*.mp3"))
    if not mp3_files:
        logger.error(f"No MP3 files found in {args.input_dir}")
        return

    logger.debug(f"Total MP3 files found: {len(mp3_files)}")

    # Filter out already-transcribed files
    pending = []
    skipped = 0
    for mp3 in mp3_files:
        txt_path = args.output_dir / f"{mp3.stem}.txt"
        if txt_path.exists():
            skipped += 1
            logger.debug(f"SKIP (exists): {mp3.name}")
        else:
            pending.append(mp3)

    if args.limit > 0:
        pending = pending[: args.limit]

    total = len(pending)
    logger.info(f"Found {len(mp3_files)} MP3 files, {skipped} already transcribed, {total} to process")

    if total == 0:
        logger.info("Nothing to do.")
        return

    # Load ASR model
    logger.info(f"Loading model '{MODEL_NAME}' on CUDA...")
    model_start = time.time()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForCTC.from_pretrained(MODEL_NAME, dtype=torch.float16)
        device = torch.device("cuda")
        model = model.to(device)
        model.eval()

        model_elapsed = time.time() - model_start
        logger.info(f"Model loaded in {model_elapsed:.1f}s")
        logger.debug(f"Model type: {type(model).__name__}")
        logger.debug(f"Device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.debug(traceback.format_exc())
        return

    # Log GPU memory after model load
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        logger.debug(f"GPU memory after model load — allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")

    # Initialize diarization engine if needed
    diarizer = None
    if use_diarization:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.error(
                "HuggingFace token required for diarization. "
                "Set HF_TOKEN env var or use --hf-token. "
                "Or use --no-diarize / --output-format timestamps."
            )
            return
        diarizer = DiarizationEngine(hf_token, device, logger)

    failed = []
    results_log = []
    overall_start = time.time()

    for i, mp3_path in enumerate(pending, 1):
        txt_path = args.output_dir / f"{mp3_path.stem}.txt"
        file_info = get_file_info(mp3_path)
        logger.info(f"[{i}/{total}] {mp3_path.name} ({file_info['size_mb']} MB) ... ")
        logger.debug(f"  Input path: {mp3_path}")
        logger.debug(f"  Output path: {txt_path}")

        file_start = time.time()
        try:
            # Load audio
            logger.debug(f"  Loading audio...")
            audio, sr = librosa.load(str(mp3_path), sr=SAMPLE_RATE, mono=True)
            load_elapsed = time.time() - file_start
            audio_duration = len(audio) / SAMPLE_RATE
            logger.debug(f"  Audio loaded in {load_elapsed:.1f}s ({audio_duration:.1f}s duration)")

            # Transcribe with timestamps
            logger.debug(f"  Starting transcription...")
            words = transcribe_audio(audio, processor, model, device, logger)
            transcribe_elapsed = time.time() - file_start
            logger.debug(f"  Transcription: {len(words)} words in {transcribe_elapsed:.1f}s")

            # Speaker diarization (if enabled)
            if diarizer:
                logger.debug(f"  Running diarization...")
                diar_start = time.time()
                speaker_segments = diarizer.diarize(mp3_path, num_speakers=args.num_speakers)
                diar_elapsed = time.time() - diar_start
                num_speakers = len(set(s[2] for s in speaker_segments))
                logger.debug(f"  Diarization: {len(speaker_segments)} segments, {num_speakers} speakers in {diar_elapsed:.1f}s")
                words = assign_speakers_to_words(words, speaker_segments)
            else:
                # Default all words to Speaker 1 for non-diarized output
                for w in words:
                    w["speaker"] = "SPEAKER_00"

            # Format output
            if args.output_format == "plain":
                text = format_plain(words)
            elif args.output_format == "timestamps":
                text = format_timestamps_only(words, interval=args.timestamp_interval)
            else:
                text = format_diarized(words, interval=args.timestamp_interval)

            word_count = len(words)
            char_count = len(text)

            # Write output
            txt_path.write_text(text, encoding="utf-8")
            output_size = txt_path.stat().st_size

            elapsed = time.time() - file_start
            speed_ratio = audio_duration / elapsed if elapsed > 0 else 0
            logger.info(f"  DONE in {elapsed:.1f}s — {word_count} words, {char_count} chars ({speed_ratio:.1f}x realtime)")
            logger.debug(f"  First 200 chars: {text[:200]}")

            results_log.append({
                "file": mp3_path.name,
                "status": "success",
                "size_mb": file_info["size_mb"],
                "audio_duration_s": round(audio_duration, 1),
                "elapsed_s": round(elapsed, 1),
                "speed_ratio": round(speed_ratio, 1),
                "words": word_count,
                "chars": char_count,
            })

        except Exception as e:
            elapsed = time.time() - file_start
            logger.error(f"  FAILED in {elapsed:.1f}s: {e}")
            logger.debug(f"  Full traceback:\n{traceback.format_exc()}")
            failed.append((mp3_path.name, str(e)))

            results_log.append({
                "file": mp3_path.name,
                "status": "failed",
                "size_mb": file_info["size_mb"],
                "elapsed_s": round(elapsed, 1),
                "error": str(e),
            })

        # Log GPU state periodically
        if torch.cuda.is_available() and i % 5 == 0:
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            logger.debug(f"  GPU memory checkpoint [{i}/{total}]: {allocated:.2f} GB allocated")

    # Summary
    total_time = time.time() - overall_start
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    succeeded = total - len(failed)
    avg_time = total_time / total if total > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("RUN SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Format:     {args.output_format}")
    logger.info(f"Diarize:    {'yes' if use_diarization else 'no'}")
    logger.info(f"Processed:  {succeeded}/{total} succeeded, {len(failed)} failed, {skipped} skipped")
    logger.info(f"Total time: {hours}h {minutes}m {seconds}s")
    logger.info(f"Avg time:   {avg_time:.1f}s per file")

    if succeeded > 0:
        success_results = [r for r in results_log if r["status"] == "success"]
        total_words = sum(r["words"] for r in success_results)
        total_audio = sum(r.get("audio_duration_s", 0) for r in success_results)
        logger.info(f"Total words: {total_words:,}")
        logger.info(f"Total audio: {total_audio/60:.1f} minutes")
        logger.info(f"Fastest:    {min(r['elapsed_s'] for r in success_results):.1f}s")
        logger.info(f"Slowest:    {max(r['elapsed_s'] for r in success_results):.1f}s")

    if failed:
        logger.info("")
        logger.info("FAILED FILES:")
        for name, err in failed:
            logger.info(f"  - {name}: {err}")

    logger.info("=" * 60)

    # Write JSON results log
    results_file = LOG_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "run_time": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "output_format": args.output_format,
            "diarization": use_diarization,
            "timestamp_interval": args.timestamp_interval,
            "chunk_seconds": CHUNK_SECONDS,
            "total_files": len(mp3_files),
            "processed": succeeded,
            "failed": len(failed),
            "skipped": skipped,
            "total_seconds": round(total_time, 1),
            "avg_seconds_per_file": round(avg_time, 1),
            "results": results_log,
        }, f, indent=2)
    logger.info(f"Results JSON: {results_file}")


if __name__ == "__main__":
    main()
