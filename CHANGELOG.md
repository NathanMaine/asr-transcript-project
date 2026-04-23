# Changelog

Complete chronological history of the ASR Transcript Project.

## 2026-02-24 — Infrastructure: NFS Migration & Model Backups (Complete)

### NAS NFS Migration

- CIFS mounts from DGX Spark to NAS were unstable ("Host is down" errors)
- Configured NFS exports on NAS (`/etc/exports`) for 5 shares: ai-models, Projects, backups, media, docker
- Export options: `rw,sync,no_subtree_check,all_squash,anonuid=1000,anongid=10` (maps to the Spark user)
- Switched Spark from CIFS to NFS — updated `/etc/fstab` (old CIFS entries preserved as comments)
- All 5 NFS mounts verified working, significantly more stable than CIFS

### Model Backups to NAS

- Copied all 3 active models from Spark cache to NAS:
  - `parakeet-ctc-1.1b` (8.0 GB) → `${NAS_MOUNT}/projects/asr-transcript-project/models/`
  - `segmentation-3.0` (5.7 MB) → same directory
  - `wespeaker-voxceleb-resnet34-LM` (26 MB) → same directory
- Unused `faster-whisper-large-v3` (2.9 GB) identified for deletion from Spark cache

### Documentation & Code Sync

- Created 6 documentation files: CHANGELOG, TODO, DEVELOPMENT_NOTES, SETUP_GUIDE, MODEL_INVENTORY, updated README
- All project files (code + docs) synced to NAS at `${NAS_MOUNT}/projects/asr-transcript-project/`
- Local copy maintained at `${LOCAL_REPO}/`

---

## 2026-02-24 — Phase 2: Speaker Diarization (Complete)

### Diarization Implementation
- Added `DiarizationEngine` class to `transcribe.py` using pyannote.audio 3.1.1
- Lazy-loads the diarization pipeline on first use
- Produces speaker segments `(start_sec, end_sec, speaker_label)` per file
- `assign_speakers_to_words()` merges word timestamps with speaker segments using midpoint matching
- `format_diarized()` generates `[HH:MM:SS] Speaker N: text...` output format
- Speaker labels auto-mapped from raw `SPEAKER_00` to `Speaker 1`, `Speaker 2`, etc.

### pyannote.audio 3.1.1 Installation & Patching
- Installed pyannote-audio==3.1.1 with `--no-deps` to avoid version conflicts
- Installed dependencies: asteroid-filterbanks, einops, lightning, speechbrain, rich, semver, torch-audiomentations
- Created `patch_pyannote.py` — 9 compatibility fixes for torchaudio nightly + numpy 2.x + PyTorch 2.6+
- Created `patch_torchaudio_load.py` — replaces torchaudio.load/info with soundfile

### Issues Resolved (Phase 2)
- **Issue 11**: `use_auth_token` vs `token` conflict — only patch hf_hub_download calls, not pyannote internals
- **Issue 12**: HuggingFace gated model 403 — manually accepted license for pyannote/segmentation-3.0
- **Issue 13**: PyTorch 2.6+ `weights_only=True` default — patched lightning_fabric cloud_io.py + model.py pl_load
- **Issue 14**: torchaudio.load requires torchcodec/libnppicc.so.12 — replaced with soundfile via `_sf_load()`
- **Issue 15**: torchaudio.info removed in nightly — replaced with soundfile-based `get_torchaudio_info()`
- **Issue 16**: Blackwell FFT CUDA kernels missing (sm_121) — run diarization on CPU (~3.2x realtime)
- **Issue 17**: `np.NAN` (uppercase) removed in numpy 2.0 — fixed 8 occurrences across 4 files
- **Issue 18**: torchaudio.set/get/list_audio_backends removed — wrapped in try/except and getattr

### Test Results
- **5-file diarized test**: 5/5 succeeded, 8,444 words, 44.0 min audio, 13m 30s total
- **20-file diarized test**: 20/20 succeeded, 36,501 words, 206.2 min audio, 1h 3m 22s total
- Speed: 3.1-4.8x realtime (CPU-bound diarization)

### CLI Enhancements
- Added `--output-format {diarized,timestamps,plain}` (default: timestamps)
- Added `--no-diarize` shortcut for timestamps-only
- Added `--num-speakers N` to force speaker count
- Added `--hf-token TOKEN` for HuggingFace authentication

---

## 2026-02-24 — Phase 1: CTC Timestamp Extraction (Complete)

### Timestamp Implementation
- Added `extract_word_timestamps()` — extracts word-level timestamps from CTC logits via frame alignment
- Each CTC output frame = 0.08 seconds (hop_length=160 @ 16kHz, 8x subsampling)
- Progressively decodes tokens to detect word boundaries (spaces in decoded text)
- `format_timestamps_only()` inserts `[HH:MM:SS]` markers every N seconds (default: 30)
- Modified `transcribe_audio()` to return word list with timestamps instead of plain text

### Test Results
- **5-file timestamps test**: 5/5 succeeded, 8,444 words, 44.0 min audio, 24s total (~110x realtime)
- **20-file timestamps test (v1)**: 20/20 succeeded, 36,560 words, 206 min audio, 1m 53s total

---

## 2026-02-23 — Initial Setup & Plain Text Transcription (Complete)

### Environment Setup on DGX Spark
- Created Python venv at `${VENV}/` (local filesystem, not NAS)
- Installed PyTorch nightly `torch-2.12.0.dev20260223+cu128` for Blackwell sm_121 support
- Installed Transformers from source `transformers-5.3.0.dev0` for Parakeet model class
- Downloaded static ffmpeg 7.0.2 ARM64 binary to `~/.local/bin/ffmpeg`
- Transferred 1,013 MP3 files (9.5GB) from an external drive to NAS input directory

### Model Selection Journey
1. **faster-whisper** — abandoned (ctranslate2 no aarch64+CUDA wheels)
2. **openai-whisper large-v3** — abandoned (~1x realtime on Blackwell, too slow)
3. **NeMo toolkit** — abandoned (lhotse version conflict)
4. **NVIDIA Parakeet CTC 1.1B via HuggingFace** — success (~110x realtime)

### Issues Resolved (Setup)
- **Issue 1**: NAS venv symlink error — use local filesystem for venv
- **Issue 2**: CTranslate2 no CUDA on aarch64 — abandoned faster-whisper
- **Issue 3**: PyTorch sm_121 incompatibility — PyTorch nightly cu128
- **Issue 4**: ffmpeg not found — static ARM64 binary
- **Issue 5**: Whisper too slow on Blackwell — switched to Parakeet CTC
- **Issue 6**: NeMo lhotse version conflict — switched to HuggingFace Transformers
- **Issue 7**: Transformers didn't recognize parakeet_ctc — install from source
- **Issue 8**: max_position_embeddings exceeded — manual 25s audio chunking
- **Issue 9**: Feature extractor key mismatch — `input_features` not `input_values`
- **Issue 10**: Float32/Float16 type mismatch — cast to float16

### Original Plain Text Run
- 1,013 files transcribed as plain text (no timestamps, no speakers)
- ~110x realtime, total run ~1.4 hours
- Output: one `.txt` file per `.mp3` in `output/` directory
