# ASR Transcript Project

Batch transcription of 1,013 YouTube videos from an educational tech creator channel (MP3 format, 9.5GB total) using **NVIDIA Parakeet CTC 1.1B** on the **DGX Spark**, with optional **speaker diarization** via pyannote.audio 3.1.1.

## Output Formats

### Diarized (default)
```
[00:00:00] Speaker 1: apple dropped a bombshell on us today and didn't
even have a keynote to announce three new machines well sort of machines

[00:00:30] Speaker 1: the mac mini was updated to m two and m two pro
well they also kind of mentioned a new mac pro but...

[00:01:00] Speaker 2: so tell me about your setup what are you
running on the mac studio
```

### Timestamps only
```
[00:00:00]
apple dropped a bombshell on us today and didn't even have a keynote...

[00:00:30]
the mac mini was updated to m two and m two pro...
```

### Plain (legacy)
```
apple dropped a bombshell on us today and didn't even have a keynote...
```

## Documentation Index

| File | Description |
|------|-------------|
| [README.md](README.md) | Overview, usage, performance, issues log (this file) |
| [CHANGELOG.md](CHANGELOG.md) | Complete chronological project history |
| [TODO.md](TODO.md) | Roadmap, next steps, backlog |
| [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) | Architecture, technical deep-dives, lessons learned |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Full DGX Spark installation from scratch |
| [MODEL_INVENTORY.md](MODEL_INVENTORY.md) | All models: locations, sizes, backup status |

## Project Layout

```
asr-transcript-project/
  transcribe.py             # Main transcription script (Phase 1 + Phase 2)
  patch_pyannote.py         # Patches pyannote.audio 3.1.1 for nightly compat (9 fixes)
  patch_torchaudio_load.py  # Patches torchaudio.load/info -> soundfile
  requirements.txt          # Python dependencies
  .gitignore                # Excludes input/, output/, logs/, .venv, test_*.py
  README.md                 # Overview + issues log
  CHANGELOG.md              # Complete project history
  TODO.md                   # Roadmap and next steps
  DEVELOPMENT_NOTES.md      # Technical architecture notes
  SETUP_GUIDE.md            # Full installation guide
  MODEL_INVENTORY.md        # Model locations and backups
  input/                    # 1,013 MP3 files (on NAS, gitignored)
  output/                   # Plain text transcripts v1 (on NAS, gitignored)
  output_v2_test/           # Phase 2 test output (on NAS, gitignored)
  logs/                     # Run logs + JSON results (on NAS, gitignored)
  models/                   # Model backups (on NAS, gitignored)
```

## Hardware

| Device | Specs |
|--------|-------|
| **NVIDIA DGX Spark** | GB10 chip, aarch64, CUDA 13.0 (Blackwell sm_121), 128GB unified memory |
| **NAS** | CIFS mount over 10GbE, NFS exports configured for stability |

### NAS Mount Paths

| Machine | Path |
|---------|------|
| Mac | `${NAS_MOUNT}/asr-transcript-project/` |
| Spark | `${NAS_MOUNT}/projects/asr-transcript-project/` |

## Models

### ASR: NVIDIA Parakeet CTC 1.1B
- **Model ID:** `nvidia/parakeet-ctc-1.1b`
- FastConformer encoder + CTC decoder
- English-only, outputs lowercase text
- ~110x realtime on DGX Spark (GPU)
- Word-level timestamps via CTC frame alignment (0.08s per frame)

### Speaker Diarization: pyannote.audio 3.1.1
- **Pipeline:** `pyannote/speaker-diarization-3.1`
- **Segmentation:** `pyannote/segmentation-3.0`
- **Embeddings:** `pyannote/wespeaker-voxceleb-resnet34-LM`
- Runs on **CPU** (Blackwell FFT kernels not yet compiled in PyTorch nightly)
- ~3.2x realtime on DGX Spark (CPU)
- Auto-detects number of speakers (or can be forced with `--num-speakers`)

## Performance

### Timestamps Only (Phase 1)

| Metric | 5-file test | 20-file test (v1) |
|--------|-------------|-------------------|
| Files processed | 5/5 | 20/20 |
| Total audio | 44.0 min | 206 min |
| Total time | 24s | 1m 53s |
| Avg per file | 4.8s | 5.7s |
| Speed | ~110x realtime | ~110x realtime |
| Total words | 8,444 | 36,560 |

### Diarized (Phase 2)

| Metric | 5-file test | 20-file test |
|--------|-------------|--------------|
| Files processed | 5/5 | 20/20 |
| Total audio | 44.0 min | 206.2 min |
| Total time | 13m 30s | 1h 3m 22s |
| Avg per file | 162.2s | 190.1s |
| Speed | ~3.2x realtime | 3.1-4.8x realtime |
| Total words | 8,444 | 36,501 |
| Fastest | 3.6s | 3.5s |
| Slowest | 333.8s | 836.0s (43 min clip) |

### Estimated Full Run (1,013 files)

| Mode | Est. Time |
|------|-----------|
| Timestamps only | ~1.5 hours |
| With diarization | ~53 hours |

Diarization is CPU-bound due to Blackwell FFT kernel limitations. Longer clips take proportionally longer (~3.1x realtime for 43 min audio vs ~4.8x for 17s clips). Once PyTorch nightly adds sm_121 FFT support, GPU diarization should bring this down significantly.

## Usage

### Prerequisites

Python venv is on the Spark's local filesystem (NAS can't support symlinks for venvs):
```
${VENV}/
```

Static ffmpeg binary:
```
${HOME}/.local/bin/ffmpeg
```

HuggingFace token (for diarization only):
```bash
huggingface-cli login
# Accept terms at: https://huggingface.co/pyannote/segmentation-3.0
# Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
```

### Running

```bash
# SSH to Spark
ssh ${SPARK_USER}@${SPARK_HOST}

# Full diarized run (all files, resumes from where it left off)
PATH=${HOME}/.local/bin:$PATH ${VENV}/bin/python \
  ${NAS_MOUNT}/projects/asr-transcript-project/transcribe.py \
  --output-format diarized --hf-token true

# Timestamps only (no diarization, ~40x faster)
PATH=${HOME}/.local/bin:$PATH ${VENV}/bin/python \
  ${NAS_MOUNT}/projects/asr-transcript-project/transcribe.py \
  --output-format timestamps

# Test run (20 files with diarization)
PATH=${HOME}/.local/bin:$PATH ${VENV}/bin/python \
  ${NAS_MOUNT}/projects/asr-transcript-project/transcribe.py \
  --output-format diarized --hf-token true --limit 20

# Plain text (legacy, no timestamps)
PATH=${HOME}/.local/bin:$PATH ${VENV}/bin/python \
  ${NAS_MOUNT}/projects/asr-transcript-project/transcribe.py \
  --output-format plain
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | `./input` | Directory containing MP3 files |
| `--output-dir` | `./output` | Directory for transcript output |
| `--limit N` | 0 (all) | Process only N files |
| `--output-format` | `timestamps` | `diarized`, `timestamps`, or `plain` |
| `--no-diarize` | false | Skip diarization (even if format is diarized) |
| `--timestamp-interval` | 30 | Insert timestamp markers every N seconds |
| `--num-speakers N` | auto | Force speaker count for diarization |
| `--hf-token TOKEN` | `$HF_TOKEN` | HuggingFace token (or `true` to use cached login) |

### Resume Support

The script automatically skips files that already have a `.txt` file in the output directory. Safe to stop and restart at any time.

### Using tmux for Long Runs

```bash
tmux new -s transcribe
# ... run the command ...
# Ctrl+B then D to detach
# tmux attach -t transcribe to reconnect
```

## Installation on DGX Spark

### Phase 1 (ASR + timestamps)

```bash
# Create venv on local filesystem (not NAS)
python3 -m venv ${PROJECT_DIR}/.venv
source ${VENV}/bin/activate

# PyTorch nightly for Blackwell
pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu128

# Transformers from source for Parakeet model
pip install git+https://github.com/huggingface/transformers.git

# Core deps
pip install librosa numpy soundfile
```

### Phase 2 (Speaker Diarization)

```bash
source ${VENV}/bin/activate

# torchaudio nightly (must match torch version)
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# pyannote.audio (no-deps to avoid version conflicts)
pip install pyannote-audio==3.1.1 --no-deps

# pyannote dependencies
pip install asteroid-filterbanks einops "lightning>=2.0.1" speechbrain rich semver torch-audiomentations

# Apply compatibility patches (MUST use venv Python)
cd ${NAS_MOUNT}/projects/asr-transcript-project
python patch_pyannote.py          # 9+ patches for nightly compat
python patch_torchaudio_load.py   # Replace torchaudio.load/info with soundfile

# HuggingFace login
huggingface-cli login
```

### Why pyannote 3.1.1 (not 4.0+)?

Version 4.0+ requires `torchcodec` which has no ARM64+CUDA binaries for the Spark. Version 3.1.1 uses torchaudio/soundfile instead, but needs patches for torchaudio nightly compatibility.

## Issues & Solutions Log

### 1. NAS Venv Symlink Error

**Error:** `[Errno 95] Operation not supported: 'lib' -> '.venv/lib64'`

**Cause:** CIFS (NAS filesystem) doesn't support symlinks. Python's `venv` tries to create a `lib64 -> lib` symlink on Linux.

**Solution:** Create the venv on Spark's local filesystem (`${VENV}/`) instead of on the NAS. The script and data stay on NAS; only the venv is local.

---

### 2. CTranslate2 No CUDA Support (faster-whisper)

**Error:** `ValueError: This CTranslate2 package was not compiled with CUDA support`

**Cause:** Pre-built ctranslate2 wheels don't support aarch64 + CUDA. The DGX Spark is ARM64, and faster-whisper depends on ctranslate2.

**Solution:** Abandoned faster-whisper entirely. Switched to openai-whisper (which uses PyTorch directly), then later to Parakeet CTC 1.1B.

---

### 3. PyTorch sm_121 Incompatibility

**Error:** `NVIDIA GB10 with CUDA capability sm_121 is not compatible with the current PyTorch installation`

**Cause:** PyTorch 2.5.1 (cu124) only supports up to sm_90a (Hopper). The DGX Spark's GB10 is Blackwell architecture (sm_121), which requires CUDA 12.8+ and PyTorch nightly.

**Solution:**
```bash
pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu128
```
This installed `torch-2.12.0.dev20260223+cu128` which recognizes sm_121.

---

### 4. ffmpeg Not Found

**Error:** `[Errno 2] No such file or directory: 'ffmpeg'`

**Cause:** ffmpeg wasn't installed on the Spark, and sudo access wasn't available for apt-get.

**Solution:** Downloaded a static ARM64 ffmpeg binary:
```bash
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-7.0.2-arm64-static.tar.xz
tar xf ffmpeg-7.0.2-arm64-static.tar.xz
cp ffmpeg-7.0.2-arm64-static/ffmpeg ~/.local/bin/
```
Then added `~/.local/bin` to PATH when running the script.

---

### 5. Whisper large-v3 Too Slow on Blackwell

**Symptom:** First file (16MB, ~15 min) took 678 seconds (~real-time speed, not the expected 10-30x speedup).

**Cause:** Blackwell GPU support in PyTorch nightly is experimental. The CUDA kernels aren't fully optimized for sm_121 yet, so Whisper's attention-heavy architecture runs near real-time.

**Solution:** Switched to NVIDIA Parakeet CTC 1.1B which is a much lighter model (1.1B params, CTC decoder instead of attention decoder). Achieved ~110x realtime.

---

### 6. NeMo lhotse Version Conflict

**Error:** `TypeError: object.__init__() takes exactly one argument` in `lhotse.dataset.sampling.DynamicCutSampler`

**Cause:** Version conflict between nemo_toolkit 2.6.2 and lhotse. NeMo installed a lhotse version with a breaking API change.

**Solution:** Abandoned the NeMo approach. Loaded the model through HuggingFace Transformers instead (`AutoModelForCTC` + `AutoProcessor`).

---

### 7. Transformers Didn't Recognize parakeet_ctc

**Error:** `ValueError: The checkpoint you are trying to load has model type 'parakeet_ctc' but Transformers does not recognize this architecture`

**Cause:** Transformers 4.53.3 (PyPI release) is too old — `parakeet_ctc` support was added after the latest stable release.

**Solution:** Install transformers from source:
```bash
pip install git+https://github.com/huggingface/transformers.git
```
This installed `transformers-5.3.0.dev0` which includes the Parakeet model class.

---

### 8. max_position_embeddings Exceeded

**Error:** `Sequence Length: 12500 has to be less or equal than config.max_position_embeddings 5000`

**Cause:** Parakeet CTC has a fixed positional encoding limit of 5000 frames (~30 seconds of audio at 10ms per frame). Long audio files exceed this.

**Solution:** Implemented manual audio chunking — split into 25-second chunks, process each through the model independently, concatenate results.

---

### 9. Feature Extractor Key Mismatch

**Error:** `AttributeError` on `inputs.input_values`

**Cause:** Parakeet CTC's feature extractor returns `input_features` (mel spectrogram), not `input_values` (raw waveform).

**Solution:** Changed `inputs.input_values` to `inputs["input_features"]`.

---

### 10. Float32/Float16 Type Mismatch

**Error:** `RuntimeError: Input type (float) and bias type (c10::Half) should be the same`

**Cause:** Model weights loaded in float16, but feature extractor outputs float32 tensors.

**Solution:** Cast input features to float16: `inputs["input_features"].to(device=device, dtype=torch.float16)`

---

### 11. pyannote `use_auth_token` vs `token` Conflict

**Error:** `hf_hub_download() got unexpected keyword argument 'use_auth_token'` / `SpeakerDiarization.__init__() got unexpected keyword argument 'token'`

**Cause:** Newer huggingface_hub dropped `use_auth_token` in favor of `token`. But pyannote 3.1.1's own function signatures (class constructors, `params.setdefault()`) still use `use_auth_token` internally. You cannot blindly rename all occurrences.

**Solution:** Only patch `hf_hub_download()` call sites to use `token=`, leave all pyannote internal `use_auth_token` signatures untouched. The user-facing API call uses `use_auth_token=` which pyannote then passes through correctly. See `patch_pyannote.py` Fix 5.

---

### 12. HuggingFace Gated Model Access (403)

**Error:** `403 Client Error: Cannot access gated repo for url...pyannote/segmentation-3.0`

**Cause:** pyannote models on HuggingFace require accepting license terms before downloading.

**Solution:** User must manually visit and accept terms at:
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1

---

### 13. PyTorch 2.6+ `weights_only=True` Default

**Error:** `_pickle.UnpicklingError: Weights only load failed... torch.torch_version.TorchVersion is not allowed`

**Cause:** PyTorch 2.6+ changed `torch.load()` to default `weights_only=True`, but pyannote checkpoints contain `TorchVersion` objects not in the safe globals list. The `lightning_fabric` loader receives `weights_only=None` which the function body treats as default (True in PyTorch 2.6+).

**Solution:** Patch `lightning_fabric/utilities/cloud_io.py` to convert `weights_only=None` to `False` inside the function body. Also patch `model.py` to pass `weights_only=False` to `pl_load`. See `patch_pyannote.py` Fixes 7-8.

---

### 14. torchaudio.load Requires torchcodec

**Error:** `ImportError: torchcodec is required...` / `libnppicc.so.12: cannot open shared object file`

**Cause:** torchaudio nightly requires torchcodec for `torchaudio.load()`. torchcodec needs system CUDA NPP libraries (`libnppicc.so.12`) which aren't available on the Spark.

**Solution:** Patched pyannote's `io.py` to use soundfile directly instead of torchaudio.load. The `_sf_load()` function is a drop-in replacement. See `patch_torchaudio_load.py`.

---

### 15. torchaudio.info Removed in Nightly

**Error:** `AttributeError: module 'torchaudio' has no attribute 'info'`

**Cause:** `torchaudio.info()` was also removed in nightly, along with `torchaudio.backend.common.AudioMetaData`.

**Solution:** Patched `get_torchaudio_info()` to use soundfile + a stub `AudioMetaData` dataclass. See `patch_torchaudio_load.py` and `patch_pyannote.py` Fix 3.

---

### 16. Blackwell FFT CUDA Kernel Missing

**Error:** `RuntimeError: CUDA error: no kernel image is available for execution on the device` (during `torch.fft.rfft`)

**Cause:** PyTorch nightly hasn't compiled FFT CUDA kernels for Blackwell sm_121 yet. The pyannote speaker embedding pipeline uses fbank features which require FFT.

**Solution:** Run diarization on CPU instead of GPU. Performance drops from ~10x to ~3.2x realtime, but is fully functional. The ASR model (Parakeet) still runs on GPU.

---

### 17. np.NAN (Uppercase) Removed in numpy 2.0

**Error:** `AttributeError: module 'numpy' has no attribute 'NAN'`

**Cause:** numpy 2.0 removed both `np.NaN` and `np.NAN`. The original patch script only handled `np.NaN` (mixed case), but several pyannote files use `np.NAN` (fully uppercase): speaker_diarization.py, resegmentation.py, speaker_verification.py (5 occurrences), inference.py.

**Solution:** Added Fix 9 to `patch_pyannote.py` to replace all `np.NAN` occurrences with `np.nan`.

---

### 18. torchaudio.set_audio_backend / get_audio_backend / list_audio_backends Removed

**Error:** `AttributeError: module 'torchaudio' has no attribute 'set_audio_backend'`

**Cause:** These functions were removed in torchaudio nightly. pyannote's `io.py` and `speaker_verification.py` call `set_audio_backend`/`get_audio_backend`, and speechbrain calls `list_audio_backends`.

**Solution:** Wrapped all calls in try/except or getattr patterns. See `patch_pyannote.py` Fixes 1, 2, 6.

## Software Versions (Working Configuration)

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.12 | System Python on Spark |
| PyTorch | 2.12.0.dev20260223+cu128 | Nightly, required for Blackwell |
| torchaudio | 2.11.0.dev20260224+cu128 | Nightly, patched for soundfile |
| Transformers | 5.3.0.dev0 | From source, required for Parakeet |
| pyannote-audio | 3.1.1 | Patched for nightly compat |
| speechbrain | latest | Patched for torchaudio nightly |
| lightning_fabric | latest | Patched for weights_only |
| librosa | 0.11.0 | Audio loading + resampling |
| numpy | 2.4.2 | |
| soundfile | 0.13.1 | Audio I/O backend (replaces torchaudio.load) |
| ffmpeg | 7.0.2 | Static ARM64 binary |

## Audio Format Notes

For best ASR results:
- **Ideal:** 16kHz mono WAV (what the model expects internally)
- **MP3 works fine** — librosa decodes and resamples automatically
- Bitrate doesn't matter for speech (128kbps is plenty)
- The model processes 80-dimensional log-mel spectrograms internally
