# Development Notes

Technical deep-dives, architecture decisions, and lessons learned.

## Architecture Overview

```
MP3 file
  │
  ├── librosa.load(sr=16000, mono=True)
  │     └── Resamples to 16kHz mono float32
  │
  ├── Manual chunking (25s chunks)
  │     └── Keeps under Parakeet's 5000 max_position_embeddings
  │
  ├── AutoProcessor.feature_extractor → 80-dim log-mel spectrogram
  │     └── Cast to float16 for GPU model
  │
  ├── AutoModelForCTC.forward() → logits (1, num_frames, vocab_size)
  │     └── Running on CUDA (GPU)
  │
  ├── extract_word_timestamps() — CTC frame alignment
  │     ├── torch.argmax(logits) → predicted_ids
  │     ├── Collapse consecutive duplicates + blanks (CTC decoding)
  │     ├── Track frame index for each token
  │     ├── Progressive decoding to detect word boundaries (spaces)
  │     └── Returns [{word, start_sec, end_sec}, ...]
  │
  ├── [Optional] DiarizationEngine.diarize()
  │     ├── pyannote/speaker-diarization-3.1 pipeline
  │     ├── Runs on CPU (Blackwell FFT kernel limitation)
  │     ├── Returns [(start, end, speaker_label), ...]
  │     └── assign_speakers_to_words() — midpoint matching
  │
  └── Format output
        ├── format_diarized()  → [HH:MM:SS] Speaker N: text
        ├── format_timestamps_only() → [HH:MM:SS]\ntext
        └── format_plain() → raw text
```

## CTC Frame Alignment — How Timestamps Work

The Parakeet CTC model outputs logits of shape `(1, num_frames, 1025)` where:
- 1025 = 1024 character tokens + 1 blank token (ID 1024)
- Each frame represents 0.08 seconds of audio

CTC decoding process:
1. `torch.argmax(logits)` → sequence of token IDs per frame
2. Collapse consecutive duplicate tokens (CTC standard)
3. Remove blank tokens (ID 1024)
4. Track which frame index each surviving token came from
5. Progressively decode token sequences to detect word boundaries
6. When a space appears in decoded text → word boundary found
7. Word start_sec = first_frame_index * 0.08 + chunk_offset

This gives word-level timestamps for free — no extra model inference needed.

### Frame timing derivation
- Audio sample rate: 16,000 Hz
- Parakeet's hop_length: 160 samples (10ms per hop)
- 8x subsampling in FastConformer encoder
- Effective: 160 * 8 = 1,280 samples per CTC frame
- Time per frame: 1,280 / 16,000 = 0.08 seconds

### Verification
- 25 seconds of audio → 25 / 0.08 = 312.5 → observed 313 frames (confirmed)

## pyannote.audio 3.1.1 Compatibility Layer

### Why 3.1.1 and not 4.0+?
pyannote 4.0+ switched from torchaudio to torchcodec for audio I/O. torchcodec requires system CUDA NPP libraries (`libnppicc.so.12`) and only has x86_64 binaries. The DGX Spark is aarch64, so torchcodec is a dead end.

### The Patching Strategy (Critical Design Decision)

pyannote 3.1.1 uses `use_auth_token` throughout its codebase:
- In function signatures: `def from_pretrained(..., use_auth_token=...)`
- In class constructors: `SpeakerDiarization.__init__(**params)` where params contains `use_auth_token`
- In internal passing: `params.setdefault("use_auth_token", use_auth_token)`
- In HuggingFace API calls: `hf_hub_download(..., use_auth_token=use_auth_token)`

But newer `huggingface_hub` dropped `use_auth_token` in favor of `token`.

**The trap**: You CANNOT blindly rename all `use_auth_token` → `token`. Only `hf_hub_download()` call sites need `token=`. All pyannote internal signatures still expect `use_auth_token`.

**What we patch:**
- `pipeline.py` line with `hf_hub_download(... use_auth_token=...)` → `token=`
- `model.py` `hf_hub_download()` calls → `token=` (via regex, function context only)

**What we do NOT patch:**
- `params.setdefault("use_auth_token", ...)` — unpacked into class constructors
- `getter.py` — calls `Model.from_pretrained()` which accepts `use_auth_token` natively
- Any function signature containing `use_auth_token`

### weights_only Fix Explained

PyTorch 2.6+ changed `torch.load()` default from `weights_only=False` to `weights_only=True`. pyannote checkpoints contain `torch.torch_version.TorchVersion` objects which aren't in PyTorch's safe globals list.

The fix chain:
1. pyannote's `model.py` calls `pl_load()` (pytorch_lightning)
2. `pl_load()` calls `lightning_fabric.utilities.cloud_io._load()`
3. `_load()` receives `weights_only=None` from default parameter
4. PyTorch 2.6+ treats `None` as "use my default" which is `True`
5. Loading fails with `UnpicklingError`

**Fix**: Patch `cloud_io.py` to convert `None → False` inside the function body. Also patch `model.py` to pass `weights_only=False` explicitly to `pl_load`.

### torchaudio.load → soundfile Replacement

torchaudio nightly removed the old `load()` function, requiring torchcodec. We replace it with `_sf_load()`:

```python
def _sf_load(path, frame_offset=0, num_frames=-1):
    data, sample_rate = sf.read(path, dtype="float32")
    # soundfile: (samples, channels) → torch: (channels, samples)
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data[np.newaxis, :])
    return waveform, sample_rate
```

Also replaced `torchaudio.info()` → `sf.info()` + `AudioMetaData` stub dataclass.

## Blackwell (sm_121) FFT Kernel Issue

PyTorch nightly `torch-2.12.0.dev20260223+cu128` supports Blackwell for most operations but hasn't compiled `cufft` kernels for sm_121 yet. This causes:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

During `torch.fft.rfft()` which pyannote uses for fbank feature extraction in the speaker embedding pipeline.

**Workaround**: Run diarization entirely on CPU. The ASR model (Parakeet) uses convolutions, not FFT, so it runs fine on GPU.

**Impact**: Diarization speed drops from ~10x to ~3.2x realtime.

**Monitor**: Check if future PyTorch nightlies add FFT support:
```python
import torch
t = torch.randn(100, device='cuda')
torch.fft.rfft(t)  # Will raise RuntimeError if still broken
```

## numpy 2.0 Deprecation Cascade

numpy 2.0 removed multiple aliases:
- `np.NaN` (mixed case) — used in pyannote inference.py, segmentation tasks
- `np.NAN` (uppercase) — used in speaker_diarization.py, resegmentation.py, speaker_verification.py (5 occurrences!), inference.py

Both replaced with `np.nan` (lowercase, the only surviving form).

## Model Cache Locations on Spark

Models are cached in TWO separate locations:

### HuggingFace Hub cache (`~/.cache/huggingface/hub/`)
- `models--nvidia--parakeet-ctc-1.1b` — **8.0 GB** (ASR model weights + config)
- `models--pyannote--segmentation-3.0` — 28 KB (config only, weights elsewhere)
- `models--pyannote--wespeaker-voxceleb-resnet34-LM` — 28 KB (config only)
- `models--Systran--faster-whisper-large-v3` — 2.9 GB (unused, from earlier experiments)

### Torch/pyannote cache (`~/.cache/torch/pyannote/`)
- `models--pyannote--segmentation-3.0/.../pytorch_model.bin` — segmentation weights
- `models--pyannote--wespeaker-voxceleb-resnet34-LM/.../pytorch_model.bin` — speaker embedding weights
- Total: ~32 MB

### Note on speaker-diarization-3.1
The `pyannote/speaker-diarization-3.1` pipeline is a configuration file that references the segmentation and embedding models. It downloads the config from HuggingFace but the actual neural network weights come from the two models above.

## File Processing Pipeline

```
1. Scan input/ for *.mp3 files (sorted alphabetically)
2. Check output/ for existing *.txt files (resume support)
3. Load Parakeet CTC model to GPU (one-time, ~5s)
4. [If diarization] Load pyannote pipeline to CPU (one-time, ~2s)
5. For each MP3:
   a. librosa.load() → 16kHz mono float32 numpy array
   b. Split into 25-second chunks
   c. For each chunk:
      - Feature extraction → 80-dim log-mel spectrogram
      - Model inference → CTC logits
      - Frame alignment → word timestamps
   d. [If diarization] pyannote pipeline → speaker segments
   e. [If diarization] Merge words + speakers via midpoint matching
   f. Format output text
   g. Write .txt file
6. Write summary to log + JSON results file
```

## Performance Characteristics

### ASR (GPU)
- Model load: ~5s (one-time)
- Per-chunk inference: ~0.1s per 25s chunk
- Bottleneck: Audio loading from NAS (~1-3s per file depending on size)
- Overall: ~110x realtime

### Diarization (CPU)
- Pipeline load: ~2s (one-time, cached after first file)
- Per-file: roughly 1x audio duration (CPU-bound)
- Short files (<30s): overhead dominates, appears 4-5x realtime
- Long files (>10min): approaches 3.1x realtime
- Bottleneck: CPU FFT computation in speaker embedding extraction

### Memory Usage
- GPU: ~2.5 GB for Parakeet model (float16)
- CPU: ~1-2 GB for pyannote pipeline
- Audio buffer: negligible (~16 MB for 15 min file)
