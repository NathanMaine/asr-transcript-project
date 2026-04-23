# Setup Guide — DGX Spark Installation

Complete step-by-step guide to reproduce this setup from scratch on the NVIDIA DGX Spark.

## Prerequisites

- NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, Blackwell sm_121)
- NAS mounted at `${NAS_MOUNT}/` via CIFS
- Python 3.12 (system Python on Spark)
- HuggingFace account with accepted model licenses

## Step 1: Create Virtual Environment

The venv MUST be on local filesystem — NAS (CIFS) doesn't support symlinks.

```bash
python3 -m venv ${PROJECT_DIR}/.venv
source ${VENV}/bin/activate
```

## Step 2: Install PyTorch Nightly

Blackwell sm_121 requires PyTorch nightly with CUDA 12.8 support.

```bash
pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
2.12.0.dev20260223+cu128
True
NVIDIA Graphics Device
```

## Step 3: Install Transformers from Source

PyPI transformers doesn't include Parakeet CTC model class yet.

```bash
pip install git+https://github.com/huggingface/transformers.git
```

Verify:
```bash
python -c "from transformers import AutoModelForCTC; print('OK')"
```

## Step 4: Install Core Dependencies

```bash
pip install librosa numpy soundfile
```

## Step 5: Install ffmpeg

No sudo access on Spark, so use static binary.

```bash
mkdir -p ~/.local/bin
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-7.0.2-arm64-static.tar.xz
tar xf ffmpeg-7.0.2-arm64-static.tar.xz
cp ffmpeg-7.0.2-arm64-static/ffmpeg ~/.local/bin/
rm -rf ffmpeg-7.0.2-arm64-static*
```

Always include in PATH when running:
```bash
PATH=${HOME}/.local/bin:$PATH
```

## Step 6: Test Phase 1 (Timestamps Only)

```bash
PATH=${HOME}/.local/bin:$PATH ${VENV}/bin/python \
  ${NAS_MOUNT}/projects/asr-transcript-project/transcribe.py \
  --output-format timestamps --limit 5
```

This should process 5 files in ~25 seconds with no errors.

## Step 7: Install Phase 2 Dependencies (Diarization)

### torchaudio nightly

```bash
source ${VENV}/bin/activate
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### pyannote.audio 3.1.1

**IMPORTANT**: Install with `--no-deps` to avoid breaking existing PyTorch nightly.

```bash
pip install pyannote-audio==3.1.1 --no-deps
```

### pyannote dependencies

```bash
pip install asteroid-filterbanks einops "lightning>=2.0.1" speechbrain rich semver torch-audiomentations
```

## Step 8: Apply Compatibility Patches

**MUST** use the venv's Python interpreter so `sys.prefix` points to the venv.

```bash
cd ${NAS_MOUNT}/projects/asr-transcript-project

# Patch 1: pyannote + numpy + torchaudio nightly compat (9 fixes)
${VENV}/bin/python patch_pyannote.py

# Patch 2: Replace torchaudio.load/info with soundfile (4 fixes)
${VENV}/bin/python patch_torchaudio_load.py
```

Expected output for patch_pyannote.py:
```
  [1] torchaudio.set_audio_backend: io.py
  [2] torchaudio backend safe wrappers: speaker_verification.py
  [3] Created torchaudio.backend.__init__.py
  [4] Created torchaudio.backend.common stub
  [5] np.NaN -> np.nan: inference.py
  [6] np.NaN -> np.nan: speaker_diarization.py
  [7] np.NaN -> np.nan: mixins.py
  [8] use_auth_token -> token in hf_hub_download call: pipeline.py
  [9] use_auth_token -> token in hf_hub calls: model.py
  [10] torchaudio.list_audio_backends: torch_audio_backend.py
  [11] weights_only None -> False for pyannote checkpoints: cloud_io.py
  [12] pl_load weights_only=False: model.py
  [13] np.NAN -> np.nan (1 occurrences): speaker_diarization.py
  [14] np.NAN -> np.nan (1 occurrences): resegmentation.py
  [15] np.NAN -> np.nan (5 occurrences): speaker_verification.py
  [16] np.NAN -> np.nan (1 occurrences): inference.py

Done: 16 patches applied.
```

Expected output for patch_torchaudio_load.py:
```
  [1] Added soundfile import
  [2] Added _sf_load helper function (before 'class Audio:')
  [3] Replaced torchaudio.load (full load)
  [4] Replaced torchaudio.load (with frame_offset)
  [5] Replaced get_torchaudio_info to use soundfile

Done: 5 patches applied to io.py
```

## Step 9: HuggingFace Authentication

### Login
```bash
${VENV}/bin/huggingface-cli login
```

Enter your HuggingFace token when prompted.

### Accept Gated Model Licenses

Visit these URLs in a browser and click "Accept":
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1

**Note**: `pyannote/wespeaker-voxceleb-resnet34-LM` does NOT require license acceptance.

### Verify Access
```bash
${VENV}/bin/python -c "
from pyannote.audio import Pipeline
p = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=True)
print('Pipeline loaded successfully')
print(f'Pipeline type: {type(p).__name__}')
"
```

## Step 10: Test Phase 2 (Diarization)

```bash
PATH=${HOME}/.local/bin:$PATH ${VENV}/bin/python \
  ${NAS_MOUNT}/projects/asr-transcript-project/transcribe.py \
  --output-format diarized --hf-token true --limit 5 \
  --output-dir ${NAS_MOUNT}/projects/asr-transcript-project/output_v2_test
```

Expected: 5/5 files succeed, ~2-3 minutes per file.

## Troubleshooting

### "CUDA error: no kernel image is available" during diarization
This is the Blackwell FFT kernel issue. Diarization automatically runs on CPU — if you see this error, the CPU fallback in `transcribe.py` needs to be verified.

### "403 Cannot access gated repo"
You haven't accepted the model license. Visit the HuggingFace URLs above.

### "weights only load failed"
The patches weren't applied correctly. Re-run `patch_pyannote.py` with the venv Python.

### "torchcodec is required" or "libnppicc.so.12"
The `patch_torchaudio_load.py` patches weren't applied. Re-run it.

### NAS mount issues
If the NAS drops (`Host is down`), remount:
```bash
sudo mount -a
```
Or check `/etc/fstab` for the CIFS entry.

## Re-patching After Reinstall

If you reinstall pyannote or speechbrain, you must re-run both patch scripts:

```bash
source ${VENV}/bin/activate
cd ${NAS_MOUNT}/projects/asr-transcript-project
python patch_pyannote.py
python patch_torchaudio_load.py
```

The scripts are idempotent — they report `[ALREADY]` for already-applied patches.
