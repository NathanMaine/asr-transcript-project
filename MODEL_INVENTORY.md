# Model Inventory

All models used or downloaded during this project, their locations, sizes, and status.

## Active Models (Used by Current Version)

### 1. NVIDIA Parakeet CTC 1.1B (ASR)
- **HuggingFace ID**: `nvidia/parakeet-ctc-1.1b`
- **Purpose**: Speech-to-text transcription
- **Size**: ~8.0 GB
- **Spark cache**: `~/.cache/huggingface/hub/models--nvidia--parakeet-ctc-1.1b/`
- **NAS backup**: `${NAS_MOUNT}/projects/asr-transcript-project/models/parakeet-ctc-1.1b/`
- **Gated**: No (publicly available)
- **Notes**: FastConformer encoder + CTC decoder. English-only, lowercase output. Loaded via `AutoModelForCTC` + `AutoProcessor`.

### 2. pyannote Segmentation 3.0 (Diarization)
- **HuggingFace ID**: `pyannote/segmentation-3.0`
- **Purpose**: Speaker turn detection / voice activity detection
- **Size**: ~17 MB (pytorch_model.bin)
- **Spark cache**: `~/.cache/torch/pyannote/models--pyannote--segmentation-3.0/`
- **NAS backup**: `${NAS_MOUNT}/projects/asr-transcript-project/models/segmentation-3.0/`
- **Gated**: Yes (license accepted 2026-02-24)
- **Notes**: Core segmentation model for pyannote pipeline. Detects speech/non-speech and speaker boundaries.

### 3. pyannote WeSpeaker VoxCeleb ResNet34-LM (Diarization)
- **HuggingFace ID**: `pyannote/wespeaker-voxceleb-resnet34-LM`
- **Purpose**: Speaker embedding extraction (who is speaking)
- **Size**: ~15 MB (pytorch_model.bin)
- **Spark cache**: `~/.cache/torch/pyannote/models--pyannote--wespeaker-voxceleb-resnet34-LM/`
- **NAS backup**: `${NAS_MOUNT}/projects/asr-transcript-project/models/wespeaker-voxceleb-resnet34-LM/`
- **Gated**: No (publicly available)
- **Notes**: Speaker verification/identification model. Trained on VoxCeleb dataset. Used by pyannote for speaker clustering.

### 4. pyannote Speaker Diarization 3.1 (Pipeline Config)
- **HuggingFace ID**: `pyannote/speaker-diarization-3.1`
- **Purpose**: Pipeline configuration (references models 2 and 3 above)
- **Size**: ~28 KB (config.yaml only, no weights)
- **Spark cache**: `~/.cache/huggingface/hub/models--pyannote--speaker-diarization-3.1/` (config only)
- **Gated**: Yes (license accepted 2026-02-24)
- **Notes**: This is NOT a neural network — it's a pipeline config that orchestrates segmentation + embedding models. The actual weights come from models 2 and 3.

## Inactive Models (Downloaded but Not Used)

### 5. Systran faster-whisper-large-v3
- **HuggingFace ID**: `Systran/faster-whisper-large-v3`
- **Purpose**: Was initial ASR attempt
- **Size**: ~2.9 GB
- **Spark cache**: `~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/`
- **Status**: UNUSED — abandoned because ctranslate2 has no aarch64+CUDA wheels
- **Can delete**: Yes, safe to remove

## Storage Summary

| Model | Size | Status | NAS Backed Up |
|-------|------|--------|---------------|
| Parakeet CTC 1.1B | 8.0 GB | Active (ASR) | Yes |
| pyannote segmentation-3.0 | ~17 MB | Active (diarization) | Yes |
| pyannote wespeaker-voxceleb | ~15 MB | Active (diarization) | Yes |
| pyannote speaker-diarization-3.1 | ~28 KB | Active (config) | Yes |
| faster-whisper-large-v3 | 2.9 GB | Inactive | No (can delete) |

### Total Active Model Storage
- Spark local cache: ~8.0 GB
- NAS backup: ~8.0 GB

## NAS Backup Location

```
${NAS_MOUNT}/projects/asr-transcript-project/models/
├── parakeet-ctc-1.1b/          # Full HuggingFace cache copy
├── segmentation-3.0/           # pytorch_model.bin + config
└── wespeaker-voxceleb-resnet34-LM/  # pytorch_model.bin + config
```

## Restoring Models from NAS Backup

If the Spark's local cache is cleared, restore by copying back:
```bash
# Parakeet (HuggingFace hub format)
cp -r ${NAS_MOUNT}/projects/asr-transcript-project/models/parakeet-ctc-1.1b \
  ~/.cache/huggingface/hub/models--nvidia--parakeet-ctc-1.1b

# pyannote models (torch cache format)
cp -r ${NAS_MOUNT}/projects/asr-transcript-project/models/segmentation-3.0 \
  ~/.cache/torch/pyannote/models--pyannote--segmentation-3.0

cp -r ${NAS_MOUNT}/projects/asr-transcript-project/models/wespeaker-voxceleb-resnet34-LM \
  ~/.cache/torch/pyannote/models--pyannote--wespeaker-voxceleb-resnet34-LM
```

Or simply re-download (requires internet + HuggingFace login for gated models):
```bash
python -c "
from transformers import AutoModelForCTC, AutoProcessor
AutoProcessor.from_pretrained('nvidia/parakeet-ctc-1.1b')
AutoModelForCTC.from_pretrained('nvidia/parakeet-ctc-1.1b')
print('Parakeet downloaded')
"

python -c "
from pyannote.audio import Pipeline
Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=True)
print('pyannote pipeline downloaded')
"
```
