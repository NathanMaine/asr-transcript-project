#!/usr/bin/env python3
"""Patch pyannote.audio 3.1.1 for compatibility with torchaudio nightly and numpy 2.x.

Run this AFTER a clean install of pyannote-audio==3.1.1 and speechbrain.
Must be run with the venv's Python interpreter so sys.prefix points to the venv.

Fixes:
1. torchaudio.set_audio_backend / get_audio_backend removed in nightly
2. torchaudio.backend.common.AudioMetaData removed in nightly
3. np.NaN removed in numpy 2.0
4. use_auth_token -> token ONLY for direct huggingface_hub API calls (hf_hub_download)
5. speechbrain torchaudio.list_audio_backends removed in nightly
6. torch.load weights_only=True default in PyTorch 2.6+ breaks pyannote checkpoint loading
7. model.py pl_load needs weights_only=False for pyannote checkpoints
8. np.NAN (uppercase) also removed in numpy 2.0 (different from np.NaN)

After running this script, also run patch_torchaudio_load.py to replace
torchaudio.load/torchaudio.info with soundfile equivalents.
"""
import os
import re
import sys

SITE_PACKAGES = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
PYANNOTE_DIR = os.path.join(SITE_PACKAGES, "pyannote", "audio")
SPEECHBRAIN_DIR = os.path.join(SITE_PACKAGES, "speechbrain")

patches_applied = 0


def patch_file(filepath, old, new, description=""):
    global patches_applied
    if not os.path.exists(filepath):
        print(f"  [SKIP] {description}: {filepath} not found")
        return False
    with open(filepath, "r") as f:
        content = f.read()
    if old not in content:
        if new in content:
            print(f"  [ALREADY] {description}: {os.path.basename(filepath)}")
        else:
            print(f"  [SKIP] {description}: pattern not found in {os.path.basename(filepath)}")
        return False
    content = content.replace(old, new)
    with open(filepath, "w") as f:
        f.write(content)
    patches_applied += 1
    print(f"  [{patches_applied}] {description}: {os.path.basename(filepath)}")
    return True


# --- Fix 1: torchaudio.set_audio_backend in io.py ---
patch_file(
    os.path.join(PYANNOTE_DIR, "core", "io.py"),
    'torchaudio.set_audio_backend("soundfile")',
    'try:\n    torchaudio.set_audio_backend("soundfile")\nexcept (AttributeError, RuntimeError):\n    pass',
    "torchaudio.set_audio_backend",
)

# --- Fix 2: torchaudio backend calls in speaker_verification.py ---
# The original code is:
#   backend = torchaudio.get_audio_backend()
#   try: ... finally: torchaudio.set_audio_backend(backend)
# Use getattr pattern to safely wrap removed functions
sv_path = os.path.join(PYANNOTE_DIR, "pipelines", "speaker_verification.py")
if os.path.exists(sv_path):
    with open(sv_path, "r") as f:
        sv_content = f.read()
    changed = False
    if "backend = torchaudio.get_audio_backend()" in sv_content:
        sv_content = sv_content.replace(
            "backend = torchaudio.get_audio_backend()",
            "backend = getattr(torchaudio, 'get_audio_backend', lambda: None)()",
        )
        changed = True
    if "    torchaudio.set_audio_backend(backend)" in sv_content:
        sv_content = sv_content.replace(
            "    torchaudio.set_audio_backend(backend)",
            "    getattr(torchaudio, 'set_audio_backend', lambda b: None)(backend)",
        )
        changed = True
    if changed:
        with open(sv_path, "w") as f:
            f.write(sv_content)
        patches_applied += 1
        print(f"  [{patches_applied}] torchaudio backend safe wrappers: speaker_verification.py")
    else:
        print(f"  [ALREADY] torchaudio backend wrappers: speaker_verification.py")

# --- Fix 3: torchaudio.backend.common.AudioMetaData stub ---
backend_dir = os.path.join(SITE_PACKAGES, "torchaudio", "backend")
os.makedirs(backend_dir, exist_ok=True)
init_path = os.path.join(backend_dir, "__init__.py")
if not os.path.exists(init_path):
    with open(init_path, "w") as f:
        f.write("# Stub for pyannote.audio 3.1 compatibility\n")
    patches_applied += 1
    print(f"  [{patches_applied}] Created torchaudio.backend.__init__.py")
else:
    print(f"  [ALREADY] torchaudio.backend.__init__.py exists")

common_path = os.path.join(backend_dir, "common.py")
if not os.path.exists(common_path):
    with open(common_path, "w") as f:
        f.write('''# Stub for backward compatibility
from dataclasses import dataclass

@dataclass
class AudioMetaData:
    sample_rate: int = 0
    num_frames: int = 0
    num_channels: int = 0
    bits_per_sample: int = 0
    encoding: str = ""
    codec: str = ""
''')
    patches_applied += 1
    print(f"  [{patches_applied}] Created torchaudio.backend.common stub")
else:
    print(f"  [ALREADY] torchaudio.backend.common.py exists")

# --- Fix 4: np.NaN -> np.nan ---
for rel_path in [
    "core/inference.py",
    "tasks/segmentation/speaker_diarization.py",
    "tasks/segmentation/mixins.py",
]:
    path = os.path.join(PYANNOTE_DIR, rel_path)
    if os.path.exists(path):
        patch_file(path, "np.NaN", "np.nan", "np.NaN -> np.nan")

# --- Fix 5: use_auth_token -> token ONLY for hf_hub_download() calls ---
# IMPORTANT: pyannote's own function signatures, class constructors, and internal
# variable passing all use 'use_auth_token'. We must NOT change those.
# We ONLY change the arguments passed directly to huggingface_hub API functions
# (hf_hub_download) which dropped use_auth_token in favor of token.

# pipeline.py: hf_hub_download(... use_auth_token=use_auth_token, ...)
# The closing ) is NOT immediately after use_auth_token due to commented-out lines,
# so we use a simple line-level replacement.
pipeline_path = os.path.join(PYANNOTE_DIR, "core", "pipeline.py")
patch_file(
    pipeline_path,
    "                    use_auth_token=use_auth_token,",
    "                    token=use_auth_token,",
    "use_auth_token -> token in hf_hub_download call",
)
# NOTE: Do NOT change params.setdefault("use_auth_token", ...) because params
# gets unpacked into SpeakerDiarization.__init__() which expects use_auth_token.

# model.py: hf_hub_download(... use_auth_token=use_auth_token)
model_path = os.path.join(PYANNOTE_DIR, "core", "model.py")
if os.path.exists(model_path):
    with open(model_path, "r") as f:
        content = f.read()
    if "use_auth_token=use_auth_token" in content:
        # Only replace in hf_hub_download context, not function signatures
        content = re.sub(
            r'(hf_hub_download\([^)]*?)use_auth_token=use_auth_token',
            r'\1token=use_auth_token',
            content,
        )
        with open(model_path, "w") as f:
            f.write(content)
        patches_applied += 1
        print(f"  [{patches_applied}] use_auth_token -> token in hf_hub calls: model.py")
    else:
        print(f"  [ALREADY] model.py hf_hub_download calls already patched")

# NOTE: getter.py calls Model.from_pretrained() (pyannote's own API) which accepts
# use_auth_token in its signature and converts internally. Do NOT change getter.py.

# --- Fix 6: speechbrain torchaudio.list_audio_backends ---
sb_backend_path = os.path.join(SPEECHBRAIN_DIR, "utils", "torch_audio_backend.py")
if os.path.exists(sb_backend_path):
    patch_file(
        sb_backend_path,
        "        available_backends = torchaudio.list_audio_backends()",
        "        try:\n            available_backends = torchaudio.list_audio_backends()\n        except AttributeError:\n            return",
        "torchaudio.list_audio_backends",
    )

# --- Fix 7: torch.load weights_only default in PyTorch 2.6+ ---
# PyTorch 2.6+ changed torch.load to default weights_only=True, but pyannote
# checkpoints contain TorchVersion globals that aren't in the safe list.
# Fix lightning_fabric to default weights_only=False when None is passed.
cloud_io_path = os.path.join(SITE_PACKAGES, "lightning_fabric", "utilities", "cloud_io.py")
patch_file(
    cloud_io_path,
    "    if not isinstance(path_or_url, (str, Path)):",
    "    # Pyannote compat: PyTorch 2.6+ defaults weights_only=True,\n"
    "    # but pyannote checkpoints require weights_only=False\n"
    "    if weights_only is None:\n"
    "        weights_only = False\n"
    "\n"
    "    if not isinstance(path_or_url, (str, Path)):",
    "weights_only None -> False for pyannote checkpoints",
)

# --- Fix 8: model.py pl_load needs weights_only=False ---
patch_file(
    model_path,
    "loaded_checkpoint = pl_load(path_for_pl, map_location=map_location)",
    "loaded_checkpoint = pl_load(path_for_pl, map_location=map_location, weights_only=False)",
    "pl_load weights_only=False",
)

# --- Fix 9: np.NAN (uppercase) -> np.nan ---
# numpy 2.0 removed both np.NaN and np.NAN. Fix 4 handles np.NaN,
# but several files also use np.NAN (fully uppercase).
for rel_path in [
    "pipelines/speaker_diarization.py",
    "pipelines/resegmentation.py",
    "pipelines/speaker_verification.py",
    "core/inference.py",
]:
    path = os.path.join(PYANNOTE_DIR, rel_path)
    if os.path.exists(path):
        with open(path, "r") as f:
            content = f.read()
        if "np.NAN" in content:
            count = content.count("np.NAN")
            content = content.replace("np.NAN", "np.nan")
            with open(path, "w") as f:
                f.write(content)
            patches_applied += 1
            print(f"  [{patches_applied}] np.NAN -> np.nan ({count} occurrences): {os.path.basename(path)}")
        elif "np.NAN" not in content and "np.nan" in content:
            print(f"  [ALREADY] np.NAN -> np.nan: {os.path.basename(path)}")

print(f"\nDone: {patches_applied} patches applied.")
