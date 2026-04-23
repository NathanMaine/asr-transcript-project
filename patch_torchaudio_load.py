#!/usr/bin/env python3
"""Patch pyannote.audio io.py to use soundfile instead of torchaudio.load/info.

torchaudio nightly requires torchcodec which needs system CUDA NPP libraries
(libnppicc.so.12) that may not be available. This patches io.py to use
soundfile directly, which is already installed and works with any audio format
that ffmpeg can handle (mp3, wav, flac, etc.).

Patches:
1. Adds soundfile import
2. Adds _sf_load() as drop-in replacement for torchaudio.load()
3. Replaces torchaudio.load() calls with _sf_load()
4. Replaces get_torchaudio_info() to use soundfile + AudioMetaData stub
"""
import os
import sys

SITE_PACKAGES = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
IO_PATH = os.path.join(SITE_PACKAGES, "pyannote", "audio", "core", "io.py")

patches = 0

with open(IO_PATH, "r") as f:
    content = f.read()

# 1. Add soundfile import if not already present
if "import soundfile as sf" not in content:
    content = content.replace(
        "import torchaudio",
        "import torchaudio\nimport soundfile as sf",
    )
    patches += 1
    print(f"  [{patches}] Added soundfile import")
else:
    print("  [ALREADY] soundfile import")

# 2. Add _sf_load helper function before the Audio class
helper = '''

def _sf_load(path, frame_offset=0, num_frames=-1):
    """Load audio using soundfile (drop-in replacement for torchaudio.load)."""
    if frame_offset > 0:
        with sf.SoundFile(path) as f:
            f.seek(frame_offset)
            if num_frames > 0:
                data = f.read(num_frames, dtype="float32")
            else:
                data = f.read(dtype="float32")
            sample_rate = f.samplerate
    else:
        if num_frames > 0:
            with sf.SoundFile(path) as f:
                data = f.read(num_frames, dtype="float32")
                sample_rate = f.samplerate
        else:
            data, sample_rate = sf.read(path, dtype="float32")
    # soundfile returns (samples, channels), torch expects (channels, samples)
    import torch
    import numpy as np
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.T)
    return waveform, sample_rate

'''

if "_sf_load" not in content:
    # Try both patterns: "class Audio(" and "class Audio:"
    if "\nclass Audio:" in content:
        content = content.replace("\nclass Audio:", helper + "\nclass Audio:")
        patches += 1
        print(f"  [{patches}] Added _sf_load helper function (before 'class Audio:')")
    elif "\nclass Audio(" in content:
        content = content.replace("\nclass Audio(", helper + "\nclass Audio(")
        patches += 1
        print(f"  [{patches}] Added _sf_load helper function (before 'class Audio(')")
    else:
        print("  [ERROR] Could not find Audio class to insert _sf_load before")
else:
    print("  [ALREADY] _sf_load helper function")

# 3. Replace torchaudio.load calls with _sf_load
# First call: simple full load
old1 = '            waveform, sample_rate = torchaudio.load(file["audio"])'
new1 = '            waveform, sample_rate = _sf_load(file["audio"])'
if old1 in content:
    content = content.replace(old1, new1)
    patches += 1
    print(f"  [{patches}] Replaced torchaudio.load (full load)")
elif new1 in content:
    print("  [ALREADY] torchaudio.load (full load) replaced")

# Second call: with frame_offset and num_frames
old2 = (
    '                data, _ = torchaudio.load(\n'
    '                    file["audio"], frame_offset=start_frame, num_frames=num_frames\n'
    '                )'
)
new2 = (
    '                data, _ = _sf_load(\n'
    '                    file["audio"], frame_offset=start_frame, num_frames=num_frames\n'
    '                )'
)
if old2 in content:
    content = content.replace(old2, new2)
    patches += 1
    print(f"  [{patches}] Replaced torchaudio.load (with frame_offset)")
elif new2 in content:
    print("  [ALREADY] torchaudio.load (with frame_offset) replaced")

# 4. Replace get_torchaudio_info to use soundfile instead of torchaudio.info
# The original function calls torchaudio.info() which is removed in nightly.
old_info = """def get_torchaudio_info(file):
    \"\"\"Protocol-aware version of torchaudio.info

    Parameters
    ----------
    file : AudioFile

    Returns
    -------
    info : torchaudio.backend.common.AudioMetaData
    \"\"\"

    info = torchaudio.info(file["audio"])"""

new_info = """def get_torchaudio_info(file):
    \"\"\"Protocol-aware version of torchaudio.info

    Parameters
    ----------
    file : AudioFile

    Returns
    -------
    info : torchaudio.backend.common.AudioMetaData
    \"\"\"

    # Use soundfile instead of torchaudio.info (removed in torchaudio nightly)
    _info = sf.info(file["audio"])
    from torchaudio.backend.common import AudioMetaData
    info = AudioMetaData(
        sample_rate=_info.samplerate,
        num_frames=_info.frames,
        num_channels=_info.channels,
        bits_per_sample=_info.subtype_info.split()[0] if _info.subtype_info else 16,
        encoding=_info.subtype or "",
    )"""

if "info = torchaudio.info(file" in content:
    content = content.replace(old_info, new_info)
    patches += 1
    print(f"  [{patches}] Replaced get_torchaudio_info to use soundfile")
elif "info = sf.info(file" in content or "_info = sf.info(file" in content:
    print("  [ALREADY] get_torchaudio_info uses soundfile")
else:
    # Fallback: try a simpler pattern match
    if "torchaudio.info(file" in content:
        print("  [WARN] torchaudio.info found but pattern didn't match exactly. Manual fix may be needed.")
    else:
        print("  [ALREADY] No torchaudio.info calls found")

with open(IO_PATH, "w") as f:
    f.write(content)

print(f"\nDone: {patches} patches applied to io.py")
