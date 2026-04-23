# TODO & Roadmap

## Current Status (2026-02-24)

Phase 1 (timestamps) and Phase 2 (diarization) are fully implemented and tested. Ready for production runs.

### Completed
- [x] Plain text transcription of all 1,013 files (output/)
- [x] Phase 1: CTC timestamp extraction
- [x] Phase 1: 5-file and 20-file timestamp tests (20/20 passed)
- [x] Phase 2: Speaker diarization via pyannote.audio 3.1.1
- [x] Phase 2: 5-file and 20-file diarized tests (20/20 passed)
- [x] Patch scripts for pyannote + torchaudio nightly compatibility
- [x] HuggingFace gated model access (segmentation-3.0 license accepted)
- [x] Documentation (README, CHANGELOG, SETUP_GUIDE, DEVELOPMENT_NOTES)
- [x] Project code in local-copy-github for version control
- [x] Model backups to NAS (Parakeet 8GB + segmentation 5.7MB + wespeaker 26MB)
- [x] NFS migration — Spark switched from CIFS to NFS, fstab updated
- [x] All project files synced to NAS via NFS

## Next Steps

### Immediate — Full Production Run
- [ ] **Decision**: Run timestamps-only (~1.5 hrs) or diarized (~53 hrs)?
  - Most videos in the corpus are solo — diarization adds minimal value for those
  - Consider: timestamps-only for all, then diarized for known multi-speaker videos
- [ ] Run full production batch in tmux on Spark
- [ ] Verify output quality on a sample of completed transcripts
- [ ] Archive plain text output/ (v1) before overwriting

### Short Term
- [ ] **GPU diarization**: Monitor PyTorch nightly for sm_121 FFT kernel support
  - Check: `python -c "import torch; t = torch.randn(100, device='cuda'); torch.fft.rfft(t)"`
  - When this works, remove CPU fallback in DiarizationEngine for ~10x speedup
- [ ] **Selective diarization**: Flag multi-speaker files and only diarize those
  - Could use a quick VAD pass to estimate speaker count before full diarization
- [ ] **Output post-processing**: Consider sentence casing, punctuation restoration
  - Parakeet outputs lowercase only — a small model could add punctuation

### Medium Term
- [ ] **Search index**: Build a searchable index across all 1,013 transcripts
  - Full-text search for finding specific topics across all transcribed videos
  - Could use SQLite FTS5 or Elasticsearch
- [ ] **Video-to-transcript mapping**: Link timestamps back to YouTube video URLs
  - Would allow "jump to 5:30 in this video" from transcript search results
- [ ] **Quality validation**: Compare transcripts against YouTube auto-captions
  - Spot-check accuracy, especially for technical terms (NativeScript, Angular, etc.)

### Backlog
- [ ] **Batch resume improvements**: Track per-file status in JSON (not just file existence)
- [ ] **Parallel processing**: Process multiple files concurrently (separate GPU streams for ASR)
- [ ] **Whisper comparison**: When PyTorch Blackwell kernels mature, benchmark Whisper vs Parakeet
- [ ] **pyannote 4.0+ migration**: When torchcodec gets ARM64 wheels, upgrade from 3.1.1
- [ ] **CMMC v3.0 training data**: Potential use of transcripts for fine-tuning data pipeline

## Performance Reference

| Mode | Speed | Full Run Est. |
|------|-------|---------------|
| Plain text | ~110x realtime | ~1.4 hours |
| Timestamps only | ~110x realtime | ~1.5 hours |
| Diarized (CPU) | ~3.2x realtime | ~53 hours |
| Diarized (GPU, future) | ~10x realtime (est.) | ~17 hours (est.) |
