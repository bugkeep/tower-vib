## tower-vib

Industrial-grade video vibrometry: phase-based subpixel displacement, no-GT consistency metrics, and robustness benchmarks.

## Overview

This repo measures **transmission tower vibration** from video.
It extracts a **golden ROI**, estimates **subpixel displacement** using **phase-based methods**, and outputs **physics-oriented features** (e.g., vibration frequency via PSD).
It also provides **no-ground-truth (no-GT)** consistency metrics and **robustness benchmarks** under degradations.

## Key features

- **Subpixel motion** from phase correlation / phase consistency (robust under illumination & texture changes)
- **Physics features**: displacement curve, PSD, peak frequency
- **No-GT self-evaluation**: temporal / resampling consistency without ground truth
- **Robustness benchmarks**: blur / compression / jitter / noise degradations

## Quick start

### 1) Install

```bash
pip install -r requirements.txt
```

### 2)Cache golden ROI sequence(200 frames)

```bash
python scripts/run_pipeline.py
```

## Expected output:

- `results/cache/day28_frames.npz`  (shape: ` (200,H,W)`e.g. ` (200,128,128) `)

- ` results/logs/*.log` 
  TODO: set ROI in ` scripts/run_pipeline.py` as `(x,y,w,h)`
  Example:` (200,100,128,128)`

## Data format

`results/cache/day28_frames.npz` contains:

- `roi` or `grays`: ROI grayscale sequence, shape `(T, H, W)` (e.g. `(200, 128, 128)`)
- `fps`: frames per second
- `roi_xywh`: ROI coordinates `(x, y, w, h)`

## Reproducibility

- All experiments are designed to be reproducible from cached ROI ` .npz`.
- Logs record ` video_path`,` fps`,` roi_xywh`,and output paths.

## Roadmap

- [ ] 2026/1/13:cache 200-frame golden ROI sequence to `.npz` with logging




