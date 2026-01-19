## tower-vib
Active development. No license granted yet.
Industrial-grade video vibrometry (WIP): ROI caching, phase-based subpixel displacement, and physics-oriented vibration features.

## Overview

This repo measures **transmission tower vibration** from video.  
Current pipeline:

1) Extract and cache a **golden ROI** video clip (default 200 frames) to `.npz`
2) Run a lightweight **ROI mean-curve sanity check** and export `.png/.csv`

## Repo structure

- `data/` : input videos (e.g. `demo_tower.mp4`)
- `scripts/`
  - `extract_roi.py` : cache ROI frames to `results/cache/*.npz`
  - `roi_mean_curve_check.py` : compute ROI mean curve and save plot/csv
  - `run_pipeline.py` : end-to-end runner (extract ROI + mean curve)
- `results/`
  - `cache/` : cached ROI `.npz`
  - `plots/` : mean curve plot `.png`
  - `data/`  : mean curve `.csv`
  - `logs/`  : run logs

## Quick start

### 1) Install

```bash
pip install -r requirements.txt
```

## 2）Run pipeline(cache ROI + mean curve)

```bash
python scripts/run_pipeline.py
```

# Excepted outputs

After running,you should get:

- ` results/cache/YYYY-MM-DD_main_roi_frames.npz`

- ` results/plots/YYYY-MM-DD_mean_curve.png`

- ` results/data/YYYY-MM-DD_mean_curve.csv`

- ` results/logs/*.log`

# NPZ format

` results/cache/*_main_roi_frames.npz`contains:

- `frames`:ROI frame sequence
  
  - shape:`(T,H,W,3)`for color ROI (OpenCV default BGR)
  
  - (optional future)`(T,H,W)`for grayscale ROI

- `roi`:ROI coordinates`(x,y,w,h)`

- `video`:source video path

- `fps`:frames per second

# Notes

- ROI mean curve is used as a sanity check:
  
  - verifies ROI extraction correctness
  
  - help detect abnormal frames via sudden instensity jumps

- ROI is currently set in`scripts/run_pipeline.py`as`(x,y,w,h)`

- 
