# Stereo Camera Calibration

Stereo calibration pipeline for a dual-camera setup (EoSens 2.0MCX12 + MC2066) using OpenCV.

## What it does

1. Loads left/right checkerboard image pairs
2. Detects corners and performs individual camera calibration
3. Runs stereo calibration to find relative pose (R, T)
4. Computes rectification maps and epipolar error
5. Saves calibration parameters (`.npz`) and generates a report

## Setup

```bash
pip install opencv-python matplotlib numpy
```

## Usage

1. Place checkerboard image pairs in a folder (left = `D1`, right = `D0` in filename)
2. Update `CALIB_DIR` in `stereo_calibration.py` to point to your image folder
3. Run:

```bash
python stereo_calibration.py
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHECKERBOARD` | `(6, 8)` | Inner corner count (cols, rows) |
| `SQUARE_SIZE` | `10` | Square size in mm |

## Output

- `stereo_calibration.npz` — intrinsics, extrinsics, rectification maps
- `calibration_report.txt` — human-readable quality report
