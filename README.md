# Stereo Camera Calibration / ステレオカメラキャリブレーション

Stereo calibration pipeline for a dual-camera setup (EoSens 2.0MCX12 + MC2066) using OpenCV.

デュアルカメラ（EoSens 2.0MCX12 + MC2066）のステレオキャリブレーションパイプライン（OpenCV使用）。

## What it does / 機能

1. Loads left/right checkerboard image pairs / 左右のチェッカーボード画像ペアを読み込む
2. Detects corners and performs individual camera calibration / コーナー検出と個別カメラキャリブレーション
3. Runs stereo calibration to find relative pose (R, T) / ステレオキャリブレーションで相対姿勢（R, T）を算出
4. Computes rectification maps and epipolar error / 平行化マップとエピポーラ誤差を計算
5. Saves calibration parameters (`.npz`) and generates a report / キャリブレーションパラメータを保存し、レポートを生成

## Setup / セットアップ

```bash
pip install opencv-python matplotlib numpy
```

## Usage / 使い方

1. Place checkerboard image pairs in a folder (left = `D1`, right = `D0` in filename)
   チェッカーボード画像ペアをフォルダに配置（左 = `D1`、右 = `D0`）
2. Update `CALIB_DIR` in `stereo_calibration.py` to point to your image folder
   `stereo_calibration.py` の `CALIB_DIR` を画像フォルダのパスに変更
3. Run / 実行:

```bash
python stereo_calibration.py
```

## Configuration / 設定

| Parameter / パラメータ | Default / デフォルト | Description / 説明 |
|-----------|---------|-------------|
| `CHECKERBOARD` | `(6, 8)` | Inner corner count (cols, rows) / 内側コーナー数（列, 行） |
| `SQUARE_SIZE` | `10` | Square size in mm / 正方形のサイズ（mm） |

## Output / 出力

- `stereo_calibration.npz` - intrinsics, extrinsics, rectification maps / 内部・外部パラメータ、平行化マップ
- `calibration_report.txt` - human-readable quality report / キャリブレーション品質レポート
