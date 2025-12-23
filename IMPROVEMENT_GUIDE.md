# YOLOv8 認識精度改善ガイド

## 現在の問題

YOLOv8n（Nano版）は軽量ですが、精度が低めです。改善するための複数の方法があります。

## 改善方法

### 1. **より大きなモデルを使用（最も効果的）** ⭐推奨

モデルのサイズ比較：

| モデル | パラメータ数 | 速度 | 精度 | GPU VRAM | 用途 |
| ------ | ---------- | ------ | ------ | --------- | ------ |
| **nano (n)** | 2.6M | 最速 | 低 | ~1GB | 軽量デバイス |
| **small (s)** | 11.2M | 高速 | 中 | ~2GB | **推奨** |
| **medium (m)** | 25.9M | 中 | 高 | ~3GB | **推奨** |
| **large (l)** | 43.7M | 低速 | 非常に高 | ~4GB | 高精度要求 |
| **xlarge (x)** | 68.2M | 最低速 | 最高 | ~6GB | 最高精度要求 |

**使用方法：**

```python
# Small版に変更（推奨）
model = YOLO('yolov8s.pt')

# または Medium版
model = YOLO('yolov8m.pt')
```

### 2. **信頼度閾値を調整**

信頼度が低い検出は誤検出の可能性があります。

```python
# デフォルト: conf=0.25
# 推奨: conf=0.5
results = model(frame, conf=0.5)
```

- **conf=0.25**: より多くの物体を検出（誤検出が多い）
- **conf=0.5**: バランス型（推奨）
- **conf=0.75**: 高精度のみを検出（検出漏れが多い）

### 3. **IOU（Intersection over Union）閾値を調整**

重複した検出結果を統合するための閾値です。

```python
# デフォルト: iou=0.7
# 推奨: iou=0.45
results = model(frame, conf=0.5, iou=0.45)
```

- **iou=0.7**: 厳しい（検出数が少ない）
- **iou=0.45**: バランス型（推奨）
- **iou=0.1**: 緩い（重複検出が多い）

### 4. **入力画像サイズを大きくする**

より詳細な情報を処理できます。

```python
# 推奨: 640x640
results = model(frame, imgsz=640)

# または最大で:
results = model(frame, imgsz=1280)
```

ただし、処理時間が増加するため、FPSが低下します。

### 5. **フレーム前処理**

画像の品質を改善：

```python
# コントラストを調整
import cv2
import numpy as np

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
frame_enhanced = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

# ノイズ除去
frame = cv2.fastNlMeansDenoising(frame)
```

### 6. **カメラ設定の最適化**

```python
# 解像度を上げる
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 明るさを調整
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)

# コントラストを調整
cap.set(cv2.CAP_PROP_CONTRAST, 0.5)

# フォーカスを固定
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
```

## 提供されるプログラム

### 1. `webcam_detecion.py` - 改善版（推奨）

YOLOv8m（Medium版）を使用した改善版です。

- 精度が大幅に向上
- バランスの取れた性能

実行方法：

```bash
python webcam_detecion.py
```

### 2. `webcam_detection_tunable.py` - パラメータ調整版

リアルタイムでパラメータを調整できます。

実行方法：

```bash
python webcam_detection_tunable.py
```

キーボード操作：

- `q`: 終了
- `c`: 信頼度を上げる（+0.05）
- `x`: 信頼度を下げる（-0.05）
- `i`: IOU閾値を上げる（+0.05）
- `u`: IOU閾値を下げる（-0.05）

## 精度比較（参考値）

典型的な改善効果（COCO dataset）：

| 設定 | 精度（mAP） | 処理速度 |
| ------ | ----------- | --------- |
| yolov8n, conf=0.25 | 37.3% | 最速 |
| yolov8n, conf=0.5 | 36.2% | 最速 |
| **yolov8s, conf=0.5** | **46.5%** | **高速** |
| yolov8m, conf=0.5 | 50.2% | 中速 |
| yolov8l, conf=0.5 | 52.8% | 低速 |

## トラブルシューティング

### 処理が遅い場合

1. モデルをより小さいサイズに変更（n → s）
2. 入力画像サイズを小さくする（1280 → 640）
3. フレームレートを制限する
4. GPU対応マシンを使用する

### 検出漏れが多い場合

1. 信頼度を下げる（0.5 → 0.3）
2. IOU閾値を下げる（0.45 → 0.25）
3. より大きなモデルを使用

### 誤検出が多い場合

1. 信頼度を上げる（0.5 → 0.7）
2. IOU閾値を上げる（0.45 → 0.65）
3. 撮影環境を改善（照明を増やすなど）

## 最も効果的な改善手順

1. **まずこれをやる**: `yolov8s` または `yolov8m` に変更
2. **次に**: 信頼度を 0.5 に設定
3. **それでも改善が必要なら**: パラメータ調整版でリアルタイム調整
4. **それでも不十分なら**: さらに大きなモデル（yolov8l）を試す

## 参考リンク

- [YOLOv8 公式ドキュメント](https://docs.ultralytics.com/)
- [YOLO パラメータ詳細](https://docs.ultralytics.com/usage/cfg/)
