# YOLOv8 Webカメラ物体検出 - サンプルプログラム

Webカメラをリアルタイムで使用して、YOLOv8を使った物体検出を行うPythonプログラムです。

## 環境セットアップ

### 必要なライブラリ

- ultralytics (YOLOv8)
- opencv-python (カメラキャプチャと画像処理)
- numpy

### インストール方法

```bash
# 仮想環境が作成されている場合
pip install ultralytics opencv-python numpy
```

## 使用方法

### 改善版（推奨）

```bash
python webcam_detecion.py
```

モデル選択メニューから起動時にモデルを選べます。実行中は以下のキーで調整が可能です。

- `m`: モデル変更（画面で選択）
- `c`/`x`: 信頼度を上げる/下げる（±0.05）
- `i`/`u`: IOU閾値を上げる/下げる（±0.05）

### パラメータ調整版

```bash
python webcam_detection_tunable.py
```

リアルタイムで検出パラメータを調整できます。

- `c`/`x`: 信頼度を上げる/下げる（±0.05）
- `i`/`u`: IOU閾値を上げる/下げる（±0.05）

## プログラムの説明

### 使用しているYOLOv8モデル

このサンプルではYOLOv8nanoを使用しています：

- **yolov8n.pt** - Nano版（最も軽量・高速）
- **yolov8s.pt** - Small版
- **yolov8m.pt** - Medium版
- **yolov8l.pt** - Large版
- **yolov8x.pt** - Extra Large版（最も高精度）

より高い精度が必要な場合は、プログラムの第1行目の `YOLO('yolov8n.pt')` を
変更してください。

### 主な機能

1. **リアルタイム検出**: Webカメラからのフレームを連続的に処理
2. **物体認識**: 80種類の物体クラスを認識可能
3. **バウンディングボックス**: 検出された物体の周辺に四角形を描画
4. **信頼度スコア**: 各検出の信頼度を表示

## トラブルシューティング

### Webカメラが認識されない場合

- Webカメラがパソコンに正しく接続されているか確認
- 他のアプリケーションがカメラを使用していないか確認
- `cv2.VideoCapture()` の引数を変更（0, 1, 2など）してみてください

### 処理が遅い場合

- モデルをより軽量な版（nano）に変更
- 入力フレームのサイズを小さくする
- GPU対応のマシンを使用する

### 初回実行時の注意

初回実行時は、YOLOv8モデルが自動的にダウンロードされます。
インターネット接続が必要です。

## カスタマイズ例

### 信頼度フィルタリング（検出精度の閾値を上げる）

```python
results = model(frame, conf=0.5)  # 信頼度50%以上のみ検出
```

### フレームサイズの変更

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### 特定のクラスのみを検出

```python
results = model(frame, classes=[0, 1, 2])  # クラスID 0,1,2のみ
```

## 参考リンク

- [YOLOv8 公式ドキュメント](https://docs.ultralytics.com/)
- [OpenCV ドキュメント](https://docs.opencv.org/)
