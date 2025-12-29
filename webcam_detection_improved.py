import cv2
import time
import numpy as np
import os
import platform
from ultralytics import YOLO

# プラットフォーム検出
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

# Windows環境でMSMFカメラバックエンドを使用（DirectShow より高速）
if IS_WINDOWS:
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '1'

# PyTorch CPU最適化
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count() or 4)

# PyTorchのスレッド数を設定
try:
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_num_interop_threads(max(1, (os.cpu_count() or 4) // 2))
except:
    pass

print("✓ CPU最適化モードを使用")
print(f"  プラットフォーム: {platform.system()} {platform.release()}")
print(f"  スレッド数: {os.cpu_count()} cores")

# カメラ読み込みを別スレッドで行うクラス（遅延削減）
import threading
from queue import Queue

class ThreadedCamera:
    def __init__(self, src=0):
        # プラットフォームに応じたバックエンドを選択
        if IS_WINDOWS:
            self.cap = cv2.VideoCapture(src, cv2.CAP_MSMF)
        elif IS_LINUX:
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)  # LinuxではV4L2を使用
        else:
            self.cap = cv2.VideoCapture(src)  # macOSなど
        
        # カメラ設定
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 最新フレームのみ保持
        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.lock = threading.Lock()
        
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
    
    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.stopped = True
        self.cap.release()
    
    def isOpened(self):
        return self.cap.isOpened()

# デバイス検出関数（CPUに固定）
def detect_best_device():
    """利用可能な最適なデバイスを検出"""
    # このハードウェアではCPUが最速
    return 'cpu'

# モデルウォームアップ（簡略版）
def warmup_model(model, imgsz=640):
    """モデルをウォームアップ"""
    try:
        print("  モデルウォームアップ中...")
        dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(2):
            _ = model(dummy_frame, imgsz=imgsz, verbose=False)
        print("  ✓ ウォームアップ完了")
    except:
        pass

def print_model_menu():
    """モデル選択メニューを表示"""
    print("\n" + "=" * 70)
    print("【モデル選択】")
    print("=" * 70)
    print("\n【標準YOLOv8モデル - 80クラス検出】")
    print("  1: yolov8n.pt    - Nano版（軽量、低精度）")
    print("  2: yolov8s.pt    - Small版（バランス型）")
    print("  3: yolov8m.pt    - Medium版（高精度）")
    print("  4: yolov8l.pt    - Large版（非常に高精度）")
    print("  5: yolov8x.pt    - Extra Large版（最高精度）")
    print("\n【YOLO-World - オープンボキャブラリー検出】")
    print("  6: yolov8s-world.pt - Small版（軽量）")
    print("  7: yolov8m-world.pt - Medium版")
    print("  8: yolov8l-world.pt - Large版")
    print("  9: yolov8x-world.pt - Extra Large版")
    print("\n【セグメンテーション版 - 輪郭検出対応】")
    print(" 10: yolov8s-seg.pt - Small版")
    print(" 11: yolov8m-seg.pt - Medium版")
    print(" 12: yolov8l-seg.pt - Large版")
    print("\n【姿勢推定版 - 骨格検出】")
    print(" 13: yolov8s-pose.pt - Small版")
    print(" 14: yolov8m-pose.pt - Medium版")
    print("\n【YOLOv9 - より高精度】")
    print(" 15: yolov9c.pt - Compact版")
    print("\n【RT-DETR - 境界ボックス最適化】")
    print(" 16: rtdetr-l.pt - Large版")
    print("\n【YOLO11 - 最新版(高速・高精度)】")
    print(" 17: yolo11n.pt - Nano版")
    print(" 18: yolo11s.pt - Small版")
    print(" 19: yolo11m.pt - Medium版")
    print(" 20: yolo11l.pt - Large版")
    print(" 21: yolo11x.pt - Extra Large版")
    print("=" * 70)

def select_model_at_startup():
    """起動時にモデルを選択"""
    model_map = {
        '1': 'yolov8n.pt',
        '2': 'yolov8s.pt',
        '3': 'yolov8m.pt',
        '4': 'yolov8l.pt',
        '5': 'yolov8x.pt',
        '6': 'yolov8s-world.pt',
        '7': 'yolov8m-world.pt',
        '8': 'yolov8l-world.pt',
        '9': 'yolov8x-world.pt',
        '10': 'yolov8s-seg.pt',
        '11': 'yolov8m-seg.pt',
        '12': 'yolov8l-seg.pt',
        '13': 'yolov8s-pose.pt',
        '14': 'yolov8m-pose.pt',
        '15': 'yolov9c.pt',
        '16': 'rtdetr-l.pt',
        '17': 'yolo11n.pt',
        '18': 'yolo11s.pt',
        '19': 'yolo11m.pt',
        '20': 'yolo11l.pt',
        '21': 'yolo11x.pt',
    }
    
    print_model_menu()
    
    while True:
        choice = input("\nモデルを選択してください (1-21): ").strip()
        if choice in model_map:
            return model_map[choice]
        else:
            print("❌ 無効な選択です。1-21の数字を入力してください。")

def select_model_interactive():
    """実行中にモデルを選択（OpenCVウィンドウ上でキー入力）"""
    model_map = {
        '1': 'yolov8n.pt',
        '2': 'yolov8s.pt',
        '3': 'yolov8m.pt',
        '4': 'yolov8l.pt',
        '5': 'yolov8x.pt',
        '6': 'yolov8s-world.pt',
        '7': 'yolov8m-world.pt',
        '8': 'yolov8l-world.pt',
        '9': 'yolov8x-world.pt',
        '10': 'yolov8s-seg.pt',
        '11': 'yolov8m-seg.pt',
        '12': 'yolov8l-seg.pt',
        '13': 'yolov8s-pose.pt',
        '14': 'yolov8m-pose.pt',
        '15': 'yolov9c.pt',
        '16': 'rtdetr-l.pt',
        '17': 'yolo11n.pt',
        '18': 'yolo11s.pt',
        '19': 'yolo11m.pt',
        '20': 'yolo11l.pt',
        '21': 'yolo11x.pt',
    }

    # ガイド用のウィンドウを表示
    h, w = 720, 1280
    base = np.zeros((h, w, 3), dtype=np.uint8)
    window_name = 'Model Select'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def render(typed: str, message: str = ""):
        img = base.copy()
        y = 40
        cv2.putText(img, "モデル選択: 数字を入力しEnterで確定 / Esc取消", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 40
        cv2.putText(img, "1-5: YOLOv8 (n/s/m/l/x)", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        y += 35
        cv2.putText(img, "6-9: YOLO-World (s/m/l/x)", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        y += 35
        cv2.putText(img, "10-12: YOLOv8-Seg (s/m/l)", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        y += 35
        cv2.putText(img, "13-14: YOLOv8-Pose (s/m)", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        y += 35
        cv2.putText(img, "15: YOLOv9c, 16: RT-DETR-l", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        y += 35
        cv2.putText(img, "17-21: YOLO11 (n/s/m/l/x)", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        y += 50
        cv2.putText(img, f"入力: {typed}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        y += 40
        if message:
            cv2.putText(img, message, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
        cv2.imshow(window_name, img)

    typed = ""
    render(typed)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):  # Esc or q
            cv2.destroyWindow(window_name)
            return None
        elif ord('0') <= key <= ord('9'):
            if len(typed) < 2:
                typed += chr(key)
                render(typed)
            else:
                render(typed, "2桁まで入力できます。Enterで確定してください。")
        elif key in (13, 10):  # Enter
            if typed in model_map:
                cv2.destroyWindow(window_name)
                return model_map[typed]
            else:
                render(typed, "無効な番号です。1-21で入力してください。")
        elif key == 8:  # Backspace (may be 8 on some systems)
            typed = typed[:-1]
            render(typed)
        else:
            render(typed, "数字キー(1-21)とEnterを使用してください。")

def main():
    """
    機能:
    1. 起動時にモデルを選択可能
    2. 実行中にモデルを変更可能（Mキー）
    3. 画面上に使用中のモデルを表示
    4. 信頼度閾値を0.5に設定（誤検出を減らす）
    5. IOU閾値を調整（重複検出を削減）
    """
    
    # === モデル選択 ===
    # 
    # 【標準YOLOv8モデル】80クラスの物体を検出
    # yolov8n.pt - Nano版（軽量、低精度）
    # yolov8s.pt - Small版（バランス型）← 推奨
    # yolov8m.pt - Medium版（高精度）
    # yolov8l.pt - Large版（非常に高精度）
    # yolov8x.pt - Extra Large版（最高精度）
    #
    # 【YOLO-Worldモデル】無制限のクラスを検出可能（オープンボキャブラリー）
    # yolov8s-world.pt - Small版（軽量）
    # yolov8m-world.pt - Medium版（推奨）
    # yolov8l-world.pt - Large版（高精度）
    # yolov8x-world.pt - Extra Large版（最高精度）
    # ※使用方法: model.set_classes(["person", "car", "phone", ...]) で検出対象を設定
    #
    # 【セグメンテーション版】80クラス + 物体の輪郭を検出
    # yolov8s-seg.pt - Small版
    # yolov8m-seg.pt - Medium版（推奨）
    # yolov8l-seg.pt - Large版
    # yolov8x-seg.pt - Extra Large版
    #
    # 【姿勢推定版】人の骨格・関節を検出
    # yolov8s-pose.pt - Small版
    # yolov8m-pose.pt - Medium版（推奨）
    # yolov8l-pose.pt - Large版
    # yolov8x-pose.pt - Extra Large版
    #
    # 【YOLOv9】80クラス、より高い検出精度
    # yolov9c.pt - Compact版
    # yolov9e.pt - Extended版（高精度）
    #
    # 【YOLOv10】80クラス、最新バージョン
    # yolov10n.pt - Nano版
    # yolov10s.pt - Small版
    # yolov10m.pt - Medium版
    # yolov10l.pt - Large版
    # yolov10x.pt - Extra Large版
    #
    # 【RT-DETR】80クラス、より正確な境界ボックス
    # rtdetr-l.pt - Large版
    # rtdetr-x.pt - Extra Large版
    
    # デバイス検出
    device = detect_best_device()
    device_name = str(device)
    is_gpu = (device == 'dml' or device == 'cuda')
    
    print(f"検出されたデバイス: {device}")
    if device == 'dml':
        print("✓ DirectML（Intel/AMD GPU）を使用 - 高速化モード有効")
    elif device == 'cuda':
        print("✓ CUDA（NVIDIA GPU）を使用 - 高速化モード有効")
    else:
        print("⚠ CPU モード - GPU が検出されませんでした")
    
    print("モデルをロード中...")
    model_name = select_model_at_startup()

    # PyTorch版を直接使用（ONNXは使わない）
    model = YOLO(model_name)
    model.fuse()

    # 検出パラメータ（調整可能）
    conf_threshold = 0.5  # 信頼度閾値（0.0-1.0）
    iou_threshold = 0.45  # IOU閾値（0.0-1.0）
    imgsz = 640  # 推論画像サイズ（小さいほど高速: 320, 480, 640）

    # モデルウォームアップ
    warmup_model(model, imgsz=imgsz)

    # Webカメラをキャプチャ（マルチスレッド版）
    cap = ThreadedCamera(0)

    if not cap.isOpened():
        print("エラー: Webカメラを開くことができません")
        return

    # カメラスレッドを起動
    cap.start()
    import time as time_module
    time_module.sleep(0.5)  # カメラの初期化待機

    print("=" * 60)
    print("YOLO - リアルタイム物体検出（高速化版）")
    print("=" * 60)
    print(f"使用モデル: {model_name}")
    print(f"デバイス: {device}")
    print(f"信頼度閾値: {conf_threshold:.2f}")
    print(f"IOU閾値: {iou_threshold:.2f}")
    print(f"画像サイズ: {imgsz}px")
    print("\nキーボード操作:")
    print("  q: 終了")
    print("  m: モデルを変更（画面で選択）")
    print("  c: 信頼度を上げる（+0.05）")
    print("  x: 信頼度を下げる（-0.05）")
    print("  i: IOU閾値を上げる（+0.05）")
    print("  u: IOU閾値を下げる（-0.05）")
    print("  s: 画像サイズ切替（320/480/640）")
    print("  f: フレームスキップ切替（なし/1おき/2おき）")
    print("=" * 60)

    # 検出したいクラスを設定（YOLO-Worldの場合）
    if 'world' in model_name:
        model.set_classes([
             "car", "dog", "cat", "phone", "laptop", 
            "cup", "bottle", "chair", "book", "pen", "clock", 
            "door", "mirror", "remote", "pillow"
        ])
    
    prev_time = 0
    frame_skip = 0  # フレームスキップカウンター
    skip_frames = 0  # 0=スキップなし、1=1フレームおき
    
    # --- ラベル履歴バッファを用意 ---
    from collections import deque, Counter, defaultdict
    LABEL_HISTORY_LEN = 100  # 過去Nフレーム
    # オブジェクトごとにID（中心座標で近いものを同一とみなす）で履歴を管理
    object_label_history = defaultdict(lambda: deque(maxlen=LABEL_HISTORY_LEN))
    def get_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    def find_nearest_object_id(center, prev_centers, threshold=50):
        # 直近フレームの中心座標リストと比較し、近いものがあればそのIDを返す
        for obj_id, prev_center in prev_centers.items():
            dist = ((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) ** 0.5
            if dist < threshold:
                return obj_id
        return None
    next_object_id = 0
    prev_object_centers = {}

    while True:
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        # フレームスキップ処理
        frame_skip += 1
        if skip_frames > 0 and frame_skip % (skip_frames + 1) != 0:
            # 前回の検出結果を再利用
            if 'annotated_frame' in locals():
                cv2.imshow('YOLO', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        # YOLOで物体検出を実行
        results = model(frame, conf=conf_threshold, iou=iou_threshold, 
                       imgsz=imgsz, verbose=False)

        # --- 検出ラベルを各オブジェクトごとに履歴に追加 ---
        boxes = results[0].boxes
        curr_object_centers = {}
        object_ids_in_frame = []
        if hasattr(boxes, 'cls'):
            clses = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            labels = [results[0].names[int(cls)] for cls in clses]
            for i, (box, label) in enumerate(zip(xyxy, labels)):
                center = get_center(box)
                obj_id = find_nearest_object_id(center, prev_object_centers)
                if obj_id is None:
                    obj_id = next_object_id
                    next_object_id += 1
                curr_object_centers[obj_id] = center
                object_label_history[obj_id].append(label)
                object_ids_in_frame.append((obj_id, box))
        prev_object_centers = curr_object_centers



        # --- モデル種別で描画方法を分岐 ---
        if 'seg' in model_name:
            # セグメンテーションモデルはplot()のマスク画像＋安定化ラベルのみ（バウンディングボックスや元ラベルは表示しない）
            annotated_frame = results[0].plot()
            if hasattr(boxes, 'xyxy') and hasattr(boxes, 'cls'):
                for obj_id, box in object_ids_in_frame:
                    x1, y1, x2, y2 = map(int, box)
                    label_hist = object_label_history[obj_id]
                    if label_hist:
                        most_common_label, count = Counter(label_hist).most_common(1)[0]
                        stable_label_text = f"{most_common_label} ({count}/{len(label_hist)})"
                    else:
                        stable_label_text = "None"
                    # バウンディングボックスや元ラベルは描画しない
                    cv2.putText(
                        annotated_frame,
                        stable_label_text,
                        (x1 + 6, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 128, 255),
                        2
                    )
        else:
            # それ以外は自前でバウンディングボックス＋ラベル
            annotated_frame = frame.copy()
            for obj_id, box in object_ids_in_frame:
                x1, y1, x2, y2 = map(int, box)
                # バウンディングボックス描画
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # オブジェクトごとの安定化ラベル
                label_hist = object_label_history[obj_id]
                if label_hist:
                    most_common_label, count = Counter(label_hist).most_common(1)[0]
                    stable_label_text = f"{most_common_label} ({count}/{len(label_hist)})"
                else:
                    stable_label_text = "None"
                cv2.putText(
                    annotated_frame,
                    stable_label_text,
                    (x1 + 6, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 128, 255),
                    2
                )

        # 情報を表示（セグメンテーションモデルでは非表示）
        if 'seg' not in model_name:
            cv2.putText(annotated_frame, f"Model: {model_name} | Device: {device}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Conf: {conf_threshold:.2f} | IOU: {iou_threshold:.2f} | ImgSize: {imgsz}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 検出数を表示
            detections = results[0].boxes
            cv2.putText(annotated_frame, f"Detected: {len(detections)} objects", 
                   (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # フレームを表示
        cv2.imshow('YOLO', annotated_frame)

        # キー入力処理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            # モデル変更（OpenCVウィンドウで選択）
            selection = select_model_interactive()
            if selection is None:
                print("モデル変更をキャンセルしました。")
            else:
                model_name = selection
                print(f"モデルをロード中: {model_name}")
                
                # 新しいモデルをロード
                model = YOLO(model_name)
                model.fuse()
                
                # YOLO-Worldの場合はクラスを設定
                if 'world' in model_name:
                    model.set_classes([
                        "person", "car", "dog", "cat", "phone", "laptop",
                        "cup", "bottle", "chair", "book", "pen", "clock",
                        "door", "mirror", "remote", "pillow"
                    ])
                
                print(f"\n✓ モデルを変更しました: {model_name}")
                prev_time = 0
        elif key == ord('c'):
            conf_threshold = min(1.0, conf_threshold + 0.05)
            print(f"信頼度を上げました: {conf_threshold:.2f}")
        elif key == ord('x'):
            conf_threshold = max(0.0, conf_threshold - 0.05)
            print(f"信頼度を下げました: {conf_threshold:.2f}")
        elif key == ord('i'):
            iou_threshold = min(1.0, iou_threshold + 0.05)
            print(f"IOU閾値を上げました: {iou_threshold:.2f}")
        elif key == ord('u'):
            iou_threshold = max(0.0, iou_threshold - 0.05)
            print(f"IOU閾値を下げました: {iou_threshold:.2f}")
        elif key == ord('s'):
            # 画像サイズを切り替え（320→480→640→320...）
            if imgsz == 640:
                imgsz = 320
            elif imgsz == 320:
                imgsz = 480
            else:
                imgsz = 640
            print(f"画像サイズを変更しました: {imgsz}px（小さいほど高速）")
        elif key == ord('f'):
            # フレームスキップを切り替え（0→1→2→0...）
            if skip_frames == 0:
                skip_frames = 1
            elif skip_frames == 1:
                skip_frames = 2
            else:
                skip_frames = 0
            skip_name = "なし" if skip_frames == 0 else f"{skip_frames}フレームおき"
            print(f"フレームスキップを変更しました: {skip_name}（高速化）")
    
    # リソースを解放
    cap.stop()
    cv2.destroyAllWindows()
    print("プログラムを終了しました")

if __name__ == "__main__":
    main()
