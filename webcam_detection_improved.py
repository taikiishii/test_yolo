import cv2
import time
import numpy as np
import os
import platform
from ultralytics import YOLO

# OpenCVのGUIバックエンドを設定（ディスプレイがない環境への対応）
DISPLAY_AVAILABLE = False
try:
    # ディスプレイ利用可能か確認
    if os.environ.get('DISPLAY') is None and platform.system() == 'Linux':
        print("⚠ DISPLAYが設定されていません - ヘッドレスモードで動作します")
        DISPLAY_AVAILABLE = False
    else:
        DISPLAY_AVAILABLE = True
except:
    DISPLAY_AVAILABLE = False

# プラットフォーム検出
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
IS_JETSON = os.path.exists('/etc/nv_tegra_release') or os.path.exists('/sys/module/tegra_fuse')

# Jetson環境検出
if IS_JETSON:
    print("🚀 Jetson環境を検出しました")
    # JetsonでのCUDA最適化設定
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    # TensorRT最適化を有効化
    os.environ['TENSORRT_VERBOSE'] = '0'

# Windows環境でMSMFカメラバックエンドを使用（DirectShow より高速）
if IS_WINDOWS:
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '1'

# PyTorch CPU最適化（GPUが使えない場合のフォールバック）
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count() or 4)

# PyTorchのスレッド数を設定
try:
    import torch
    # CUDA使用可能かチェック
    if torch.cuda.is_available():
        print(f"✓ CUDA利用可能: {torch.cuda.get_device_name(0)}")
        print(f"  CUDAバージョン: {torch.version.cuda}")
        print(f"  GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # Jetson最適化
        if IS_JETSON:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    else:
        print("⚠ CUDAが使用できません - CPUモードで動作します")
        torch.set_num_threads(os.cpu_count() or 4)
        torch.set_num_interop_threads(max(1, (os.cpu_count() or 4) // 2))
except ImportError:
    print("⚠ PyTorchがインストールされていません")
    pass

print(f"  プラットフォーム: {platform.system()} {platform.release()}")
print(f"  CPU: {os.cpu_count()} cores")

# カメラ読み込みを別スレッドで行うクラス（遅延削減）
import threading
from queue import Queue

class ThreadedCamera:
    def __init__(self, src=0):
        # プラットフォームに応じたバックエンドを選択
        if IS_JETSON:
            # JetsonではGStreamerを優先（最高速）
            try:
                gst_str = (
                    f"v4l2src device=/dev/video{src} ! "
                    "video/x-raw, width=1280, height=720, framerate=30/1 ! "
                    "videoconvert ! video/x-raw, format=BGR ! "
                    "appsink drop=1"
                )
                self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
                if not self.cap.isOpened():
                    raise Exception("GStreamerバックエンド失敗")
                print("  カメラバックエンド: GStreamer（Jetson最適化）")
            except:
                # フォールバック: V4L2
                self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                print("  カメラバックエンド: V4L2")
        elif IS_WINDOWS:
            self.cap = cv2.VideoCapture(src, cv2.CAP_MSMF)
            print("  カメラバックエンド: MSMF")
        elif IS_LINUX:
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            print("  カメラバックエンド: V4L2")
        else:
            self.cap = cv2.VideoCapture(src)  # macOSなど
            print("  カメラバックエンド: デフォルト")
        
        # カメラ設定（GStreamer使用時はパイプライン内で設定済み）
        if not IS_JETSON or not self.cap.isOpened():
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

# デバイス検出関数
def detect_best_device():
    """利用可能な最適なデバイスを検出"""
    try:
        import torch
        if torch.cuda.is_available():
            # CUDA使用可能
            if IS_JETSON:
                print("  Jetson環境: CUDA GPU を使用")
                return 'cuda:0'
            else:
                return 'cuda'
    except ImportError:
        pass
    
    # DirectML検出（Windows）
    if IS_WINDOWS:
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device()
        except ImportError:
            pass
    
    # フォールバック: CPU
    if IS_JETSON:
        print("  Jetson環境: CUDAが利用できないためCPUを使用")
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
    print("  1: yolov8n.pt    - Nano版（軽量、低精度）← Jetson推奨")
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
    print(" 17: yolo11n.pt - Nano版 ← Jetson推奨")
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
    
    # Jetson環境では推奨モデルを表示
    if IS_JETSON:
        print("\n💡 Jetson環境では 17 (yolo11n.pt) または 1 (yolov8n.pt) を推奨します")
        print("   Enterキーで推奨モデル(yolo11n.pt)を自動選択できます")
    
    while True:
        choice = input("\nモデルを選択してください (1-21): ").strip()
        
        # Jetson環境でEnterキーのみの場合は推奨モデルを選択
        if choice == "" and IS_JETSON:
            print("→ 推奨モデル yolo11n.pt を選択しました")
            return 'yolo11n.pt'
        
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

    # GUI機能がない場合またはDISPLAY_AVAILABLEがFalseの場合は常にコンソール入力
    if not DISPLAY_AVAILABLE or not hasattr(cv2, 'namedWindow'):
        print("\nコンソールから選択してください:")
        print_model_menu()
        while True:
            choice = input("\nモデルを選択 (1-21, Enterでキャンセル): ").strip()
            if choice == "":
                return None
            if choice in model_map:
                return model_map[choice]
            print("❌ 無効な選択です。1-21の数字を入力してください。")

    try:
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
    except Exception as e:
        print(f"❌ ウィンドウ表示エラー: {e}")
        print("コンソールから選択してください:")
        print_model_menu()
        while True:
            choice = input("\nモデルを選択 (1-21, Enterでキャンセル): ").strip()
            if choice == "":
                return None
            if choice in model_map:
                return model_map[choice]
            print("❌ 無効な選択です。1-21の数字を入力してください。")

def main():
    """
    Jetson Orin Nano最適化版 YOLOリアルタイム物体検出
    - CUDA GPU加速対応
    - GStreamerカメラバックエンド（Jetson最適化）
    - Windows/Linux互換性確保
    """
    
    # デバイス検出
    device = detect_best_device()
    device_name = str(device)
    is_gpu = ('cuda' in str(device) or 'dml' in str(device))
    
    print(f"\n検出されたデバイス: {device}")
    if 'cuda' in str(device):
        print("✓ CUDA（NVIDIA GPU）を使用 - GPU高速化モード有効")
        if IS_JETSON:
            print("  Jetson最適化を適用")
    elif 'dml' in str(device):
        print("✓ DirectML（Intel/AMD GPU）を使用 - GPU高速化モード有効")
    else:
        print("⚠ CPU モード - GPU が検出されませんでした")
    
    print("\nモデルをロード中...")
    model_name = select_model_at_startup()
    
    # Jetson向けモデルサイズ警告
    if IS_JETSON:
        large_models = ['yolo11x.pt', 'yolo11l.pt', 'yolov8x.pt', 'yolov8l.pt', 'yolov9c.pt', 'rtdetr-l.pt']
        if model_name in large_models:
            print(f"⚠ 警告: {model_name} は大きなモデルです。")
            print("  Jetson Orin Nanoでは、GPUメモリ不足が発生する可能性があります。")
            print("  推奨モデル: yolo11n.pt, yolo11s.pt, yolov8n.pt, yolov8s.pt")
    
    # PyTorch版を直接使用（ONNXは使わない）
    model = YOLO(model_name)
    
    # デバイスにモデルを移動（GPUメモリ不足時はCPUにフォールバック）
    if is_gpu:
        try:
            model.to(device)
            print(f"  ✓ モデルを {device} にロードしました")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"  ⚠ GPUメモリ不足: {e}")
                print("  → CPUモードにフォールバック")
                device = 'cpu'
                is_gpu = False
                model.to('cpu')
            else:
                raise
    
    # モデル最適化
    try:
        model.fuse()
        print("  ✓ モデル最適化（fuse）完了")
    except:
        pass
    
    # 検出パラメータ（調整可能）
    conf_threshold = 0.5  # 信頼度閾値（0.0-1.0）
    iou_threshold = 0.45  # IOU閾値（0.0-1.0）
    # Jetsonの場合は小さめの画像サイズからスタート（メモリ効率化）
    imgsz = 320 if IS_JETSON else 640  # 推論画像サイズ（小さいほど高速: 320, 416, 480, 640）
    
    # モデルウォームアップ（エラーハンドリング付き）
    try:
        warmup_model(model, imgsz=imgsz)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower() or "nvml" in str(e).lower():
            print(f"  ⚠ GPUメモリ不足が検出されました")
            if is_gpu:
                print("  → CPUモードに切り替えます")
                device = 'cpu'
                is_gpu = False
                # モデルをCPUに移動
                model.to('cpu')
                # 再度ウォームアップ
                try:
                    warmup_model(model, imgsz=imgsz)
                except:
                    print("  ⚠ ウォームアップをスキップします")
        else:
            raise
    
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
    print("YOLO - リアルタイム物体検出（Jetson最適化版）")
    print("=" * 60)
    print(f"使用モデル: {model_name}")
    print(f"デバイス: {device}")
    print(f"信頼度閾値: {conf_threshold:.2f}")
    print(f"IOU閾値: {iou_threshold:.2f}")
    print(f"画像サイズ: {imgsz}px")
    print("\nキーボード操作:")
    print("  q: 終了")
    print("  r: プログラムを再起動（GPU状態をリセット）")
    print("  m: モデルを変更（画面で選択）")
    print("  c: 信頼度を上げる（+0.05）")
    print("  x: 信頼度を下げる（-0.05）")
    print("  i: IOU閾値を上げる（+0.05）")
    print("  u: IOU閾値を下げる（-0.05）")
    print("  s: 画像サイズ切替（320/416/480/640）")
    print("  f: フレームスキップ切替（なし/1おき/2おき）")
    if IS_JETSON:
        print("\n💡 ヒント: GPUメモリ不足後にGPUに戻すには 'r' で再起動してください")
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
    
    # ディスプレイ利用可能かチェック（opencv-python-headlessでは常にFalse）
    display_available = DISPLAY_AVAILABLE
    if display_available:
        try:
            # GUI機能が利用可能か確認
            if hasattr(cv2, 'namedWindow'):
                cv2.namedWindow('YOLO_TEST', cv2.WINDOW_NORMAL)
                cv2.destroyWindow('YOLO_TEST')
                display_available = True
                print("✓ ディスプレイが利用可能です")
            else:
                print("⚠ OpenCV headless版を使用中 - GUI表示なし")
                print("  検出結果をコンソールに出力します")
                display_available = False
        except Exception as e:
            print(f"⚠ ディスプレイを使用できません: {e}")
            print("  検出結果をコンソールに出力します")
            display_available = False
    else:
        print("⚠ ディスプレイなし環境 - GUI表示スキップ")
        print("  検出結果をコンソールに出力します")
    
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
            if display_available and 'annotated_frame' in locals():
                cv2.imshow('YOLO', annotated_frame)
            key = cv2.waitKey(1) & 0xFF if display_available else ord('z')
            if key == ord('q'):
                break
            continue
        
        # YOLOで物体検出を実行（デバイス指定、エラーハンドリング付き）
        try:
            results = model(frame, conf=conf_threshold, iou=iou_threshold, 
                           imgsz=imgsz, verbose=False, device=device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower() or "nvml" in str(e).lower():
                print(f"\n⚠ GPUメモリ不足エラー: {e}")
                if is_gpu:
                    print("→ CPUモードに切り替えます...")
                    device = 'cpu'
                    is_gpu = False
                    model.to('cpu')
                    # CUDAキャッシュをクリア
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except:
                        pass
                    print("✓ CPUモードに切り替えました")
                    # 再試行
                    results = model(frame, conf=conf_threshold, iou=iou_threshold, 
                                   imgsz=imgsz, verbose=False, device=device)
                else:
                    raise
            else:
                raise
        
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
            # セグメンテーションモデルはplot()のマスク画像＋安定化ラベルのみ
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
                    # 安定化ラベルを表示
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
            # それ以外は自前でバウンディングボックス＋安定化ラベル
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
        
        # 検出されたオブジェクトの名前を取得（コンソール出力用）
        detected_names = []
        if len(boxes) > 0 and hasattr(boxes, 'cls'):
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_names.append(class_name)
        
        # FPS情報をコンソールに出力（ディスプレイなし時）
        if not display_available:
            if detected_names:
                names_str = ", ".join(detected_names)
                print(f"\rFPS: {fps:.2f} | Detected: {len(boxes)} objects ({names_str})", end='', flush=True)
            else:
                print(f"\rFPS: {fps:.2f} | Detected: {len(boxes)} objects", end='', flush=True)
        
        # フレームを表示
        if display_available:
            try:
                cv2.imshow('YOLO', annotated_frame)
            except Exception as e:
                print(f"\n❌ 表示エラー: {e}")
                display_available = False
        
        # キー入力処理
        key = cv2.waitKey(1) & 0xFF if display_available else -1
        
        # Ctrl+C でも終了できるように
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('r'):
            # プログラムを再起動
            print("\n🔄 プログラムを再起動します...")
            cap.stop()
            if display_available:
                cv2.destroyAllWindows()
            
            # メモリをクリア
            try:
                del model
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
            
            print("✓ リソースを解放しました")
            import sys
            import os
            print(f"✓ Pythonを再実行: {sys.executable} {sys.argv[0]}")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        elif key == ord('m'):
            # モデル変更（OpenCVウィンドウで選択）
            selection = select_model_interactive()
            if selection is None:
                print("モデル変更をキャンセルしました。")
            else:
                model_name = selection
                print(f"\nモデルをロード中: {model_name}")
                
                # Jetson向けモデルサイズ警告
                if IS_JETSON:
                    large_models = ['yolo11x.pt', 'yolo11l.pt', 'yolov8x.pt', 'yolov8l.pt', 'yolov9c.pt', 'rtdetr-l.pt']
                    if model_name in large_models:
                        print(f"⚠ 警告: {model_name} は大きなモデルです。")
                        print("  Jetson Orin Nanoでは、GPUメモリ不足が発生する可能性があります。")
                
                # 古いモデルを削除してメモリを解放
                try:
                    del model
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    print("  ✓ 旧モデルのメモリを解放しました")
                except:
                    pass
                
                import time as time_module
                time_module.sleep(0.5)  # メモリ解放の待機
                
                # デバイスを再検出（GPUが利用可能であれば再度使用）
                original_device = detect_best_device()
                device = original_device
                is_gpu = ('cuda' in str(device) or 'dml' in str(device))
                print(f"  使用デバイス: {device}")
                
                # 新しいモデルをロード
                model = YOLO(model_name)
                
                # デバイスにモデルを移動（GPUメモリ不足時はCPUにフォールバック）
                if is_gpu:
                    try:
                        model.to(device)
                        print(f"  ✓ モデルを {device} にロードしました")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "cuda" in str(e).lower() or "cublas" in str(e).lower():
                            print(f"  ⚠ GPUメモリ不足またはGPUエラー")
                            print("  → CPUモードにフォールバック")
                            device = 'cpu'
                            is_gpu = False
                            model.to('cpu')
                        else:
                            raise
                
                # モデル最適化
                try:
                    model.fuse()
                except:
                    pass
                
                # YOLO-Worldの場合はクラスを設定
                if 'world' in model_name:
                    model.set_classes([
                        "car", "dog", "cat", "phone", "laptop",
                        "cup", "bottle", "chair", "book", "pen", "clock",
                        "door", "mirror", "remote", "pillow"
                    ])
                
                print(f"✓ モデルを変更しました: {model_name} (デバイス: {device})")
                
                # GPUに戻れなかった場合のメッセージ
                if not is_gpu and torch.cuda.is_available():
                    print("  ⚠ 注意: GPUメモリの状態が不安定なため、CPUモードで動作しています")
                    print("  → GPUを使用するには、プログラムを再起動してください")
                
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
            # 画像サイズを切り替え（320→416→480→640→320...）
            if imgsz == 640:
                imgsz = 320
            elif imgsz == 320:
                imgsz = 416
            elif imgsz == 416:
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
    if display_available:
        cv2.destroyAllWindows()
    print("\nプログラムを終了しました")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Ctrl+Cで終了しました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
