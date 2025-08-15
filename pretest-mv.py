import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse

# ----------------------------------------------------
# 1. 動画ファイルの読み込み設定
# ----------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='動画ファイルを使用した物体検出')
    parser.add_argument('--video', type=str, default='test.mov',
                       help='動画ファイルのパス（デフォルト: test.mov）')
    return parser.parse_args()

args = parse_arguments()

# 動画ファイルの設定
cap = cv2.VideoCapture(args.video)
print(f"動画ファイルを読み込み中: {args.video}")

if not cap.isOpened():
    print("エラー: 動画ファイルを開けませんでした。")
    print("ファイルパスが正しいか確認してください。")
    print("使用例:")
    print("  python pretest-mv.py --video video.mp4")
    exit()

# YOLOv8nモデルのパス
model = YOLO('weights.pt')  # モデルのパスに置き換え

print("リアルタイム検出を開始します。'q'キーで終了します。")

# スクリーンサイズ取得
try:
    import tkinter as tk
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
except Exception:
    screen_width = 1280
    screen_height = 720

window_width = screen_width
window_height = screen_height

cv2.namedWindow('Detection Viewer', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection Viewer', window_width, window_height)
cv2.setWindowProperty('Detection Viewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- 四分割用の履歴 ---
detected_history = []
HISTORY_SIZE = 1

# --- 左上用変数 ---
leftup_image = None
leftup_info = None
leftup_paused = False
leftup_frame_count = 0

# --- ランドルト環クラス ---
LANDOLT_CLASS_NAMES = ["randoruto", "randoruto2"]  # 2種類対応
landolt_detected = False
last_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを読み込めませんでした。")
        break

    now = time.time()

    # 推論
    results = model(frame)
    annotated_frame = results[0].plot()
    new_crops = []
    new_infos = []
    landolt_found = False

    for r in results:
        boxes = r.boxes
        names = r.names if hasattr(r, "names") else {}
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0]) if hasattr(box, "cls") else None
            class_name = names.get(cls_id, str(cls_id)) if cls_id is not None else ""
            if class_name in LANDOLT_CLASS_NAMES:
                landolt_found = True
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0:
                new_crops.append(crop)
                new_infos.append({'box': (x1, y1, x2, y2), 'class': class_name})

    # 検出フラグ更新
    landolt_detected = landolt_found

    # 最新の履歴保存
    if new_crops:
        detected_history = [new_crops[0]]
        detected_info = new_infos[0]
    else:
        detected_info = None

    # --- 左上パネル更新ロジック ---
    if landolt_detected and leftup_image is None and new_crops:
        # 初回検出時に即反映
        leftup_image = new_crops[0].copy()
        leftup_info = detected_info
        leftup_frame_count = 1
        last_frame_time = now

    if landolt_detected and not leftup_paused and new_crops:
        # 1秒ごとに更新
        if now - last_frame_time >= 1.0:
            leftup_image = new_crops[0].copy()
            leftup_info = detected_info
            leftup_frame_count += 1
            last_frame_time = now

    # 表示切り替え
    if not landolt_detected:
        # 検出なし → 一画面表示
        cv2.imshow('Detection Viewer', cv2.resize(annotated_frame, (window_width, window_height)))
        leftup_image = None  # 検出が途切れたらリセット
    else:
        # 四分割パネル
        half_w = max(1, window_width // 2)
        half_h = max(1, window_height // 2)

        # 右上
        panel_ru = cv2.resize(annotated_frame, (half_w, half_h))

        # 左下
        if detected_history:
            panel_ld = cv2.resize(detected_history[0], (half_w, half_h))
        else:
            panel_ld = np.zeros((half_h, half_w, 3), dtype=np.uint8)
            cv2.putText(panel_ld, 'No Image', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)

        # 左上
        if leftup_image is not None:
            panel_lu = cv2.resize(leftup_image, (half_w, half_h))
            status_text = "PAUSED" if leftup_paused else "PLAYING"
            cv2.putText(panel_lu, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if not leftup_paused else (0, 0, 255), 2)
        else:
            panel_lu = np.zeros((half_h, half_w, 3), dtype=np.uint8)
            cv2.putText(panel_lu, 'No Image', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)

        # 右下
        panel_rd = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        info_lines = []
        if leftup_info:
            info_lines.append(f'座標: {leftup_info["box"]}')
            info_lines.append(f'クラス: {leftup_info["class"]}')
        else:
            info_lines.append('No Detection')
        info_lines.append(f'フレーム: {leftup_frame_count}')
        info_lines.append(f'状態: {"一時停止" if leftup_paused else "再生中"}')
        for i, line in enumerate(info_lines):
            cv2.putText(panel_rd, line, (20, 60 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        combined = np.vstack([np.hstack([panel_lu, panel_ru]),
                              np.hstack([panel_ld, panel_rd])])
        cv2.imshow('Detection Viewer', combined)

    # キー操作
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 13:  # Enterキー
        if leftup_paused:
            leftup_paused = False
            print("再生を再開しました")
        else:
            leftup_paused = True
            print("一時停止しました")

cap.release()
cv2.destroyAllWindows()
print("検出を終了しました。")
