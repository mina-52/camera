import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os
from datetime import datetime
#kome
# ----------------------------------------------------
# 1. 動画ファイルの読み込み設定
# ----------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='動画ファイルを使用した物体検出')
    parser.add_argument('--video', type=str, default=r"C:\DJI_0008.MOV",
                       help='動画ファイルのパス（デフォルト: test.mov）')
    return parser.parse_args()

args = parse_arguments()

# randorutoフォルダーの作成
save_folder = "randoruto"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"フォルダーを作成しました: {save_folder}")

# 実行セッション用のフォルダーを作成（一回の実行につき一個）
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_folder = os.path.join(save_folder, f"session_{session_timestamp}")
os.makedirs(session_folder)
print(f"セッションフォルダーを作成しました: {session_folder}")

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
model = YOLO('weights.pt')  # ここをあなたのモデルのパスに置き換えてください

print("リアルタイム検出を開始します。'q'キーで終了します。")

# スクリーンサイズ取得（起動時に一度だけ）
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

# --- 四分割用の履歴リスト ---
detected_history = []  # 検出物体の切り抜き履歴（最新が先頭）
HISTORY_SIZE = 1  # 今回は最新のみで十分

# --- エンターキーで切り替え用 ---
leftup_image = None
leftup_info = None
leftup_paused = False  # 一時停止フラグ
leftup_frame_count = 0  # フレームカウンター
last_frame_time = time.time()  # 最後のフレーム更新時間

# --- 左下用の変数（左上と同期） ---
leftdown_image = None
leftdown_info = None

# --- 保存用のカウンター ---
save_counter = 0

def save_detection_images(frame, detections_with_confidence, frame_count):
    """一時停止時に検出画像を保存する関数"""
    global save_counter
    
    if not detections_with_confidence:
        return
    
    # 全ての検出物体にバウンディングボックスを付けた画像を作成
    annotated_img = frame.copy()
    
    # 各検出物体に対してバウンディングボックスとラベルを追加
    for i, detection in enumerate(detections_with_confidence):
        x1, y1, x2, y2 = detection['box']
        label = detection['label']
        confidence = detection['confidence']
        
        # バウンディングボックスを描画
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # ラベルと信頼度のテキストを追加
        text = f"{label}: {confidence:.3f}"
        cv2.putText(annotated_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # ファイル名を生成（検出数も含める）
    detection_count = len(detections_with_confidence)
    filename = f"detection_frame{frame_count}_{detection_count}objects.jpg"
    filepath = os.path.join(session_folder, filename)
    
    # 画像を保存
    cv2.imwrite(filepath, annotated_img)
    print(f"画像を保存しました: {session_folder}/{filename} (検出物体数: {detection_count})")
    save_counter += 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを読み込めませんでした。")
        break

    now = time.time()

    # 推論
    results = model(frame)
    annotated_frame = results[0].plot()
    detection_count = 0

    # 信頼度・ラベル付きで検出結果を収集
    detections_with_confidence = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detection_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])  # 信頼度
            cls_id = int(box.cls[0]) if hasattr(box, 'cls') else -1
            label = model.names[cls_id] if (cls_id in getattr(model, 'names', {})) else str(cls_id)
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0:
                detections_with_confidence.append({
                    'crop': crop,
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'cls_id': cls_id,
                    'label': label
                })

    # 信頼度でソート（高い順）
    detections_with_confidence.sort(key=lambda x: x['confidence'], reverse=True)

    # 左上用：従来通り、もっとも高い信頼度を選択
    if detections_with_confidence:
        detected_history = [detections_with_confidence[0]['crop']]
        detected_info = {
            'box': detections_with_confidence[0]['box'],
            'confidence': detections_with_confidence[0]['confidence'],
            'cls_id': detections_with_confidence[0]['cls_id'],
            'label': detections_with_confidence[0]['label']
        }
    else:
        detected_info = None

    # 左上/左下のフレーム更新（0.5秒ごと、一時停止中でない場合）
    if not leftup_paused and (now - last_frame_time > 0.5):
        if detected_history:
            leftup_image = detected_history[0].copy()
            leftup_info = detected_info
            leftup_frame_count += 1
            last_frame_time = now

        # 左下も左上の更新タイミングに同期
        # 二重検出があるときのみ更新。無いときは前回の表示を維持。
        if len(detections_with_confidence) >= 2:
            candidate = None
            if leftup_info and 'label' in leftup_info:
                same_label = [d for d in detections_with_confidence if d['label'] == leftup_info['label']]
                if len(same_label) >= 2:
                    candidate = same_label[1]
            if candidate is None:
                candidate = detections_with_confidence[1]
            leftdown_image = candidate['crop'].copy()
            leftdown_info = {
                'box': candidate['box'],
                'confidence': candidate['confidence'],
                'cls_id': candidate['cls_id'],
                'label': candidate['label']
            }
        # else: 何もしない（継続表示）

    # 四分割パネル作成
    half_w = max(1, window_width // 2)
    half_h = max(1, window_height // 2)

    # 左上：従来通り（0.5秒ごと更新・一時停止可能）
    if leftup_image is not None:
        panel_lu = cv2.resize(leftup_image, (half_w, half_h))
        status_text = "PAUSED" if leftup_paused else "PLAYING"
        cv2.putText(panel_lu, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if not leftup_paused else (0, 0, 255), 2)
        if leftup_info and 'confidence' in leftup_info:
            conf_text = f"{leftup_info.get('label','')}: {leftup_info['confidence']:.3f}"
            cv2.putText(panel_lu, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_lu = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_lu, 'No Image', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)

    # 右上：検出中の画像（通常表示）
    panel_ru = cv2.resize(annotated_frame, (half_w, half_h))

    # 左下：二つ検出時のみ更新。無い場合は前回の画像を継続表示。
    if leftdown_image is not None:
        panel_ld = cv2.resize(leftdown_image, (half_w, half_h))
        if leftdown_info and 'confidence' in leftdown_info:
            conf_text = f"{leftdown_info.get('label','')}: {leftdown_info['confidence']:.3f}"
            cv2.putText(panel_ld, conf_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_ld = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_ld, 'No Image', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)

    # 右下：認識情報
    panel_rd = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    info_lines = []
    if detections_with_confidence:
        info_lines.append(f'検出数: {len(detections_with_confidence)}')
        if len(detections_with_confidence) >= 1:
            info_lines.append(f'左上(1): {detections_with_confidence[0]["label"]} {detections_with_confidence[0]["confidence"]:.3f}')
        if len(detections_with_confidence) >= 2:
            if leftup_info:
                same_label = [d for d in detections_with_confidence if d['label'] == leftup_info['label']]
                if len(same_label) >= 2:
                    sec = same_label[1]
                else:
                    sec = detections_with_confidence[1]
                info_lines.append(f'左下(2): {sec["label"]} {sec["confidence"]:.3f}')
    else:
        info_lines.append('No Detection')
    info_lines.append(f'フレーム: {leftup_frame_count}')
    info_lines.append(f'状態: {"一時停止" if leftup_paused else "再生中"}')
    for i, line in enumerate(info_lines):
        cv2.putText(panel_rd, line, (20, 60 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # 常に四分割で表示
    top = np.hstack([panel_lu, panel_ru])
    bottom = np.hstack([panel_ld, panel_rd])
    combined = np.vstack([top, bottom])

    cv2.imshow('Detection Viewer', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 13:  # エンターキー（ASCII: 13）
        if leftup_paused:
            # 一時停止中の場合：現在のフレームまで飛ばす（左上・左下とも同期）
            if detected_history:
                leftup_image = detected_history[0].copy()
                leftup_info = detected_info
                leftup_frame_count += 1
                last_frame_time = now
            # 左下は、二重検出がある場合のみ更新。無い場合は前回の画像を保持。
            if len(detections_with_confidence) >= 2:
                candidate = None
                if leftup_info and 'label' in leftup_info:
                    same_label = [d for d in detections_with_confidence if d['label'] == leftup_info['label']]
                    if len(same_label) >= 2:
                        candidate = same_label[1]
                if candidate is None:
                    candidate = detections_with_confidence[1]
                leftdown_image = candidate['crop'].copy()
                leftdown_info = {
                    'box': candidate['box'],
                    'confidence': candidate['confidence'],
                    'cls_id': candidate['cls_id'],
                    'label': candidate['label']
                }
            leftup_paused = False
            print("再生を再開しました")
        else:
            # 再生中の場合：一時停止
            leftup_paused = True
            print("一時停止しました")
            # 一時停止時に検出画像を保存
            if detections_with_confidence:
                save_detection_images(frame, detections_with_confidence, leftup_frame_count)
                print(f"一時停止時の検出画像を保存しました（合計: {save_counter}枚）")

cap.release()
cv2.destroyAllWindows()
print("検出を終了しました。")
print(f"保存された画像の総数: {save_counter}枚")
print(f"保存先フォルダー: {session_folder}")


  