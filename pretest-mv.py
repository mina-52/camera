import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
#kome
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
    new_crops = []
    new_infos = []
    
    # 信頼度付きで検出結果を収集
    detections_with_confidence = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detection_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])  # 信頼度を取得
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0:
                detections_with_confidence.append({
                    'crop': crop,
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence
                })
    
    # 信頼度でソート（高い順）
    detections_with_confidence.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 最新の検出物体画像・情報を履歴に保存
    if detections_with_confidence:
        detected_history = [detections_with_confidence[0]['crop']]
        detected_info = {
            'box': detections_with_confidence[0]['box'],
            'confidence': detections_with_confidence[0]['confidence']
        }
    else:
        detected_info = None

    # 左上のフレーム更新（一秒ごと、一時停止中でない場合）
    if not leftup_paused and (now - last_frame_time > 1.0):
        if detected_history:
            leftup_image = detected_history[0].copy()
            leftup_info = detected_info
            leftup_frame_count += 1
            last_frame_time = now

    # 四分割パネル作成
    half_w = max(1, window_width // 2)
    half_h = max(1, window_height // 2)
    # 右上：検出中の画像
    panel_ru = cv2.resize(annotated_frame, (half_w, half_h))
    
    # 左下：信頼度の低い方（2番目）を表示、左上と同期
    if len(detections_with_confidence) >= 2:
        # 2つ以上の物体が検出された場合、信頼度の低い方を左下に表示
        if not leftup_paused and leftup_image is not None:
            # 左上と同期して更新
            leftdown_image = detections_with_confidence[1]['crop'].copy()
            leftdown_info = {
                'box': detections_with_confidence[1]['box'],
                'confidence': detections_with_confidence[1]['confidence']
            }
        panel_ld = cv2.resize(leftdown_image, (half_w, half_h)) if leftdown_image is not None else np.zeros((half_h, half_w, 3), dtype=np.uint8)
    elif detected_history:
        # 1つの物体のみの場合
        if not leftup_paused and leftup_image is not None:
            leftdown_image = detected_history[0].copy()
            leftdown_info = detected_info
        panel_ld = cv2.resize(leftdown_image, (half_w, half_h)) if leftdown_image is not None else np.zeros((half_h, half_w, 3), dtype=np.uint8)
    else:
        panel_ld = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_ld, 'No Image', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    
    # 左上：信頼度の高い方（1番目）を表示（一時停止可能）
    if leftup_image is not None:
        panel_lu = cv2.resize(leftup_image, (half_w, half_h))
        # 一時停止状態を表示
        status_text = "PAUSED" if leftup_paused else "PLAYING"
        cv2.putText(panel_lu, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if not leftup_paused else (0, 0, 255), 2)
        # 信頼度を表示
        if leftup_info and 'confidence' in leftup_info:
            conf_text = f"Conf: {leftup_info['confidence']:.3f}"
            cv2.putText(panel_lu, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_lu = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_lu, 'No Image', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    
    # 右下：認識情報
    panel_rd = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    info_lines = []
    if detections_with_confidence:
        info_lines.append(f'検出数: {len(detections_with_confidence)}')
        if len(detections_with_confidence) >= 1:
            info_lines.append(f'左上(1): {detections_with_confidence[0]["confidence"]:.3f}')
        if len(detections_with_confidence) >= 2:
            info_lines.append(f'左下(2): {detections_with_confidence[1]["confidence"]:.3f}')
    else:
        info_lines.append('No Detection')
    info_lines.append(f'フレーム: {leftup_frame_count}')
    info_lines.append(f'状態: {"一時停止" if leftup_paused else "再生中"}')
    for i, line in enumerate(info_lines):
        cv2.putText(panel_rd, line, (20, 60 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # 表示ロジックの変更
    if detection_count > 0:
        # 物体を認識しているときは四分割で表示
        # 上下連結
        top = np.hstack([panel_lu, panel_ru])
        bottom = np.hstack([panel_ld, panel_rd])
        combined = np.vstack([top, bottom])
    elif not leftup_paused:
        # 一時停止中ではないかつ物体を認識していないときは右上の画面だけを拡大して表示
        combined = cv2.resize(annotated_frame, (window_width, window_height))
    else:
        # 一時停止中で物体を認識していない場合は四分割表示を維持
        top = np.hstack([panel_lu, panel_ru])
        bottom = np.hstack([panel_ld, panel_rd])
        combined = np.vstack([top, bottom])
    
    cv2.imshow('Detection Viewer', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 13:  # エンターキー（ASCII: 13）
        if leftup_paused:
            # 一時停止中の場合：現在のフレームまで飛ばす
            if detected_history:
                leftup_image = detected_history[0].copy()
                leftup_info = detected_info
                leftup_frame_count += 1
                last_frame_time = now
                # 左下も同期して更新
                if len(detections_with_confidence) >= 2:
                    leftdown_image = detections_with_confidence[1]['crop'].copy()
                    leftdown_info = {
                        'box': detections_with_confidence[1]['box'],
                        'confidence': detections_with_confidence[1]['confidence']
                    }
                elif detected_history:
                    leftdown_image = detected_history[0].copy()
                    leftdown_info = detected_info
            leftup_paused = False
            print("再生を再開しました")
        else:
            # 再生中の場合：一時停止
            leftup_paused = True
            print("一時停止しました")

cap.release()
cv2.destroyAllWindows()
print("検出を終了しました。")


  