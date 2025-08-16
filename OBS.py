import cv2
import numpy as np
from ultralytics import YOLO
import time
#kome
# ----------------------------------------------------
# 1. OBS仮想カメラの読み込み設定
# ----------------------------------------------------
cap = cv2.VideoCapture(1)  # 仮想カメラのインデックス。環境に合わせて変更してください
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("エラー: OBS仮想カメラを開けませんでした。")
    print("OBSで「仮想カメラを開始」しているか、正しいカメラインデックスか確認してください。")
    exit()

# YOLOv8nモデルのパス
model = YOLO('weights.pt')  # ここをあなたのモデルのパスに置き換えてください
# 第二モデル（number.pt）の追加
number_model = YOLO('number.pt')  # 数字認識用モデル

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
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detection_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0:
                new_crops.append(crop)
                new_infos.append({'box': (x1, y1, x2, y2)})
    
    # 第二モデル（number.pt）での推論
    number_results = number_model(frame)
    number_detection_count = 0
    number_detections = []
    for r in number_results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            # 信頼度が90%以上の場合のみ処理
            if conf >= 0.9:
                number_detection_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                # クラス名を取得（利用可能な場合）
                try:
                    class_name = r.names[cls]
                except:
                    class_name = f"Class_{cls}"
                number_detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class': cls,
                    'class_name': class_name
                })
    
    # 最新の検出物体画像・情報を履歴に保存
    if new_crops:
        detected_history = [new_crops[0]]
        detected_info = new_infos[0]
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
    # 左下：フレームごとに最新
    if detected_history:
        panel_ld = cv2.resize(detected_history[0], (half_w, half_h))
    else:
        panel_ld = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_ld, 'No Image', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    # 左上：一秒ごと更新（一時停止可能）
    if leftup_image is not None:
        panel_lu = cv2.resize(leftup_image, (half_w, half_h))
        # 一時停止状態を表示
        status_text = "PAUSED" if leftup_paused else "PLAYING"
        cv2.putText(panel_lu, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if not leftup_paused else (0, 0, 255), 2)
    else:
        panel_lu = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_lu, 'No Image', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    # 右下：第二モデル（number.pt）の検出結果（バウンディングボックス付き）
    if number_detections:
        # 第二モデルの検出結果を描画した画像を作成
        number_annotated_frame = frame.copy()
        for det in number_detections:
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            class_name = det['class_name']
            
            # バウンディングボックスを描画
            cv2.rectangle(number_annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ラベルを描画
            label = f'{class_name} {conf:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(number_annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(number_annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        panel_rd = cv2.resize(number_annotated_frame, (half_w, half_h))
    else:
        # 検出がない場合は元のフレームを表示
        panel_rd = cv2.resize(frame, (half_w, half_h))
        cv2.putText(panel_rd, 'No Number Detection', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    
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
            leftup_paused = False
            print("再生を再開しました")
        else:
            # 再生中の場合：一時停止
            leftup_paused = True
            print("一時停止しました")

cap.release()
cv2.destroyAllWindows()
print("検出を終了しました。")


  