import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os
from datetime import datetime
#test
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

# randoruto-number-subフォルダーの作成
save_folder = "randoruto-number-sub"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"フォルダーを作成しました: {save_folder}")

# 実行セッション用のフォルダーを作成（一回の実行につき一個）
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_folder = os.path.join(save_folder, f"session_{session_timestamp}")
os.makedirs(session_folder)
print(f"セッションフォルダーを作成しました: {session_folder}")

# YOLOv8nモデルのパス - target.ptを使用
model = YOLO('target.pt')  # target.ptファイルを使用

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

cv2.namedWindow('Target Detection Viewer', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Target Detection Viewer', window_width, window_height)
cv2.setWindowProperty('Target Detection Viewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

def save_detection_images(frame, detections_with_confidence, frame_count, leftup_info, leftdown_info):
    """一時停止時に検出画像を保存する関数"""
    global save_counter, session_folder
    
    print(f"デバッグ: save_detection_images関数が呼び出されました")
    print(f"デバッグ: frame_count = {frame_count}")
    print(f"デバッグ: session_folder = {session_folder}")
    
    if not detections_with_confidence:
        print("デバッグ: 検出物体がないため関数を終了します")
        return
    
    # セッションフォルダーが存在することを確認
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
        print(f"セッションフォルダーを作成しました: {session_folder}")
    
    # 四分割サイズを計算
    half_w = max(1, window_width // 2)
    half_h = max(1, window_height // 2)
    
    # 1. 全ての検出物体にバウンディングボックスを付けた画像を作成
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
    filename = f"target_detection_frame{frame_count}_{detection_count}objects.jpg"
    filepath = os.path.join(session_folder, filename)
    
    # 画像を保存
    print(f"デバッグ: 画像保存を試行します - {filepath}")
    success = cv2.imwrite(filepath, annotated_img)
    if success:
        print(f"画像を保存しました: {filepath} (検出物体数: {detection_count})")
        print(f"セッションフォルダー: {session_folder}")
        save_counter += 1
    else:
        print(f"エラー: 画像の保存に失敗しました - {filepath}")
    
    # 2. 右上の画像を個別保存（元の左上、四分割サイズに拡大、バウンディングボックス付き）
    if leftup_info:
        leftup_crop = create_cropped_image_with_bbox(frame, leftup_info, half_w, half_h)
        if leftup_crop is not None:
            leftup_filename = f"target_detection_frame{frame_count}_rightup_{leftup_info['label']}_{leftup_info['confidence']:.3f}.jpg"
            leftup_filepath = os.path.join(session_folder, leftup_filename)
            cv2.imwrite(leftup_filepath, leftup_crop)
            print(f"右上画像を保存しました: {leftup_filename}")
            save_counter += 1
    
    # 3. 右下の画像を個別保存（元の左下、四分割サイズに拡大、バウンディングボックス付き）
    if leftdown_info:
        leftdown_crop = create_cropped_image_with_bbox(frame, leftdown_info, half_w, half_h)
        if leftdown_crop is not None:
            leftdown_filename = f"target_detection_frame{frame_count}_rightdown_{leftdown_info['label']}_{leftdown_info['confidence']:.3f}.jpg"
            leftdown_filepath = os.path.join(session_folder, leftdown_filename)
            cv2.imwrite(leftdown_filepath, leftdown_crop)
            print(f"右下画像を保存しました: {leftdown_filename}")
            save_counter += 1

def create_cropped_image_with_bbox(frame, detection_info, target_width, target_height):
    """検出情報から切り抜き画像とバウンディングボックスを生成（指定サイズにリサイズ）"""
    if not detection_info:
        return None
    
    x1, y1, x2, y2 = detection_info['box']
    label = detection_info['label']
    confidence = detection_info['confidence']
    
    # 元画像から切り抜き範囲を拡張（バウンディングボックスを含める余白を追加）
    margin = 20  # 余白のピクセル数
    crop_x1 = max(0, x1 - margin)
    crop_y1 = max(0, y1 - margin)
    crop_x2 = min(frame.shape[1], x2 + margin)
    crop_y2 = min(frame.shape[0], y2 + margin)
    
    # 切り抜き画像を作成
    cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    
    if cropped_img.size == 0:
        return None
    
    # 四分割サイズにリサイズ
    resized_img = cv2.resize(cropped_img, (target_width, target_height))
    
    # リサイズ後の画像でのバウンディングボックス座標を計算
    scale_x = target_width / (crop_x2 - crop_x1)
    scale_y = target_height / (crop_y2 - crop_y1)
    
    # リサイズ後の相対座標
    relative_x1 = int((x1 - crop_x1) * scale_x)
    relative_y1 = int((y1 - crop_y1) * scale_y)
    relative_x2 = int((x2 - crop_x1) * scale_x)
    relative_y2 = int((y2 - crop_y1) * scale_y)
    
    # バウンディングボックスを描画
    cv2.rectangle(resized_img, (relative_x1, relative_y1), (relative_x2, relative_y2), (0, 255, 0), 2)
    
    # ラベルと信頼度のテキストを追加（フォントサイズを調整）
    text = f"{label}: {confidence:.3f}"
    font_scale = min(target_width, target_height) / 600  # サイズに応じてフォントサイズを調整
    cv2.putText(resized_img, text, (relative_x1, relative_y1-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    
    return resized_img

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
        status_text = "PAUSED" if leftup_paused else "TARGET VIEW"
        cv2.putText(panel_lu, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if not leftup_paused else (0, 0, 255), 2)
        if leftup_info and 'confidence' in leftup_info:
            conf_text = f"{leftup_info.get('label','')}: {leftup_info['confidence']:.3f}"
            cv2.putText(panel_lu, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_lu = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_lu, 'No Target Image', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)

    # 右上：検出中の画像（通常表示）
    panel_ru = cv2.resize(annotated_frame, (half_w, half_h))

    # 左下：二つ検出時のみ更新。無い場合は前回の画像を継続表示。
    if leftdown_image is not None:
        panel_ld = cv2.resize(leftdown_image, (half_w, half_h))
        cv2.putText(panel_ld, "SECONDARY TARGET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if leftdown_info and 'confidence' in leftdown_info:
            conf_text = f"{leftdown_info.get('label','')}: {leftdown_info['confidence']:.3f}"
            cv2.putText(panel_ld, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_ld = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_ld, 'No Secondary Target', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    # 右下：認識情報
    panel_rd = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    info_lines = []
    if detections_with_confidence:
        info_lines.append(f'Target Detections: {len(detections_with_confidence)}')
        if len(detections_with_confidence) >= 1:
            info_lines.append(f'Primary Target: {detections_with_confidence[0]["label"]} {detections_with_confidence[0]["confidence"]:.3f}')
        if len(detections_with_confidence) >= 2:
            if leftup_info:
                same_label = [d for d in detections_with_confidence if d['label'] == leftup_info['label']]
                if len(same_label) >= 2:
                    sec = same_label[1]
                else:
                    sec = detections_with_confidence[1]
                info_lines.append(f'Secondary Target: {sec["label"]} {sec["confidence"]:.3f}')
    else:
        info_lines.append('No Target Detection')
    info_lines.append(f'Frame: {leftup_frame_count}')
    info_lines.append(f'Status: {"Paused" if leftup_paused else "Active"}')
    for i, line in enumerate(info_lines):
        cv2.putText(panel_rd, line, (20, 60 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    # 常に四分割で表示（左右を入れ替え）
    top = np.hstack([panel_ru, panel_lu])
    bottom = np.hstack([panel_rd, panel_ld])
    combined = np.vstack([top, bottom])
    
    cv2.imshow('Target Detection Viewer', combined)

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
            print("Target detection resumed")
        else:
            # 再生中の場合：一時停止
            leftup_paused = True
            print("Target detection paused")
            # 一時停止時に検出画像を保存
            print(f"デバッグ: 検出物体数 = {len(detections_with_confidence)}")
            if detections_with_confidence:
                print("デバッグ: 保存処理を開始します")
                save_detection_images(frame, detections_with_confidence, leftup_frame_count, leftup_info, leftdown_info)
                print(f"Saved target detection images during pause (Total: {save_counter} files)")
            else:
                print("デバッグ: 検出物体がないため保存をスキップしました")

cap.release()
cv2.destroyAllWindows()
print("ターゲット検出を終了しました。")
print(f"保存された画像の総数: {save_counter}枚")
print(f"保存先フォルダー: {session_folder}")


