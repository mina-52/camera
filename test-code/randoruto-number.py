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

# randoruto-numberフォルダーの作成
save_folder = "randoruto-number"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"フォルダーを作成しました: {save_folder}")

# 実行セッション用のフォルダーを作成（一回の実行につき一個）
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_folder = os.path.join(save_folder, f"session_{session_timestamp}")
os.makedirs(session_folder)
print(f"セッションフォルダーを作成しました: {session_folder}")

# カウンター変数の初期化
save_counter = 0
pause_counter = 0
image_save_counter = 0

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

cv2.namedWindow('Target Detection Viewer', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Target Detection Viewer', window_width, window_height)
cv2.setWindowProperty('Target Detection Viewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- 四分割用の履歴リスト ---
detected_history = []  # 検出物体の切り抜き履歴（最新が先頭）
HISTORY_SIZE = 1  # 今回は最新のみで十分

# --- エンターキーで切り替え用 ---
leftup_image = None
leftup_info = None
leftup_paused = False  # 左上一時停止フラグ
leftup_frame_count = 0  # 左上フレームカウンター
leftup_last_frame_time = time.time()  # 左上最後のフレーム更新時間

# --- 左下用の変数（独立制御） ---
leftdown_image = None
leftdown_info = None
leftdown_paused = False  # 左下一時停止フラグ
leftdown_frame_count = 0  # 左下フレームカウンター
leftdown_last_frame_time = time.time()  # 左下最後のフレーム更新時間

# --- 保存用のカウンター ---
save_counter = 0

def save_detection_images(frame, detections_with_confidence, frame_count, leftup_info, leftdown_info, save_leftup=False, save_leftdown=False, leftup_image=None, leftdown_image=None):
    """一時停止時に検出画像を保存する関数（選択的保存）"""
    global save_counter, session_folder, image_save_counter
    
    print(f"デバッグ: save_detection_images関数が呼び出されました")
    print(f"デバッグ: frame_count = {frame_count}")
    print(f"デバッグ: session_folder = {session_folder}")
    
    # 検出物体がない場合でも保存を継続
    
    # セッションフォルダーが存在することを確認
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
        print(f"セッションフォルダーを作成しました: {session_folder}")
    
    # 四分割サイズを計算
    half_w = max(1, window_width // 2)
    half_h = max(1, window_height // 2)
    
    # 1. 全ての検出物体にバウンディングボックスを付けた画像を作成（常に保存）
    annotated_img = frame.copy()
    
    # 各検出物体に対してバウンディングボックスとラベルを追加
    for i, detection in enumerate(detections_with_confidence):
        x1, y1, x2, y2 = detection['box']
        label = detection['label']
        confidence = detection['confidence']
        
        # バウンディングボックスを描画
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # ラベルと信頼度のテキストを追加（targetに統一）
        text = f"target: {confidence:.3f}"
        cv2.putText(annotated_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # ファイル名を生成（撮影順序で並ぶように調整）
    detection_count = len(detections_with_confidence)
    # 現在の日時を取得
    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    # 連番でファイル名を生成（6桁のゼロパディング）
    filename = f"capture_{save_counter + 1:06d}_{time_str}_frame{frame_count}_{detection_count}objects.jpg"
    filepath = os.path.join(session_folder, filename)
    
    # 画像を保存
    print(f"デバッグ: 画像保存を試行します - {filepath}")
    success = cv2.imwrite(filepath, annotated_img)
    if success:
        print(f"画像を保存しました: {filepath} (検出物体数: {detection_count})")
        print(f"セッションフォルダー: {session_folder}")
        save_counter += 1
        image_save_counter += 1
    else:
        print(f"エラー: 画像の保存に失敗しました - {filepath}")
    
    # 2. 左上パネルの画像を保存（save_leftup=Trueの場合のみ）
    if save_leftup:
        # 左上パネルに表示されている画像をそのまま保存
        if leftup_info:
            leftup_filename = f"capture_{save_counter:06d}_{time_str}_frame{frame_count}_leftup_{leftup_info['label']}_{leftup_info['confidence']:.3f}.jpg"
        else:
            leftup_filename = f"capture_{save_counter:06d}_{time_str}_frame{frame_count}_leftup_no_target.jpg"
        leftup_filepath = os.path.join(session_folder, leftup_filename)
        
        # 左上パネルの画像を取得
        if leftup_image is not None:
            # パネルサイズにリサイズ
            panel_lu = cv2.resize(leftup_image, (half_w, half_h))
            
            # 物体検出がある場合のみバウンディングボックスを追加
            if leftup_info:
                # バウンディングボックスを追加
                # 元画像での座標をパネルサイズにスケール
                x1, y1, x2, y2 = leftup_info['box']
                scale_x = half_w / leftup_image.shape[1]
                scale_y = half_h / leftup_image.shape[0]
                
                panel_x1 = int(x1 * scale_x)
                panel_y1 = int(y1 * scale_y)
                panel_x2 = int(x2 * scale_x)
                panel_y2 = int(y2 * scale_y)
                
                # 座標が画像範囲内かチェック
                panel_x1 = max(0, min(panel_x1, half_w-1))
                panel_y1 = max(0, min(panel_y1, half_h-1))
                panel_x2 = max(0, min(panel_x2, half_w-1))
                panel_y2 = max(0, min(panel_y2, half_h-1))
                
                # バウンディングボックスを描画
                cv2.rectangle(panel_lu, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 255, 0), 2)
                
                # ラベルと信頼度を追加
                label = leftup_info['label']
                confidence = leftup_info['confidence']
                text = f"{label}: {confidence:.3f}"
                cv2.putText(panel_lu, text, (panel_x1, panel_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # パネル画像を保存
            cv2.imwrite(leftup_filepath, panel_lu)
            print(f"左上パネル画像を保存しました: {leftup_filename}")
            save_counter += 1
            image_save_counter += 1
    
    # 3. 左下パネルの画像を保存（save_leftdown=Trueの場合のみ）
    if save_leftdown:
        # 左下パネルに表示されている画像をそのまま保存
        if leftdown_info:
            leftdown_filename = f"capture_{save_counter:06d}_{time_str}_frame{frame_count}_leftdown_{leftdown_info['label']}_{leftdown_info['confidence']:.3f}.jpg"
        else:
            leftdown_filename = f"capture_{save_counter:06d}_{time_str}_frame{frame_count}_leftdown_no_target.jpg"
        leftdown_filepath = os.path.join(session_folder, leftdown_filename)
        
        # 左下パネルの画像を取得
        if leftdown_image is not None:
            # パネルサイズにリサイズ
            panel_ld = cv2.resize(leftdown_image, (half_w, half_h))
            
            # 物体検出がある場合のみバウンディングボックスを追加
            if leftdown_info:
                # バウンディングボックスを追加
                # 元画像での座標をパネルサイズにスケール
                x1, y1, x2, y2 = leftdown_info['box']
                scale_x = half_w / leftdown_image.shape[1]
                scale_y = half_h / leftdown_image.shape[0]
                
                panel_x1 = int(x1 * scale_x)
                panel_y1 = int(y1 * scale_y)
                panel_x2 = int(x2 * scale_x)
                panel_y2 = int(y2 * scale_y)
                
                # 座標が画像範囲内かチェック
                panel_x1 = max(0, min(panel_x1, half_w-1))
                panel_y1 = max(0, min(panel_y1, half_h-1))
                panel_x2 = max(0, min(panel_x2, half_w-1))
                panel_y2 = max(0, min(panel_y2, half_h-1))
                
                # バウンディングボックスを描画
                cv2.rectangle(panel_ld, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 255, 0), 2)
                
                # ラベルと信頼度を追加
                label = leftdown_info['label']
                confidence = leftdown_info['confidence']
                text = f"{label}: {confidence:.3f}"
                cv2.putText(panel_ld, text, (panel_x1, panel_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # パネル画像を保存
            cv2.imwrite(leftdown_filepath, panel_ld)
            print(f"左下パネル画像を保存しました: {leftdown_filename}")
            save_counter += 1
            image_save_counter += 1

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
    
    # ラベルと信頼度のテキストを追加（フォントサイズを調整、targetに統一）
    text = f"target: {confidence:.3f}"
    font_scale = min(target_width, target_height) / 600  # サイズに応じてフォントサイズを調整
    cv2.putText(resized_img, text, (relative_x1, relative_y1-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    
    return resized_img

def add_colored_border(panel, is_paused, border_thickness=8):
    """パネルに一時停止/再生状態を示す色付きの縁取りを追加"""
    if is_paused:
        # 停止中：オレンジ色の縁取り
        border_color = (0, 165, 255)  # オレンジ色 (BGR)
    else:
        # 再生中：緑色の縁取り
        border_color = (0, 255, 0)  # 緑色 (BGR)
    
    # パネルの周囲に色付きの縁取りを描画
    height, width = panel.shape[:2]
    
    # 上辺
    cv2.rectangle(panel, (0, 0), (width, border_thickness), border_color, -1)
    # 下辺
    cv2.rectangle(panel, (0, height - border_thickness), (width, height), border_color, -1)
    # 左辺
    cv2.rectangle(panel, (0, 0), (border_thickness, height), border_color, -1)
    # 右辺
    cv2.rectangle(panel, (width - border_thickness, 0), (width, height), border_color, -1)
    
    return panel

while True:
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを読み込めませんでした。")
        break

    now = time.time()

    # 推論
    results = model(frame)
    # カスタムアノテーション用にフレームをコピー
    annotated_frame = frame.copy()
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
            # ラベル名を「target」に変更
            label = "target"  # 元のラベル名に関わらず「target」に統一
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
    
    # 右上パネル用：カスタムアノテーションを適用
    for detection in detections_with_confidence:
        x1, y1, x2, y2 = detection['box']
        confidence = detection['confidence']
        
        # バウンディングボックスを描画
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # ラベルと信頼度のテキストを追加（targetに統一）
        text = f"target: {confidence:.3f}"
        cv2.putText(annotated_frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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

    # 左上のフレーム更新（0.9秒ごと、一時停止中でない場合）
    if not leftup_paused and (now - leftup_last_frame_time > 0.9):
        if detections_with_confidence:
            # 左上：最も信頼度の高い検出物体を選択
            leftup_image = detections_with_confidence[0]['crop'].copy()
            leftup_info = {
                'box': detections_with_confidence[0]['box'],
                'confidence': detections_with_confidence[0]['confidence'],
                'cls_id': detections_with_confidence[0]['cls_id'],
                'label': detections_with_confidence[0]['label']
            }
            leftup_frame_count += 1
            leftup_last_frame_time = now

    # 左下のフレーム更新（0.9秒ごと、一時停止中でない場合）
    if not leftdown_paused and (now - leftdown_last_frame_time > 0.9):
        if detections_with_confidence:
            # 左下：検出物体がある場合は2番目、ない場合は1番目を表示
            if len(detections_with_confidence) >= 2:
                # 2個以上検出された場合：2番目に信頼度の高い検出物体を選択
                target_detection = detections_with_confidence[1]
            else:
                # 1個のみ検出された場合：1番目の検出物体を選択
                target_detection = detections_with_confidence[0]
            
            leftdown_image = target_detection['crop'].copy()
            leftdown_info = {
                'box': target_detection['box'],
                'confidence': target_detection['confidence'],
                'cls_id': target_detection['cls_id'],
                'label': target_detection['label']
            }
            leftdown_frame_count += 1
            leftdown_last_frame_time = now
        # else: 何もしない（継続表示）

    # 四分割パネル作成
    half_w = max(1, window_width // 2)
    half_h = max(1, window_height // 2)

    # 左上：0.9秒ごと更新・独立一時停止可能
    if leftup_image is not None:
        panel_lu = cv2.resize(leftup_image, (half_w, half_h))
        if leftup_paused:
            status_text = "PAUSED"
            color = (0, 0, 255)
        else:
            status_text = "TARGET VIEW"
            color = (0, 255, 0)
        cv2.putText(panel_lu, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if leftup_info and 'confidence' in leftup_info:
            conf_text = f"target: {leftup_info['confidence']:.3f}"
            cv2.putText(panel_lu, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_lu = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_lu, 'No Target Image', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    
    # 左上パネルに色付きの縁取りを追加
    panel_lu = add_colored_border(panel_lu, leftup_paused)

    # 右上：検出中の画像（通常表示）
    # 左上または左下が一時停止中の場合、一時停止していない方に検出ラベルを表示
    if (leftup_paused and not leftdown_paused) or (leftdown_paused and not leftup_paused):
        # どちらか一方が一時停止中の場合、右上に検出ラベル付きの画像を表示
        panel_ru = cv2.resize(annotated_frame, (half_w, half_h))
        cv2.putText(panel_ru, "DETECTION VIEW", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    else:
        # 通常表示
        panel_ru = cv2.resize(annotated_frame, (half_w, half_h))

    # 左下：二つ検出時のみ更新・独立一時停止可能
    if leftdown_image is not None:
        panel_ld = cv2.resize(leftdown_image, (half_w, half_h))
        if leftdown_paused:
            status_text = "PAUSED"
            color = (0, 0, 255)
        else:
            status_text = "SECONDARY TARGET"
            color = (255, 255, 0)
        cv2.putText(panel_ld, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if leftdown_info and 'confidence' in leftdown_info:
            conf_text = f"target: {leftdown_info['confidence']:.3f}"
            cv2.putText(panel_ld, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_ld = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_ld, 'No Secondary Target', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    
    # 左下パネルに色付きの縁取りを追加
    panel_ld = add_colored_border(panel_ld, leftdown_paused)
    # 右下：認識情報
    panel_rd = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    info_lines = []
    if detections_with_confidence:
        info_lines.append(f'Target Detections: {len(detections_with_confidence)}')
        if len(detections_with_confidence) >= 1:
            info_lines.append(f'Top-Left(1): target {detections_with_confidence[0]["confidence"]:.3f}')
        if len(detections_with_confidence) >= 2:
            info_lines.append(f'Bottom-Left(2): target {detections_with_confidence[1]["confidence"]:.3f}')
    else:
        info_lines.append('No Target Detection')
    info_lines.append(f'LeftUp Frame: {leftup_frame_count}')
    info_lines.append(f'LeftDown Frame: {leftdown_frame_count}')
    info_lines.append(f'LeftUp: {"Paused" if leftup_paused else "Active"}')
    info_lines.append(f'LeftDown: {"Paused" if leftdown_paused else "Active"}')
    info_lines.append(f'Pause Count: {pause_counter}')
    info_lines.append(f'Image Saves: {image_save_counter}')
    info_lines.append('Controls: 1=LeftUp, 2=LeftDown, Enter=Both')
    for i, line in enumerate(info_lines):
        cv2.putText(panel_rd, line, (20, 40 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # 常に四分割で表示（左右を入れ替え）
    top = np.hstack([panel_ru, panel_lu])
    bottom = np.hstack([panel_rd, panel_ld])
    combined = np.vstack([top, bottom])
    
    cv2.imshow('Target Detection Viewer', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):  # 1キー：左上の一時停止/再開
        leftup_paused = not leftup_paused
        if leftup_paused:
            print("左上を一時停止しました")
            # 左上一時停止時に画像を保存（右上のみ）
            save_detection_images(frame, detections_with_confidence, leftup_frame_count, leftup_info, leftdown_info, save_leftup=True, save_leftdown=False, leftup_image=leftup_image, leftdown_image=leftdown_image)
            print(f"左上一時停止時の画像を保存しました (Total: {save_counter} files)")
            pause_counter += 1
        else:
            print("左上を再開しました")
    elif key == ord('2'):  # 2キー：左下の一時停止/再開
        leftdown_paused = not leftdown_paused
        if leftdown_paused:
            print("左下を一時停止しました")
            # 左下一時停止時に画像を保存（右下のみ）
            save_detection_images(frame, detections_with_confidence, leftdown_frame_count, leftup_info, leftdown_info, save_leftup=False, save_leftdown=True, leftup_image=leftup_image, leftdown_image=leftdown_image)
            print(f"左下一時停止時の画像を保存しました (Total: {save_counter} files)")
            pause_counter += 1
        else:
            print("左下を再開しました")
    elif key == 13:  # エンターキー：1キーと2キーを同時に押したもの（両方の一時停止/再開）
        # 両方の状態を同時に切り替え
        leftup_paused = not leftup_paused
        leftdown_paused = not leftdown_paused
        
        if leftup_paused and leftdown_paused:
            print("左上と左下を同時に一時停止しました")
            # 両方一時停止時に画像を保存（両方）
            save_detection_images(frame, detections_with_confidence, leftup_frame_count, leftup_info, leftdown_info, save_leftup=True, save_leftdown=True, leftup_image=leftup_image, leftdown_image=leftdown_image)
            print(f"両方一時停止時の画像を保存しました (Total: {save_counter} files)")
            pause_counter += 1
        else:
            print("左上と左下を同時に再開しました")

cap.release()
cv2.destroyAllWindows()
print("ターゲット検出を終了しました。")
print(f"保存された画像の総数: {save_counter}枚")
print(f"保存先フォルダー: {session_folder}")


  