import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime

# ----------------------------------------------------
# 1. OBS仮想カメラの読み込み設定
# ----------------------------------------------------
cap = cv2.VideoCapture(1)  # 仮想カメラのインデックス。環境に合わせて変更してください
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open OBS virtual camera.")
    print("Please check if 'Start Virtual Camera' is enabled in OBS or verify the correct camera index.")
    exit()

# sabiフォルダーの作成
save_folder = "sabi"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Created folder: {save_folder}")

# 実行セッション用のフォルダーを作成（一回の実行につき一個）
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_folder = os.path.join(save_folder, f"session_{session_timestamp}")
os.makedirs(session_folder)
print(f"Created session folder: {session_folder}")

# YOLOv8モデルのパス（sabi.ptで板検出を行う）
model = YOLO('sabi.pt')  # sabi.ptで板を検出し、板領域内でHSV錆分析を実行

print("Starting real-time rust detection. Press 'q' to exit.")

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

cv2.namedWindow('Rust Detection Viewer', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Rust Detection Viewer', window_width, window_height)
cv2.setWindowProperty('Rust Detection Viewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

def analyze_rust_area_detailed(frame, plate_detections):
    """板検出結果に基づいてHSV色空間で詳細な錆分析を行う関数"""
    rust_analysis = {
        'total_rust_area': 0,
        'rust_count': 0,
        'rust_ratio': 0,
        'board_area': 200 * 200,  # 板の実寸面積 (200×200 mm)
        'detected_plates': len(plate_detections),
        'rust_details': []
    }
    
    if not plate_detections:
        return rust_analysis
    
    # 各板について詳細な錆分析を実行
    for i, plate_detection in enumerate(plate_detections):
        x1, y1, x2, y2 = plate_detection['box']
        
        # 板の領域を切り出し
        plate_region = frame[y1:y2, x1:x2]
        
        if plate_region.size == 0:
            continue
            
        # HSV色空間に変換
        hsv = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)
        
        # 錆っぽい色範囲でマスクを作成
        lower_brown = np.array([5, 80, 40])    # H, S, V の下限
        upper_brown = np.array([20, 255, 200]) # H, S, V の上限
        rust_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # ノイズ除去
        kernel = np.ones((3,3), np.uint8)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 有効な錆輪郭をフィルタリング
        valid_rust_contours = []
        plate_rust_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            if perimeter == 0:
                continue
            
            # 面積フィルタ (小さすぎ/大きすぎを除外) - 最大面積を増加
            if area < 30 or area > 10000:  # 最大面積を5000から10000に増加
                continue
            
            # 円形度フィルタ
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.6:  # 円に近いかどうか
                continue
            
            valid_rust_contours.append(cnt)
            plate_rust_area += area
        
        # 板領域の面積
        plate_area = (x2 - x1) * (y2 - y1)
        plate_rust_ratio = (plate_rust_area / plate_area) * 100 if plate_area > 0 else 0
        
        # 板ごとの詳細情報を記録
        plate_detail = {
            'plate_id': i + 1,
            'plate_box': (x1, y1, x2, y2),
            'plate_area': plate_area,
            'rust_contours_count': len(valid_rust_contours),
            'rust_area': plate_rust_area,
            'rust_ratio': plate_rust_ratio,
            'rust_mask': rust_mask,
            'plate_region': plate_region
        }
        
        rust_analysis['rust_details'].append(plate_detail)
        rust_analysis['total_rust_area'] += plate_rust_area
        rust_analysis['rust_count'] += len(valid_rust_contours)
    
    # 全体の錆割合を計算
    total_plate_area = sum(detail['plate_area'] for detail in rust_analysis['rust_details'])
    if total_plate_area > 0:
        rust_analysis['rust_ratio'] = (rust_analysis['total_rust_area'] / total_plate_area) * 100
    
    return rust_analysis

def analyze_rust_area(frame, detections_with_confidence):
    """後方互換性のための簡易版錆分析関数"""
    # 板検出（sabi.ptでの検出結果）のみを抽出
    plate_detections = [d for d in detections_with_confidence if 'plate' in d.get('label', '').lower() or 'sabi' in d.get('label', '').lower()]
    
    # 板が検出されていない場合は従来の方法を使用
    if not plate_detections:
        plate_detections = detections_with_confidence  # 全ての検出を板として扱う
    
    return analyze_rust_area_detailed(frame, plate_detections)

def create_rust_mask(frame, detections_with_confidence):
    """錆検出結果からマスク画像を作成"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for detection in detections_with_confidence:
        x1, y1, x2, y2 = detection['box']
        # バウンディングボックス内を白で塗りつぶし
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # マスクをカラー画像に変換
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    return mask_colored

def mark_rust_on_frame(frame, rust_analysis):
    """フレーム上に錆を分かりやすくマーキングする関数"""
    marked_frame = frame.copy()
    
    rust_details = rust_analysis.get('rust_details', [])
    
    for detail in rust_details:
        plate_box = detail['plate_box']
        x1, y1, x2, y2 = plate_box
        plate_region = frame[y1:y2, x1:x2]
        
        if plate_region.size == 0:
            continue
            
        # HSV変換して錆検出
        hsv = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)
        lower_brown = np.array([5, 80, 40])
        upper_brown = np.array([20, 255, 200])
        rust_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # ノイズ除去
        kernel = np.ones((3,3), np.uint8)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rust_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            if perimeter == 0:
                continue
            
            # sabi-M.pyスタイルの適応的面積フィルタ（画像サイズに応じて調整）
            plate_area_pixels = (x2 - x1) * (y2 - y1)
            min_area = max(10, plate_area_pixels // 1000)  # 板サイズに応じた最小面積
            max_area = plate_area_pixels // 2  # 板の1/2以下に増加（より大きな錆も検出）
            
            if area < min_area or area > max_area:
                continue
            
            # より緩い円形度フィルタ（sabi-M.pyから）
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.4:  # より緩い条件で多様な錆形状を検出
                continue
            
            rust_count += 1
            
            # 元のフレーム座標系に変換
            offset_cnt = cnt + np.array([x1, y1])
            
            # 錆領域を強調表示
            # 1. 半透明の赤でハイライト
            rust_overlay = marked_frame.copy()
            cv2.fillPoly(rust_overlay, [offset_cnt], (0, 150, 255))  # 明るい赤
            marked_frame = cv2.addWeighted(marked_frame, 0.8, rust_overlay, 0.2, 0)
            
            # 2. 太い赤い輪郭
            cv2.drawContours(marked_frame, [offset_cnt], -1, (0, 0, 255), 4)
            
            # 3. 中心に目立つマーカー
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"]) + x1
                cy = int(M["m01"]/M["m00"]) + y1
                
                # 大きな黄色い×マーク
                cv2.line(marked_frame, (cx-15, cy-15), (cx+15, cy+15), (0, 255, 255), 6)
                cv2.line(marked_frame, (cx-15, cy+15), (cx+15, cy-15), (0, 255, 255), 6)
                cv2.line(marked_frame, (cx-15, cy-15), (cx+15, cy+15), (0, 0, 0), 2)
                cv2.line(marked_frame, (cx-15, cy+15), (cx+15, cy-15), (0, 0, 0), 2)
                
                # sabi-M.pyスタイルの改良された錆番号表示
                # 大きな白い円の背景
                cv2.circle(marked_frame, (cx, cy), 20, (255, 255, 255), -1)  # 白い背景円
                cv2.circle(marked_frame, (cx, cy), 20, (0, 0, 255), 3)       # 赤い縁取り
                
                # 番号を大きく見やすく表示
                rust_text = str(rust_count)
                font_scale = 1.0
                thickness = 2
                
                # テキストサイズを取得してセンタリング
                (text_width, text_height), baseline = cv2.getTextSize(rust_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = cx - text_width // 2
                text_y = cy + text_height // 2
                
                # 番号を描画
                cv2.putText(marked_frame, rust_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        
        # 板のバウンディングボックスも強調
        cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (255, 255, 0), 3)  # 水色の枠
        
        # 板情報を表示
        plate_text = f"PLATE-{detail['plate_id']} ({rust_count} rusts)"
        cv2.putText(marked_frame, plate_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    return marked_frame

def mark_rust_on_detected_image(image, detection_info, target_width, target_height):
    """sabi-M.pyスタイルの検出された個別画像に錆をマーキングする関数"""
    if detection_info is None or image is None:
        return image
    
    # 画像のコピーを作成
    marked_image = image.copy()
    
    try:
        # HSV色空間に変換
        hsv = cv2.cvtColor(marked_image, cv2.COLOR_BGR2HSV)
        
        # 錆っぽい色範囲でマスクを作成
        lower_brown = np.array([5, 80, 40])    # H, S, V の下限
        upper_brown = np.array([20, 255, 200]) # H, S, V の上限
        rust_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # ノイズ除去
        kernel = np.ones((3,3), np.uint8)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 有効な錆輪郭をフィルタリングして描画
        rust_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            if perimeter == 0:
                continue
            
            # 画像サイズに応じた面積フィルタ（sabi-M.pyから）
            min_area = max(10, (target_width * target_height) // 10000)
            max_area = (target_width * target_height) // 2  # 画像の1/2以下に増加
            
            if area < min_area or area > max_area:
                continue
            
            # より緩い円形度フィルタ
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.4:
                continue
            
            rust_count += 1
            
            # 錆の輪郭を太い赤線で描画
            cv2.drawContours(marked_image, [cnt], -1, (0, 0, 255), 3)
            
            # 錆の中心に番号を表示（sabi-M.pyスタイル）
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                
                # 白い背景円に番号を表示
                radius = 15
                cv2.circle(marked_image, (cx, cy), radius, (255, 255, 255), -1)  # 白い背景
                cv2.circle(marked_image, (cx, cy), radius, (0, 0, 255), 2)       # 赤い縁取り
                
                # 番号を描画
                rust_text = str(rust_count)
                font_scale = 0.7
                thickness = 2
                
                (text_width, text_height), baseline = cv2.getTextSize(rust_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = cx - text_width // 2
                text_y = cy + text_height // 2
                
                cv2.putText(marked_image, rust_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        
    except Exception as e:
        print(f"Error in rust marking on detected image: {e}")
        return marked_image
    
    return marked_image

def save_detection_images(frame, detections_with_confidence, frame_count, leftup_info, leftdown_info, rust_analysis):
    """一時停止時に検出画像を保存する関数"""
    global save_counter
    
    if not detections_with_confidence:
        return
    
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
    
    # 詳細な錆分析結果を画像上に表示
    analysis_text = [
        f"Plates: {rust_analysis.get('detected_plates', 0)}",
        f"Rust Spots: {rust_analysis['rust_count']}",
        f"Total Area: {rust_analysis['total_rust_area']:.0f}px",
        f"Ratio: {rust_analysis['rust_ratio']:.2f}%"
    ]
    
    for i, text in enumerate(analysis_text):
        cv2.putText(annotated_img, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # ファイル名を生成（検出数も含める）
    detection_count = len(detections_with_confidence)
    filename = f"rust_detection_{session_timestamp}_{frame_count:06d}.jpg"
    filepath = os.path.join(session_folder, filename)
    
    # 画像を保存
    cv2.imwrite(filepath, annotated_img)
    print(f"Saved rust detection image: {filename} (Detected rust count: {detection_count})")
    save_counter += 1
    
    # 2. 各板の詳細な錆マスク画像を個別保存
    for detail in rust_analysis.get('rust_details', []):
        plate_id = detail['plate_id']
        rust_mask = detail['rust_mask']
        plate_region = detail['plate_region']
        
        # 錆マスクを保存
        mask_filename = f"plate_{plate_id}_rust_mask_{session_timestamp}_{frame_count:06d}.jpg"
        mask_filepath = os.path.join(session_folder, mask_filename)
        cv2.imwrite(mask_filepath, rust_mask)
        
        # 板領域に錆の輪郭を描画した画像を保存
        plate_with_rust = plate_region.copy()
        
        # HSV変換して錆検出
        hsv = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)
        lower_brown = np.array([5, 80, 40])
        upper_brown = np.array([20, 255, 200])
        temp_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # ノイズ除去
        kernel = np.ones((3,3), np.uint8)
        temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, kernel)
        
        # 輪郭検出と描画
        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rust_contour_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            if perimeter == 0 or area < 30 or area > 10000:  # 最大面積を5000から10000に増加
                continue
                
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.6:
                continue
            
            rust_contour_count += 1
            
            # 錆の輪郭を太い赤線で強調
            cv2.drawContours(plate_with_rust, [cnt], -1, (0, 0, 255), 3)  # 赤枠（太くした）
            
            # 錆領域を半透明の赤で塗りつぶし
            rust_overlay = plate_with_rust.copy()
            cv2.fillPoly(rust_overlay, [cnt], (0, 100, 255))  # 明るい赤で塗りつぶし
            plate_with_rust = cv2.addWeighted(plate_with_rust, 0.7, rust_overlay, 0.3, 0)
            
            # 錆の中心に円形マーカーを追加
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                
                # 大きな黄色い円で中心をマーク
                cv2.circle(plate_with_rust, (cx, cy), 8, (0, 255, 255), -1)  # 黄色の塗りつぶし円
                cv2.circle(plate_with_rust, (cx, cy), 8, (0, 0, 0), 2)      # 黒い枠線
                
                # 錆番号を大きく、背景付きで表示
                text = f"RUST-{rust_contour_count}"
                font_scale = 0.8
                thickness = 2
                
                # テキストサイズを取得
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # 背景の矩形を描画（白背景、黒枠）
                bg_x1 = cx - text_width // 2 - 5
                bg_y1 = cy - 25 - text_height
                bg_x2 = cx + text_width // 2 + 5
                bg_y2 = cy - 25 + baseline
                
                cv2.rectangle(plate_with_rust, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)  # 白背景
                cv2.rectangle(plate_with_rust, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), 1)        # 黒枠
                
                # テキストを描画
                cv2.putText(plate_with_rust, text, (cx - text_width // 2, cy - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        
        # 板上の錆分析画像を保存
        plate_rust_filename = f"plate_{plate_id}_rust_analysis_{session_timestamp}_{frame_count:06d}.jpg"
        plate_rust_filepath = os.path.join(session_folder, plate_rust_filename)
        cv2.imwrite(plate_rust_filepath, plate_with_rust)
        
        print(f"Saved rust analysis image for plate {plate_id}: {plate_rust_filename}")
        save_counter += 2  # マスクと分析画像の2枚
    
    # 3. 詳細な分析結果をテキストファイルとして保存
    analysis_filename = f"rust_analysis_{session_timestamp}_{frame_count:06d}.txt"
    analysis_filepath = os.path.join(session_folder, analysis_filename)
    with open(analysis_filepath, 'w', encoding='utf-8') as f:
        f.write(f"Rust Detection Analysis Results - Frame {frame_count}\n")
        f.write(f"Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Detected Plates: {rust_analysis.get('detected_plates', 0)}\n")
        f.write(f"Total Rust Count: {rust_analysis['rust_count']}\n")
        f.write(f"Total Rust Area (pixels): {rust_analysis['total_rust_area']:.0f}\n")
        f.write(f"Overall Rust Ratio (%): {rust_analysis['rust_ratio']:.2f}\n")
        f.write(f"Plate Reference Area (mm²): {rust_analysis['board_area']}\n")
        f.write("\n=== Plate-by-Plate Details ===\n")
        
        for detail in rust_analysis.get('rust_details', []):
            f.write(f"\nPlate {detail['plate_id']}:\n")
            f.write(f"  Position: ({detail['plate_box'][0]}, {detail['plate_box'][1]}) - ({detail['plate_box'][2]}, {detail['plate_box'][3]})\n")
            f.write(f"  Plate Area: {detail['plate_area']:.0f} pixels\n")
            f.write(f"  Rust Spots: {detail['rust_contours_count']}\n")
            f.write(f"  Rust Area: {detail['rust_area']:.0f} pixels\n")
            f.write(f"  Rust Ratio: {detail['rust_ratio']:.2f}%\n")
        
        f.write("\n=== YOLO Detection Info ===\n")
        for i, detection in enumerate(detections_with_confidence):
            f.write(f"Detection {i+1}: {detection['label']} (Confidence: {detection['confidence']:.3f})\n")
            f.write(f"  Position: ({detection['box'][0]}, {detection['box'][1]}) - ({detection['box'][2]}, {detection['box'][3]})\n")
    
    print(f"Saved detailed analysis results: {analysis_filename}")

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

# メインループ
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
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

    # 錆の面積分析を実行
    rust_analysis = analyze_rust_area(frame, detections_with_confidence)

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

    # 四分割パネル作成
    half_w = max(1, window_width // 2)
    half_h = max(1, window_height // 2)

    # 左上：錆マーキング強化版（0.5秒ごと更新・一時停止可能）
    if leftup_image is not None:
        # sabi-M.pyスタイルの錆マーキングを適用
        marked_leftup = mark_rust_on_detected_image(leftup_image, leftup_info, half_w, half_h)
        panel_lu = cv2.resize(marked_leftup, (half_w, half_h))
        
        status_text = "PAUSED" if leftup_paused else "RUST ANALYSIS"
        # 背景付きでステータステキストを表示
        cv2.rectangle(panel_lu, (15, 15), (300, 55), (0, 0, 0), -1)
        cv2.putText(panel_lu, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if not leftup_paused else (0, 0, 255), 2)
        
        if leftup_info and 'confidence' in leftup_info:
            conf_text = f"{leftup_info.get('label','')}: {leftup_info['confidence']:.3f}"
            cv2.rectangle(panel_lu, (15, 60), (350, 100), (0, 0, 0), -1)
            cv2.putText(panel_lu, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_lu = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_lu, 'No Rust Detected', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)

    # 右上：検出中の画像（YOLOバウンディングボックス表示）
    panel_ru = cv2.resize(annotated_frame, (half_w, half_h))

    # 左下：二つ検出時のみ更新。無い場合は前回の画像を継続表示。
    if leftdown_image is not None:
        panel_ld = cv2.resize(leftdown_image, (half_w, half_h))
        if leftdown_info and 'confidence' in leftdown_info:
            conf_text = f"{leftdown_info.get('label','')}: {leftdown_info['confidence']:.3f}"
            cv2.putText(panel_ld, conf_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_ld = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_ld, 'No Secondary Rust', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    
    # 右下：認識情報と詳細錆分析結果
    panel_rd = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    info_lines = []
    
    if detections_with_confidence:
        # 基本情報
        info_lines.append(f'Detected Plates: {rust_analysis.get("detected_plates", 0)}')
        info_lines.append(f'Rust Spots: {rust_analysis["rust_count"]}')
        info_lines.append(f'Total Rust Area: {rust_analysis["total_rust_area"]:.0f}px')
        info_lines.append(f'Rust Ratio: {rust_analysis["rust_ratio"]:.2f}%')
        
        # 板別詳細情報（最大2板まで表示）
        rust_details = rust_analysis.get('rust_details', [])
        for i, detail in enumerate(rust_details[:2]):  # 最大2板まで
            info_lines.append(f'Plate{detail["plate_id"]}: {detail["rust_contours_count"]} spots {detail["rust_ratio"]:.1f}%')
        
        # YOLO検出情報（最大2個まで）
        if len(detections_with_confidence) >= 1:
            info_lines.append(f'Detection1: {detections_with_confidence[0]["label"]} {detections_with_confidence[0]["confidence"]:.3f}')
        if len(detections_with_confidence) >= 2:
            if leftup_info:
                same_label = [d for d in detections_with_confidence if d['label'] == leftup_info['label']]
                if len(same_label) >= 2:
                    sec = same_label[1]
                else:
                    sec = detections_with_confidence[1]
                info_lines.append(f'Detection2: {sec["label"]} {sec["confidence"]:.3f}')
    else:
        info_lines.append('No Rust Detected')
    
    info_lines.append(f'Frame: {leftup_frame_count}')
    info_lines.append(f'Status: {"Paused" if leftup_paused else "Playing"}')
    info_lines.append('Enter: Save/Play')
    
    # フォントサイズを調整して表示
    font_scale = 0.8
    line_height = 40
    for i, line in enumerate(info_lines):
        y_pos = 30 + i * line_height
        if y_pos < half_h - 20:  # 画面からはみ出さないように
            cv2.putText(panel_rd, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    
    # 常に四分割で表示（左右を入れ替え）
    top = np.hstack([panel_ru, panel_lu])
    bottom = np.hstack([panel_rd, panel_ld])
    combined = np.vstack([top, bottom])
    
    cv2.imshow('Rust Detection Viewer', combined)

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
            print("Resumed playback")
        else:
            # 再生中の場合：一時停止
            leftup_paused = True
            print("Paused")
            # 一時停止時に検出画像を保存
            if detections_with_confidence:
                save_detection_images(frame, detections_with_confidence, leftup_frame_count, leftup_info, leftdown_info, rust_analysis)
                print(f"Saved rust detection images and analysis results (Total: {save_counter} files)")

cap.release()
cv2.destroyAllWindows()
print("Rust detection completed.")
print(f"Total saved images: {save_counter} files")
print(f"Save folder: {session_folder}")
