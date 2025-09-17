
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime

# ----------------------------------------------------
# 1. 動画ファイルの読み込み設定
# ----------------------------------------------------
import sys

# 動画ファイルパスを指定（コマンドライン引数または既定のファイル）
if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    # 既定の動画ファイル（プロジェクトフォルダ内の動画を検索）
    possible_videos = ["sabi-test.MOV", "test.mov", "akarui.mkv", "akarui-ari.mp4"]
    video_path = None
    for video_file in possible_videos:
        if os.path.exists(video_file):
            video_path = video_file
            break
    
    if video_path is None:
        print("エラー: テスト用動画ファイルが見つかりません。")
        print("使用方法: python sabi-M.py [動画ファイルパス]")
        print("または以下のファイルのいずれかを配置してください:")
        for video_file in possible_videos:
            print(f"  - {video_file}")
        exit()

print(f"動画ファイルを読み込みます: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"エラー: 動画ファイル '{video_path}' を開けませんでした。")
    print("ファイルが存在し、対応フォーマットかご確認ください。")
    exit()

# 動画情報を取得
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames / fps if fps > 0 else 0

print(f"動画情報: {total_frames}フレーム, {fps:.2f}FPS, {duration:.2f}秒")

# sabiフォルダーの作成
save_folder = "sabi"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"フォルダーを作成しました: {save_folder}")

# 実行セッション用のフォルダーを作成（一回の実行につき一個）
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_folder = os.path.join(save_folder, f"session_{session_timestamp}")
os.makedirs(session_folder)
print(f"セッションフォルダーを作成しました: {session_folder}")

# YOLOv8モデルのパス（sabi.ptで板検出を行う）
model = YOLO('sabi.pt')  # sabi.ptで板を検出し、板領域内でHSV錆分析を実行

print("錆検出動画分析を開始します。")
print("操作方法:")
print("  q: 終了")
print("  Enter: 検出結果保存")
print("  スペース: 動画の一時停止/再生")
print("  ←/→: 前/次のフレーム（動画一時停止中）")
print("  r: 動画を最初から再生")
print("  d: 検出表示の一時停止/再生")

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

# --- 左下用の変数（錆マスク表示用） ---
leftdown_rust_mask = None
leftdown_rust_info = None

# --- 保存用のカウンター ---
save_counter = 0

# --- 動画再生制御用の変数 ---
video_paused = False  # 動画の一時停止フラグ
current_frame_pos = 0  # 現在のフレーム位置
frame_step = 1  # フレームステップ（通常再生時）

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
            
            # 面積フィルタ (小さすぎ/大きすぎを除外)
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

def save_detection_images(frame, detections_with_confidence, frame_count, leftup_info, leftdown_rust_info, rust_analysis):
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
    print(f"錆検出画像を保存しました: {filename} (検出錆数: {detection_count})")
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
            
            # 錆のサイズを分類（2分別）
            if area < 1000:
                rust_size = "SMALL"
                rust_color = (0, 255, 255)    # 黄色
                line_thickness = 2
                marker_radius = 8
                font_scale = 0.7
            else:
                rust_size = "LARGE"
                rust_color = (0, 0, 255)      # 赤
                line_thickness = 4
                marker_radius = 12
                font_scale = 0.8
            
            # サイズに応じた錆の輪郭を描画
            cv2.drawContours(plate_with_rust, [cnt], -1, rust_color, line_thickness)
            
            # 錆領域を半透明で塗りつぶし（サイズに応じて色を変更）
            rust_overlay = plate_with_rust.copy()
            cv2.fillPoly(rust_overlay, [cnt], rust_color)
            plate_with_rust = cv2.addWeighted(plate_with_rust, 0.8, rust_overlay, 0.2, 0)
            
            # 錆の中心にサイズ別マーカーを追加
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                
                # サイズに応じた円形マーカー
                cv2.circle(plate_with_rust, (cx, cy), marker_radius, (255, 255, 255), -1)  # 白い背景円
                cv2.circle(plate_with_rust, (cx, cy), marker_radius, (0, 0, 0), 2)         # 黒い枠線
                
                # 錆番号とサイズを表示
                text = f"{rust_size}-{rust_contour_count}"
                thickness = 2
                
                # テキストサイズを取得
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # 背景の矩形を描画（白背景、サイズ色の枠）
                bg_x1 = cx - text_width // 2 - 5
                bg_y1 = cy - 30 - text_height
                bg_x2 = cx + text_width // 2 + 5
                bg_y2 = cy - 30 + baseline
                
                cv2.rectangle(plate_with_rust, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)  # 白背景
                cv2.rectangle(plate_with_rust, (bg_x1, bg_y1), (bg_x2, bg_y2), rust_color, 2)       # サイズ色の枠
                
                # テキストを描画
                cv2.putText(plate_with_rust, text, (cx - text_width // 2, cy - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, rust_color, thickness)
                
                # 面積も小さく表示
                area_text = f"{area}px"
                cv2.putText(plate_with_rust, area_text, (cx - 20, cy + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, rust_color, 1)
        
        # 板上の錆分析画像を保存
        plate_rust_filename = f"plate_{plate_id}_rust_analysis_{session_timestamp}_{frame_count:06d}.jpg"
        plate_rust_filepath = os.path.join(session_folder, plate_rust_filename)
        cv2.imwrite(plate_rust_filepath, plate_with_rust)
        
        print(f"板{plate_id}の錆分析画像を保存しました: {plate_rust_filename}")
        save_counter += 2  # マスクと分析画像の2枚
    
    # 3. 詳細な分析結果をテキストファイルとして保存
    analysis_filename = f"rust_analysis_{session_timestamp}_{frame_count:06d}.txt"
    analysis_filepath = os.path.join(session_folder, analysis_filename)
    with open(analysis_filepath, 'w', encoding='utf-8') as f:
        f.write(f"錆検出分析結果 - フレーム {frame_count}\n")
        f.write(f"検出日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"検出板数: {rust_analysis.get('detected_plates', 0)}\n")
        f.write(f"錆の総個数: {rust_analysis['rust_count']}\n")
        f.write(f"錆の総面積 (ピクセル): {rust_analysis['total_rust_area']:.0f}\n")
        f.write(f"全体錆割合 (%): {rust_analysis['rust_ratio']:.2f}\n")
        f.write(f"板の基準面積 (mm²): {rust_analysis['board_area']}\n")
        f.write("\n=== 板別詳細情報 ===\n")
        
        for detail in rust_analysis.get('rust_details', []):
            f.write(f"\n板 {detail['plate_id']}:\n")
            f.write(f"  位置: ({detail['plate_box'][0]}, {detail['plate_box'][1]}) - ({detail['plate_box'][2]}, {detail['plate_box'][3]})\n")
            f.write(f"  板面積: {detail['plate_area']:.0f} ピクセル\n")
            f.write(f"  錆スポット数: {detail['rust_contours_count']}\n")
            f.write(f"  錆面積: {detail['rust_area']:.0f} ピクセル\n")
            f.write(f"  錆割合: {detail['rust_ratio']:.2f}%\n")
        
        f.write("\n=== YOLO検出情報 ===\n")
        for i, detection in enumerate(detections_with_confidence):
            f.write(f"検出 {i+1}: {detection['label']} (信頼度: {detection['confidence']:.3f})\n")
            f.write(f"  位置: ({detection['box'][0]}, {detection['box'][1]}) - ({detection['box'][2]}, {detection['box'][3]})\n")
    
    print(f"詳細分析結果を保存しました: {analysis_filename}")

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

def mark_rust_on_image(image, detection_info, target_width, target_height):
    """検出された板の画像に錆をマークして表示用画像を作成"""
    if detection_info is None:
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
            
            # 面積フィルタ (画像サイズに応じて調整)
            min_area = max(10, (target_width * target_height) // 10000)  # 画像サイズに応じた最小面積
            max_area = (target_width * target_height) // 2  # 画像の1/2以下に増加
            
            if area < min_area or area > max_area:
                continue
            
            # 円形度フィルタ
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.4:  # より緩い条件
                continue
            
            rust_count += 1
            
            # 錆のサイズを分類（2分別）
            if area < 1000:
                rust_size = "S"
                rust_color = (0, 255, 255)    # 黄色
                line_thickness = 2
                radius = 15
                font_scale = 0.7
            else:
                rust_size = "L"
                rust_color = (0, 0, 255)      # 赤
                line_thickness = 4
                radius = 18
                font_scale = 0.8
            
            # サイズに応じた錆の輪郭を描画
            cv2.drawContours(marked_image, [cnt], -1, rust_color, line_thickness)
            
            # 錆の中心にサイズ別マーカーを表示
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                
                # サイズに応じた背景円
                cv2.circle(marked_image, (cx, cy), radius, (255, 255, 255), -1)  # 白い背景
                cv2.circle(marked_image, (cx, cy), radius, rust_color, 2)        # サイズ色の縁取り
                
                # 番号とサイズを描画
                rust_text = f"{rust_size}{rust_count}"
                thickness = 2
                
                (text_width, text_height), baseline = cv2.getTextSize(rust_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = cx - text_width // 2
                text_y = cy + text_height // 2
                
                cv2.putText(marked_image, rust_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, rust_color, thickness)
        
        # 錆の総数表示は削除（文字なし）
        
    except Exception as e:
        print(f"Error in rust marking: {e}")
        return marked_image
    
    return marked_image

# メインループ
while True:
    # 動画が一時停止中でなければフレームを読み込み
    if not video_paused:
        ret, frame = cap.read()
        if not ret:
            print("動画が終了しました。")
            print("'r'キーで最初から再生するか、'q'キーで終了してください。")
            # 動画終了時は一時停止状態にする
            video_paused = True
            continue
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # 一時停止中は現在のフレームを保持
    if video_paused and 'frame' not in locals():
        # 最初の一時停止時にフレームがない場合は読み込み
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

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

    # 左上の更新（新しい検出があった場合のみ、一時停止中でない場合）
    if not leftup_paused and detected_history:
        # 新しい検出があった場合のみ更新
        leftup_image = detected_history[0].copy()
        leftup_info = detected_info
        leftup_frame_count += 1

        # 左下用錆マスクは後で設定（pause時など）

    # 四分割パネル作成
    half_w = max(1, window_width // 2)
    half_h = max(1, window_height // 2)

    # 左上：錆をマークした画像表示（一時停止可能）
    if leftup_image is not None:
        # 錆をマークした画像を作成
        marked_image = mark_rust_on_image(leftup_image, leftup_info, half_w, half_h)
        panel_lu = cv2.resize(marked_image, (half_w, half_h))
        
        # pause状態を表示
        status_text = "PAUSED" if leftup_paused else "RUST ANALYSIS"
        status_color = (0, 0, 255) if leftup_paused else (0, 255, 0)
        
        # 背景付きでステータステキストを表示
        cv2.rectangle(panel_lu, (15, 15), (300, 55), (0, 0, 0), -1)
        cv2.putText(panel_lu, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        if leftup_info and 'confidence' in leftup_info:
            conf_text = f"{leftup_info.get('label','')}: {leftup_info['confidence']:.3f}"
            cv2.rectangle(panel_lu, (15, 60), (350, 100), (0, 0, 0), -1)
            cv2.putText(panel_lu, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_lu = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_lu, 'No Detection Yet', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)

    # 右上：検出中の画像（通常表示）
    panel_ru = cv2.resize(annotated_frame, (half_w, half_h))

    # 左下：錆マスク表示用（pause時に更新される）
    if leftdown_rust_mask is not None:
        # rust_maskを3チャンネルのカラー画像に変換
        rust_mask_color = cv2.cvtColor(leftdown_rust_mask, cv2.COLOR_GRAY2BGR)
        # 錆部分を赤色で強調
        rust_mask_color[leftdown_rust_mask > 0] = [0, 0, 255]  # 赤色
        
        panel_ld = cv2.resize(rust_mask_color, (half_w, half_h))
        cv2.putText(panel_ld, "RUST MASK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if leftdown_rust_info:
            info_text = f"Spots: {leftdown_rust_info.get('rust_count', 0)}"
            cv2.putText(panel_ld, info_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            area_text = f"Area: {leftdown_rust_info.get('rust_area', 0):.0f}px"
            cv2.putText(panel_ld, area_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ratio_text = f"Ratio: {leftdown_rust_info.get('rust_ratio', 0):.2f}%"
            cv2.putText(panel_ld, ratio_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        panel_ld = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_ld, 'Pause to Show Rust Mask', (20, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # 右下：認識情報と詳細錆分析結果
    panel_rd = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    info_lines = []
    
    if detections_with_confidence:
        # 基本情報（英語表示で文字化け回避）
        info_lines.append(f'Plates: {rust_analysis.get("detected_plates", 0)}')
        info_lines.append(f'Rust Spots: {rust_analysis["rust_count"]}')
        info_lines.append(f'Total Area: {rust_analysis["total_rust_area"]:.0f}px')
        info_lines.append(f'Rust Ratio: {rust_analysis["rust_ratio"]:.2f}%')
        
        # 板別詳細情報（最大2板まで表示）
        rust_details = rust_analysis.get('rust_details', [])
        for i, detail in enumerate(rust_details[:2]):  # 最大2板まで
            info_lines.append(f'Plate{detail["plate_id"]}: {detail["rust_contours_count"]} spots {detail["rust_ratio"]:.1f}%')
        
        # YOLO検出情報（最大2個まで）
        if len(detections_with_confidence) >= 1:
            info_lines.append(f'Det1: {detections_with_confidence[0]["label"]} {detections_with_confidence[0]["confidence"]:.3f}')
        if len(detections_with_confidence) >= 2:
            if leftup_info:
                same_label = [d for d in detections_with_confidence if d['label'] == leftup_info['label']]
                if len(same_label) >= 2:
                    sec = same_label[1]
                else:
                    sec = detections_with_confidence[1]
                info_lines.append(f'Det2: {sec["label"]} {sec["confidence"]:.3f}')
    else:
        info_lines.append('No Rust Detected')
    
    info_lines.append(f'Frame: {current_frame_pos}/{total_frames}')
    info_lines.append(f'Time: {current_frame_pos/fps:.1f}s/{duration:.1f}s')
    info_lines.append(f'Video: {"Paused" if video_paused else "Playing"}')
    info_lines.append(f'Display: {"Paused" if leftup_paused else "Playing"}')
    info_lines.append('Enter:Save Space:Play/Pause')
    
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

    key = cv2.waitKey(30) & 0xFF  # 動画再生のため少し長めの待機時間
    
    if key == ord('q'):
        break
    elif key == 13:  # エンターキー（ASCII: 13） - 左上一時停止/再開
        if leftup_paused:
            # 一時停止中の場合：現在のフレームまで飛ばす
            if detected_history:
                leftup_image = detected_history[0].copy()
                leftup_info = detected_info
                leftup_frame_count += 1
            leftup_paused = False
            print("Left-up display resumed")
        else:
            # 再生中の場合：一時停止
            leftup_paused = True
            print("Left-up display paused")
            
            # 一時停止時に錆マスクを左下に表示するため更新
            if rust_analysis and rust_analysis.get('rust_details'):
                # 最初の板のrust_maskを使用
                first_plate_detail = rust_analysis['rust_details'][0]
                leftdown_rust_mask = first_plate_detail.get('rust_mask')
                leftdown_rust_info = {
                    'rust_count': first_plate_detail.get('rust_contours_count', 0),
                    'rust_area': first_plate_detail.get('rust_area', 0),
                    'rust_ratio': first_plate_detail.get('rust_ratio', 0),
                    'plate_id': first_plate_detail.get('plate_id', 1)
                }
                print(f"Updated rust mask for left-down panel (Plate {leftdown_rust_info['plate_id']})")
            
            # 一時停止時に検出画像を保存
            if detections_with_confidence:
                save_detection_images(frame, detections_with_confidence, current_frame_pos, leftup_info, leftdown_rust_info, rust_analysis)
                print(f"Rust detection images and analysis saved (Frame: {current_frame_pos}, Total: {save_counter} files)")
            else:
                print("No detection results to save.")
    elif key == ord(' '):  # スペースキー - 動画の一時停止/再生
        video_paused = not video_paused
        status = "paused" if video_paused else "playing"
        print(f"Video {status} (Frame: {current_frame_pos})")
    elif key == ord('r'):  # rキー - 動画を最初から再生
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current_frame_pos = 0
        video_paused = False
        leftup_frame_count = 0
        print("Video restarted from beginning")
    elif video_paused:  # 一時停止中のみフレーム移動可能
        if key == 81 or key == 2:  # 左矢印キー - 前のフレーム
            new_pos = max(0, current_frame_pos - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            if ret:
                current_frame_pos = new_pos
                print(f"Previous frame: {current_frame_pos}")
        elif key == 83 or key == 3:  # 右矢印キー - 次のフレーム
            new_pos = min(total_frames - 1, current_frame_pos + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            if ret:
                current_frame_pos = new_pos
                print(f"Next frame: {current_frame_pos}")
    
    # 検出部分の一時停止制御（従来の機能を維持）
    if key == ord('d'):  # dキー - 検出部分の一時停止/再生
        if leftup_paused:
            # 検出一時停止中の場合：現在のフレームまで飛ばす
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
            print("Detection display resumed")
        else:
            # 検出再生中の場合：一時停止
            leftup_paused = True
            print("Detection display paused")

cap.release()
cv2.destroyAllWindows()
print("Rust detection analysis completed.")
print(f"Total saved images: {save_counter} files")
print(f"Output folder: {session_folder}")
