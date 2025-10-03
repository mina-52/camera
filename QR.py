import cv2 
import time
import numpy as np
import csv
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import requests
import webbrowser

# QRコード履歴とトラッキング用辞書
qr_history = []
detected_qr_codes = {}
qr_manager = {}
qr_counter = 1
MAX_HISTORY = 10
TIMEOUT = 1.0
CSV_FILE = "qr_history.csv"

# QRコード内容表示用
qr_contents = {}  # QRコードの内容を保存
current_display_qr = None  # 現在表示中のQRコード

# フォント設定（日本語・英語対応）
FONT_PATH_JAPANESE = "C:/Windows/Fonts/meiryo.ttc"
FONT_PATH_ENGLISH = "C:/Windows/Fonts/arial.ttf"

# --- 保存用のカウンター ---
save_counter = 0
pause_counter = 0  # 一時停止した回数
image_save_counter = 0  # 画像を保存した回数
save_preview_flag = False  # プレビュー画像保存フラグ

# スクリーンサイズ取得
try:
    import tkinter as tk
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
except Exception:
    screen_width = 1920
    screen_height = 1080

# 画面を三分割に設定
WINDOW_WIDTH = screen_width
WINDOW_HEIGHT = screen_height

# 上部：認識中の画像（画面の上半分）
TOP_HEIGHT = WINDOW_HEIGHT // 2
TOP_WIDTH = WINDOW_WIDTH

# 下部：履歴とQRコード内容（画面の下半分を左右分割）
BOTTOM_HEIGHT = WINDOW_HEIGHT // 2
LEFT_WIDTH = WINDOW_WIDTH // 2  # 左側：履歴
RIGHT_WIDTH = WINDOW_WIDTH // 2  # 右側：QRコード内容

# 右下プレビューエリアのサイズ（少し大きく）
PREVIEW_WIDTH = int(RIGHT_WIDTH * 0.6)  # 右側の60%の幅
PREVIEW_HEIGHT = int(BOTTOM_HEIGHT * 0.6)  # 下部の60%の高さ

# 見やすさ向上のためのパラメータ調整
HISTORY_FONT_SIZE = 14  # 文字サイズ
HISTORY_ENTRY_SPACING = 40  # 履歴間の間隔
CONTENT_FONT_SIZE = 12  # QRコード内容表示の文字サイズ

def save_qr_to_csv(qr_number, qr_data, timestamp):
    """QRコードのデータをCSVファイルに保存する"""
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([qr_number, qr_data, timestamp])

def open_url_in_browser(url):
    """URLをブラウザで開く"""
    try:
        webbrowser.open(url)
        print(f"URLをブラウザで開きました: {url}")
        return True
    except Exception as e:
        print(f"URLを開くのに失敗しました: {e}")
        return False

def save_preview_image(preview_frame, qr_data, frame_count, qr_number):
    """プレビューエリアの画像を保存"""
    global save_counter, image_save_counter
    
    # プレビューエリアの座標を計算
    preview_x = LEFT_WIDTH + RIGHT_WIDTH // 2 + 10
    preview_y = TOP_HEIGHT + BOTTOM_HEIGHT // 2 + 10
    
    # プレビューエリアを切り抜き
    preview_img = preview_frame[preview_y:preview_y + PREVIEW_HEIGHT, preview_x:preview_x + PREVIEW_WIDTH]
    
    if preview_img.size > 0:
        # ファイル名を生成
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{save_counter + 1:06d}_{time_str}_frame{frame_count}_preview_qr{qr_number}.jpg"
        filepath = os.path.join(session_folder, filename)
        
        # 画像を保存
        success = cv2.imwrite(filepath, preview_img)
        if success:
            print(f"プレビュー画像を保存しました: {filename}")
            save_counter += 1
            image_save_counter += 1
            return True
        else:
            print(f"エラー: プレビュー画像の保存に失敗しました - {filepath}")
            return False
    else:
        print("プレビューエリアが無効です")
        return False

def get_qr_content_info(qr_data):
    """QRコードの内容から情報を取得"""
    content_info = {
        'type': 'text',
        'content': qr_data,
        'is_url': False,
        'is_image_url': False,
        'preview_image': None
    }
    
    # URLかどうかチェック
    if qr_data.startswith(('http://', 'https://')):
        content_info['is_url'] = True
        content_info['type'] = 'url'
        
        # 画像URLかどうかチェック
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        if any(qr_data.lower().endswith(ext) for ext in image_extensions):
            content_info['is_image_url'] = True
            content_info['type'] = 'image_url'
            # 画像を読み込んでプレビュー用にリサイズ
            try:
                import requests
                response = requests.get(qr_data, timeout=5)
                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        # プレビュー用にリサイズ
                        preview_size = 200
                        h, w = img.shape[:2]
                        if h > w:
                            new_h = preview_size
                            new_w = int(w * preview_size / h)
                        else:
                            new_w = preview_size
                            new_h = int(h * preview_size / w)
                        content_info['preview_image'] = cv2.resize(img, (new_w, new_h))
            except Exception as e:
                print(f"画像読み込みエラー: {e}")
    
    return content_info

def save_qr_detection_images(frame, detected_qr_positions, frame_count):
    """QRコード検出画像を保存する関数"""
    global save_counter, image_save_counter
    
    # 1. 全体画像（検出されたQRコードに枠を付けた画像）を保存
    annotated_img = frame.copy()
    detection_count = len(detected_qr_positions)
    
    # 各検出されたQRコードに対して枠とラベルを追加
    for qr_data, rect in detected_qr_positions.items():
        # 管理番号部分を除去して元のQRコードデータを取得
        original_qr_data = qr_data.split('] ', 1)[1] if '] ' in qr_data else qr_data
        
        x = int(rect.left * TOP_WIDTH / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        y = int(rect.top * TOP_HEIGHT / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(rect.width * TOP_WIDTH / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(rect.height * TOP_HEIGHT / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # QRコードを囲む枠を描画
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # ラベルを追加
        cv2.putText(annotated_img, qr_data, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # ファイル名を生成（撮影順序で並ぶように調整）
    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{save_counter + 1:06d}_{time_str}_frame{frame_count}_{detection_count}qr_codes.jpg"
    filepath = os.path.join(session_folder, filename)
    
    # 画像を保存
    success = cv2.imwrite(filepath, annotated_img)
    if success:
        print(f"QRコード検出画像を保存しました: {filename} (検出QRコード数: {detection_count})")
        save_counter += 1
        image_save_counter += 1
    else:
        print(f"エラー: 画像の保存に失敗しました - {filepath}")
    
    # 2. QRコード内容の画像も保存（画像URLの場合）
    for qr_data, rect in detected_qr_positions.items():
        original_qr_data = qr_data.split('] ', 1)[1] if '] ' in qr_data else qr_data
        if original_qr_data in qr_contents:
            content_info = qr_contents[original_qr_data]
            if content_info['is_image_url'] and content_info['preview_image'] is not None:
                # QRコード内容画像を保存
                content_filename = f"capture_{save_counter:06d}_{time_str}_frame{frame_count}_qr_content_{qr_manager[original_qr_data]}.jpg"
                content_filepath = os.path.join(session_folder, content_filename)
                cv2.imwrite(content_filepath, content_info['preview_image'])
                print(f"QRコード内容画像を保存しました: {content_filename}")
                save_counter += 1
                image_save_counter += 1
    

def display_qr_history():
    """プログラム終了時に重複を除いたQRコードの履歴を表示する"""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, encoding='utf-8')
        df = df.drop_duplicates(subset=[df.columns[0]])  # 管理ナンバーで重複を削除
        print("\n==== QRコード履歴 ====")
        print(df.to_string(index=False))

def get_appropriate_font(size, text=""):
    """テキストに適したフォントを取得"""
    try:
        # 日本語文字が含まれているかチェック
        if any('\u3040' <= char <= '\u309F' or  # ひらがな
               '\u30A0' <= char <= '\u30FF' or  # カタカナ
               '\u4E00' <= char <= '\u9FAF' or  # 漢字
               '\uFF00' <= char <= '\uFFEF'     # 全角文字
               for char in text):
            return ImageFont.truetype(FONT_PATH_JAPANESE, size)
        else:
            return ImageFont.truetype(FONT_PATH_ENGLISH, size)
    except:
        # フォントが見つからない場合はデフォルトフォントを使用
        try:
            return ImageFont.truetype(FONT_PATH_JAPANESE, size)
        except:
            return ImageFont.load_default()

def draw_text_with_outline(img, text, position, font, text_color, outline_color=(0, 0, 0)):
    """文字を見やすくするために黒縁を適度に細くし、白字を適度に強調"""
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        x, y = position
        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in offsets:
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        draw.text((x, y), text, font=font, fill=text_color)
        draw.text((x + 0.5, y + 0.5), text, font=font, fill=text_color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"文字描画エラー: {e}")
        return img

def detect_qr_code(frame):
    global qr_counter
    qr_positions = {}
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # OpenCVのQRコード検出器を使用
    qr_detector = cv2.QRCodeDetector()
    
    try:
        # QRコードを検出
        retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(frame)
        
        if retval and decoded_info is not None:
            for i, qr_data in enumerate(decoded_info):
                if qr_data:  # 空でない場合
                    is_new_qr = qr_data not in qr_manager
                    if is_new_qr:
                        qr_manager[qr_data] = qr_counter
                        save_qr_to_csv(qr_counter, qr_data, current_time)
                        # QRコード内容情報を取得・保存
                        qr_contents[qr_data] = get_qr_content_info(qr_data)
                        qr_counter += 1
                        
                        # 新しいURLの場合は自動的にブラウザで開く
                        if qr_data.startswith(('http://', 'https://')):
                            open_url_in_browser(qr_data)
                    
                    managed_data = f"[{qr_manager[qr_data]}] {qr_data}"
                    detected_qr_codes[managed_data] = time.time()
                    
                    # 最新検出のQRコードを自動表示
                    global current_display_qr
                    current_display_qr = qr_data
                    
                    # 新しいQRコードの場合はプレビュー画像を保存
                    if is_new_qr:
                        # プレビュー画像を保存するためのフラグを設定
                        global save_preview_flag
                        save_preview_flag = True
                    
                    # 座標情報を取得（pointsから矩形を作成）
                    if points is not None and len(points) > i:
                        pts = points[i]
                        if pts is not None and len(pts) >= 4:
                            # 4点から矩形の境界を計算
                            x_coords = pts[:, 0]
                            y_coords = pts[:, 1]
                            left = int(min(x_coords))
                            top = int(min(y_coords))
                            right = int(max(x_coords))
                            bottom = int(max(y_coords))
                            
                            # 矩形オブジェクトを作成（pyzbarのrect形式に合わせる）
                            class Rect:
                                def __init__(self, left, top, width, height):
                                    self.left = left
                                    self.top = top
                                    self.width = width
                                    self.height = height
                            
                            rect = Rect(left, top, right - left, bottom - top)
                            qr_positions[managed_data] = rect
                    
                    if managed_data not in qr_history:
                        qr_history.append(managed_data)
                        if len(qr_history) > MAX_HISTORY:
                            qr_history.pop(0)
    
    except Exception as e:
        print(f"QRコード検出エラー: {e}")
    
    return qr_positions

# OBS仮想カメラの読み込み設定
cap = cv2.VideoCapture(1)  # 仮想カメラのインデックス。環境に合わせて変更してください
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TOP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TOP_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("エラー: OBS仮想カメラを開けませんでした。")
    print("OBSで「仮想カメラを開始」しているか、正しいカメラインデックスか確認してください。")
    exit()

# QRフォルダーの作成
save_folder = "qr_detection"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"フォルダーを作成しました: {save_folder}")

# 実行セッション用のフォルダーを作成（一回の実行につき一個）
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_folder = os.path.join(save_folder, f"session_{session_timestamp}")
os.makedirs(session_folder)
print(f"セッションフォルダーを作成しました: {session_folder}")

cv2.namedWindow("QRコードトラッキング", cv2.WINDOW_NORMAL)
cv2.resizeWindow("QRコードトラッキング", WINDOW_WIDTH, WINDOW_HEIGHT)

# フレームカウンター
frame_count = 0

print("QRコード検出を開始します。")
print("キー操作:")
print("  's'キー: QRコード検出画像を保存")
print("  '1-9'キー: 履歴からQRコードを選択して内容表示")
print("  'o'キー: 現在表示中のURLをブラウザで開く")
print("  'q'キー: 終了")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ フレームの取得に失敗しました。")
        break

    frame_count += 1
    detected_qr_positions = detect_qr_code(frame)
    current_time = time.time()
    
    # 三分割画面を作成
    output_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    
    # 上部：認識中の画像
    resized_frame = cv2.resize(frame, (TOP_WIDTH, TOP_HEIGHT))
    output_frame[0:TOP_HEIGHT, 0:TOP_WIDTH] = resized_frame
    
    # 検出されたQRコードに枠を描画
    for qr_data, rect in detected_qr_positions.items():
        x = int(rect.left * TOP_WIDTH / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        y = int(rect.top * TOP_HEIGHT / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(rect.width * TOP_WIDTH / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(rect.height * TOP_HEIGHT / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(output_frame, qr_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 下部左：履歴表示
    cv2.rectangle(output_frame, (0, TOP_HEIGHT), (LEFT_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
    
    # カウンター情報を表示
    counter_y = TOP_HEIGHT + 20
    counter_font = get_appropriate_font(HISTORY_FONT_SIZE, "Pause Count")
    output_frame = draw_text_with_outline(output_frame, f"Pause Count: {pause_counter}", (20, counter_y), counter_font, (255, 255, 255))
    counter_y += 30
    output_frame = draw_text_with_outline(output_frame, f"Image Saves: {image_save_counter}", (20, counter_y), counter_font, (255, 255, 255))
    counter_y += 50
    
    # QRコード履歴を表示
    y_offset = counter_y
    for i, qr_data in enumerate(reversed(qr_history[-MAX_HISTORY:])):
        text_color = (255, 255, 255)
        if qr_data in detected_qr_codes and (current_time - detected_qr_codes[qr_data]) < TIMEOUT:
            text_color = (255, 0, 0)
        
        # 番号付きで表示
        display_text = f"{i+1}. {qr_data}"
        history_font = get_appropriate_font(HISTORY_FONT_SIZE, display_text)
        output_frame = draw_text_with_outline(output_frame, display_text, (20, y_offset), history_font, text_color)
        y_offset += HISTORY_ENTRY_SPACING
    
    # 下部右：QRコード内容表示エリア（上部）
    cv2.rectangle(output_frame, (LEFT_WIDTH, TOP_HEIGHT), (WINDOW_WIDTH, TOP_HEIGHT + BOTTOM_HEIGHT // 2), (40, 40, 40), -1)
    
    # 右下：プレビューエリア
    cv2.rectangle(output_frame, (LEFT_WIDTH + RIGHT_WIDTH // 2, TOP_HEIGHT + BOTTOM_HEIGHT // 2), (WINDOW_WIDTH, WINDOW_HEIGHT), (60, 60, 60), -1)
    
    # 最新検出のQRコードを自動表示（選択されていない場合）
    if not current_display_qr and qr_history:
        latest_qr = qr_history[-1]
        latest_original_qr_data = latest_qr.split('] ', 1)[1] if '] ' in latest_qr else latest_qr
        current_display_qr = latest_original_qr_data
    
    if current_display_qr and current_display_qr in qr_contents:
        content_info = qr_contents[current_display_qr]
        content_y = TOP_HEIGHT + 20
        
        # タイトル
        title_font = get_appropriate_font(HISTORY_FONT_SIZE, "QRコード内容")
        output_frame = draw_text_with_outline(output_frame, "QRコード内容:", (LEFT_WIDTH + 20, content_y), title_font, (255, 255, 255))
        content_y += 40
        
        # 内容タイプ
        type_text = f"タイプ: {content_info['type']}"
        type_font = get_appropriate_font(HISTORY_FONT_SIZE, type_text)
        output_frame = draw_text_with_outline(output_frame, type_text, (LEFT_WIDTH + 20, content_y), type_font, (0, 255, 255))
        content_y += 30
        
        # URLかどうか
        if content_info['is_url']:
            url_text = "URL: はい"
            url_font = get_appropriate_font(HISTORY_FONT_SIZE, url_text)
            output_frame = draw_text_with_outline(output_frame, url_text, (LEFT_WIDTH + 20, content_y), url_font, (0, 255, 0))
            content_y += 30
        
        # 画像URLかどうか
        if content_info['is_image_url']:
            img_url_text = "画像URL: はい"
            img_font = get_appropriate_font(HISTORY_FONT_SIZE, img_url_text)
            output_frame = draw_text_with_outline(output_frame, img_url_text, (LEFT_WIDTH + 20, content_y), img_font, (255, 165, 0))
            content_y += 30
        
        # 内容テキスト（改行対応）
        content_text = content_info['content']
        max_chars_per_line = 25  # プレビューエリアに合わせて短く
        lines = [content_text[i:i+max_chars_per_line] for i in range(0, len(content_text), max_chars_per_line)]
        
        # テキスト表示エリアの制限（プレビューエリアの高さに合わせて）
        max_lines = 4  # 最大4行まで表示
        for line in lines[:max_lines]:
            line_font = get_appropriate_font(HISTORY_FONT_SIZE, line)
            output_frame = draw_text_with_outline(output_frame, line, (LEFT_WIDTH + 20, content_y), line_font, (255, 255, 255))
            content_y += 25
        
        # テキストが長い場合は省略表示
        if len(lines) > max_lines:
            remaining_lines = len(lines) - max_lines
            remaining_text = f"...他{remaining_lines}行"
            remaining_font = get_appropriate_font(HISTORY_FONT_SIZE, remaining_text)
            output_frame = draw_text_with_outline(output_frame, remaining_text, (LEFT_WIDTH + 20, content_y), remaining_font, (200, 200, 200))
            content_y += 25
        
        # 右下プレビューエリアで内容を表示
        preview_x = LEFT_WIDTH + RIGHT_WIDTH // 2 + 10
        preview_y = TOP_HEIGHT + BOTTOM_HEIGHT // 2 + 10
        
        if content_info['is_image_url'] and content_info['preview_image'] is not None:
            # 画像URLの場合は画像プレビュー
            preview_title_font = get_appropriate_font(HISTORY_FONT_SIZE, "画像プレビュー")
            output_frame = draw_text_with_outline(output_frame, "画像プレビュー:", (preview_x, preview_y), preview_title_font, (255, 255, 255))
            
            # 画像プレビューの処理
            preview_img = content_info['preview_image']
            h, w = preview_img.shape[:2]
            
            # プレビューエリアのサイズに合わせて画像をリサイズ
            max_img_height = PREVIEW_HEIGHT - 40
            max_img_width = PREVIEW_WIDTH - 20
            
            if h > max_img_height:
                scale = max_img_height / h
                new_h = int(h * scale)
                new_w = int(w * scale)
                preview_img = cv2.resize(preview_img, (new_w, new_h))
                h, w = new_h, new_w
            
            if w > max_img_width:
                scale = max_img_width / w
                new_h = int(h * scale)
                new_w = int(w * scale)
                preview_img = cv2.resize(preview_img, (new_w, new_h))
                h, w = new_h, new_w
            
            # プレビューエリアに画像を配置
            img_start_y = preview_y + 30
            if img_start_y + h < WINDOW_HEIGHT and preview_x + w < WINDOW_WIDTH:
                output_frame[img_start_y:img_start_y+h, preview_x:preview_x+w] = preview_img
                cv2.rectangle(output_frame, (preview_x, img_start_y), (preview_x + w, img_start_y + h), (255, 255, 255), 2)
        else:
            # テキスト内容の場合はテキストプレビュー
            content_preview_font = get_appropriate_font(HISTORY_FONT_SIZE, "内容プレビュー")
            output_frame = draw_text_with_outline(output_frame, "内容プレビュー:", (preview_x, preview_y), content_preview_font, (255, 255, 255))
            preview_y += 30
            
            # プレビューエリアにテキスト内容を表示
            preview_lines = [content_text[i:i+18] for i in range(0, len(content_text), 18)]  # 少し短くして表示
            max_preview_lines = 8  # プレビューエリアに表示できる最大行数
            
            for i, line in enumerate(preview_lines[:max_preview_lines]):
                if preview_y + i * 18 < WINDOW_HEIGHT - 20:
                    # テキストの色を内容タイプに応じて変更
                    text_color = (200, 255, 200)  # 通常テキスト
                    if content_info['is_url']:
                        text_color = (200, 200, 255)  # URLは青系
                    line_preview_font = get_appropriate_font(HISTORY_FONT_SIZE, line)
                    output_frame = draw_text_with_outline(output_frame, line, (preview_x, preview_y + i * 18), line_preview_font, text_color)
            
            # テキストが長い場合は省略表示
            if len(preview_lines) > max_preview_lines:
                remaining_preview_lines = len(preview_lines) - max_preview_lines
                remaining_preview_text = f"...他{remaining_preview_lines}行"
                remaining_preview_font = get_appropriate_font(HISTORY_FONT_SIZE, remaining_preview_text)
                output_frame = draw_text_with_outline(output_frame, remaining_preview_text, (preview_x, preview_y + max_preview_lines * 18), remaining_preview_font, (150, 150, 150))
    
    elif current_display_qr:
        # QRコード内容が取得できていない場合
        content_y = TOP_HEIGHT + 20
        fallback_title_font = get_appropriate_font(HISTORY_FONT_SIZE, "QRコード内容")
        output_frame = draw_text_with_outline(output_frame, "QRコード内容:", (LEFT_WIDTH + 20, content_y), fallback_title_font, (255, 255, 255))
        content_y += 40
        fallback_loading_font = get_appropriate_font(HISTORY_FONT_SIZE, "内容を取得中")
        output_frame = draw_text_with_outline(output_frame, "内容を取得中...", (LEFT_WIDTH + 20, content_y), fallback_loading_font, (255, 255, 0))
        content_y += 30
        fallback_content_font = get_appropriate_font(HISTORY_FONT_SIZE, current_display_qr)
        output_frame = draw_text_with_outline(output_frame, current_display_qr, (LEFT_WIDTH + 20, content_y), fallback_content_font, (255, 255, 255))
    
    cv2.imshow("QRコードトラッキング", output_frame)
    
    # プレビュー画像を保存（新しいQRコード検出時）
    if save_preview_flag and current_display_qr and current_display_qr in qr_manager:
        qr_number = qr_manager[current_display_qr]
        save_preview_image(output_frame, current_display_qr, frame_count, qr_number)
        save_preview_flag = False  # フラグをリセット

    # キー操作の処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):  # sキー：QRコード検出画像を保存
        save_qr_detection_images(frame, detected_qr_positions, frame_count)
        pause_counter += 1
        print(f"QRコード検出画像を保存しました (Total: {save_counter} files)")
    elif key >= ord('1') and key <= ord('9'):  # 1-9キー：履歴から選択
        selected_index = key - ord('1')
        if 0 <= selected_index < len(qr_history):
            selected_qr = qr_history[-(selected_index+1)]
            # 管理番号部分を除去して元のQRコードデータを取得
            original_qr_data = selected_qr.split('] ', 1)[1] if '] ' in selected_qr else selected_qr
            current_display_qr = original_qr_data
            print(f"選択されたQRコード: {selected_qr}")
    elif key == ord('o') or key == ord('O'):  # oキー：現在表示中のURLをブラウザで開く
        if current_display_qr and current_display_qr.startswith(('http://', 'https://')):
            open_url_in_browser(current_display_qr)
        else:
            print("現在表示中のQRコードはURLではありません。")
    elif key == ord('q') or key == ord('Q'):  # qキー：終了
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 プログラムを終了しました。")
display_qr_history()
