import cv2 
import time
import numpy as np
import csv
import os
import pandas as pd
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont

# QRコード履歴とトラッキング用辞書
qr_history = []
detected_qr_codes = {}
qr_manager = {}
qr_counter = 1
MAX_HISTORY = 6
TIMEOUT = 1.0
CSV_FILE = "qr_history.csv"

FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"

# 映像サイズ
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
HISTORY_WIDTH = 250
WINDOW_WIDTH = FRAME_WIDTH + HISTORY_WIDTH
WINDOW_HEIGHT = FRAME_HEIGHT

# 見やすさ向上のためのパラメータ調整
HISTORY_FONT_SIZE = 14  # 文字サイズ
HISTORY_ENTRY_SPACING = 120  # 履歴間の間隔

def save_qr_to_csv(qr_number, qr_data, timestamp):
    """QRコードのデータをCSVファイルに保存する"""
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([qr_number, qr_data, timestamp])
    

def display_qr_history():
    """プログラム終了時に重複を除いたQRコードの履歴を表示する"""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, encoding='utf-8')
        df = df.drop_duplicates(subset=[df.columns[0]])  # 管理ナンバーで重複を削除
        print("\n==== QRコード履歴 ====")
        print(df.to_string(index=False))

def draw_text_with_outline(img, text, position, font, text_color, outline_color=(0, 0, 0)):
    """文字を見やすくするために黒縁を適度に細くし、白字を適度に強調"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    x, y = position
    offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in offsets:
        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=text_color)
    draw.text((x + 0.5, y + 0.5), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def detect_qr_code(frame):
    global qr_counter
    qr_positions = {}
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    qr_codes = decode(frame)
    for qr in qr_codes:
        try:
            qr_data = qr.data.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            qr_data = "デコードエラー"
        
        if qr_data:
            if qr_data not in qr_manager:
                qr_manager[qr_data] = qr_counter
                save_qr_to_csv(qr_counter, qr_data, current_time)
                qr_counter += 1
            
            managed_data = f"[{qr_manager[qr_data]}] {qr_data}"
            detected_qr_codes[managed_data] = time.time()
            qr_positions[managed_data] = qr.rect
            
            if managed_data not in qr_history:
                qr_history.append(managed_data)
                if len(qr_history) > MAX_HISTORY:
                    qr_history.pop(0)
    
    return qr_positions

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("❌ カメラを開けませんでした。")
    exit()

cv2.namedWindow("QRコードトラッキング", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ フレームの取得に失敗しました。")
        break

    detected_qr_positions = detect_qr_code(frame)
    current_time = time.time()
    
    output_frame = np.zeros((FRAME_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    output_frame[:, HISTORY_WIDTH:HISTORY_WIDTH + FRAME_WIDTH] = resized_frame
    
    cv2.rectangle(output_frame, (0, 0), (HISTORY_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
    
    font = ImageFont.truetype(FONT_PATH, HISTORY_FONT_SIZE)
    
    y_offset = 20
    for qr_data in reversed(qr_history[-MAX_HISTORY:]):
        text_color = (255, 255, 255)
        if qr_data in detected_qr_codes and (current_time - detected_qr_codes[qr_data]) < TIMEOUT:
            text_color = (255, 0, 0)
        output_frame = draw_text_with_outline(output_frame, qr_data, (20, y_offset), font, text_color)
        y_offset += HISTORY_ENTRY_SPACING  

    for qr_data, rect in detected_qr_positions.items():
        x = int(rect.left * FRAME_WIDTH / cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + HISTORY_WIDTH
        y = int(rect.top * FRAME_HEIGHT / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(rect.width * FRAME_WIDTH / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(rect.height * FRAME_HEIGHT / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        output_frame = draw_text_with_outline(output_frame, qr_data, (x, y - 40), font, (0, 255, 0))

    cv2.imshow("QRコードトラッキング", output_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 プログラムを終了しました。")
display_qr_history()
