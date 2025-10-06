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
import sys
import unicodedata

# QRコード履歴とトラッキング用辞書
qr_history = []
detected_qr_codes = {}
qr_manager = {}
qr_counter = 1  # システム起動時に1から開始
MAX_HISTORY = 10
TIMEOUT = 1.0
CSV_FILE = "qr_history.csv"

# QRコード内容表示用
qr_contents = {}  # QRコードの内容を保存
current_display_qr = None  # 現在表示中のQRコード
current_history_index = 0  # 現在の履歴インデックス（1キーで順番切り替え用）

# Mac環境での文字化け対策関数
def process_text_for_mac(text):
    """Mac環境での文字化け対策（完全対応）"""
    if not text:
        return ""
    
    try:
        # 文字列を安全に処理
        if isinstance(text, bytes):
            # 複数のエンコーディングを試す
            encodings = ['utf-8', 'utf-8-sig', 'shift_jis', 'cp932', 'euc-jp', 'iso-2022-jp', 'latin1', 'mac-roman']
            for encoding in encodings:
                try:
                    text = text.decode(encoding)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                text = text.decode('utf-8', errors='replace')
        
        # 文字列を正規化
        safe_text = str(text)
        
        # Mac環境での特殊処理
        if sys.platform == 'darwin':
            # Unicode正規化（Mac標準）
            safe_text = unicodedata.normalize('NFC', safe_text)
            
            # 特殊文字の除去
            safe_text = safe_text.replace('\ufeff', '')  # BOM除去
            safe_text = safe_text.replace('\u200b', '')  # ゼロ幅スペース除去
            safe_text = safe_text.replace('\u200c', '')  # ゼロ幅非結合子除去
            safe_text = safe_text.replace('\u200d', '')  # ゼロ幅結合子除去
            safe_text = safe_text.replace('\u2060', '')  # 単語結合子除去
        
        # 不正な文字を除去
        safe_text = ''.join(char for char in safe_text if ord(char) < 0x110000 and ord(char) != 0xFFFD)
        
        # 最終的な安全なエンコード/デコード
        safe_text = safe_text.encode('utf-8', errors='replace').decode('utf-8')
        
        return safe_text
        
    except Exception as e:
        print(f"Mac文字処理エラー: {e}")
        return str(text) if text else ""

# フォント設定（Mac/Windows対応）
if sys.platform == 'darwin':  # Mac環境
    FONT_PATH_JAPANESE = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    FONT_PATH_ENGLISH = "/System/Library/Fonts/Helvetica.ttc"
else:  # Windows環境
    FONT_PATH_JAPANESE = "C:/Windows/Fonts/meiryo.ttc"
    FONT_PATH_ENGLISH = "C:/Windows/Fonts/arial.ttf"

# --- 保存用のカウンター ---
save_counter = 0
pause_counter = 0  # 一時停止した回数
image_save_counter = 0  # 画像を保存した回数
auto_save_flag = False  # 自動保存フラグ
pause_sequence_counter = 0  # pauseごとの連番カウンター

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

# 画面を新しいレイアウトに設定
WINDOW_WIDTH = screen_width
WINDOW_HEIGHT = screen_height

# 左側：認識中の動画（画面の左半分）
LEFT_VIDEO_WIDTH = WINDOW_WIDTH // 2
LEFT_VIDEO_HEIGHT = WINDOW_HEIGHT

# 右側：内容プレビュー（画面の右半分）
RIGHT_PREVIEW_WIDTH = WINDOW_WIDTH // 2
RIGHT_PREVIEW_HEIGHT = WINDOW_HEIGHT

# 履歴表示エリア（左側の下部）
HISTORY_AREA_HEIGHT = int(LEFT_VIDEO_HEIGHT * 0.3)  # 左側の30%の高さ
VIDEO_AREA_HEIGHT = LEFT_VIDEO_HEIGHT - HISTORY_AREA_HEIGHT  # 残りの高さを動画に

# 見やすさ向上のためのパラメータ調整
HISTORY_FONT_SIZE = 14  # 文字サイズ
HISTORY_ENTRY_SPACING = 40  # 履歴間の間隔
CONTENT_FONT_SIZE = 24  # QRコード内容表示の文字サイズ（右側エリアを活用してさらに大きく）
CONTENT_LINE_SPACING = 50  # 内容表示の行間隔（重複完全防止）

def reset_csv_counter():
    """CSVファイルの通し番号を1から開始し、セッションフォルダーに新しいCSVファイルを作成する"""
    global qr_counter, CSV_FILE
    qr_counter = 1
    
    # セッションフォルダー内に新しいCSVファイルを作成
    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    session_csv_filename = f"qr_history_{time_str}.csv"
    CSV_FILE = os.path.join(session_folder, session_csv_filename)
    
    # 新しいCSVファイルにヘッダー行を書き込み
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['QR番号', 'QRコード内容', '検出時刻', 'プレビュー保存済み'])
    
    print(f"新しいCSVファイルを作成しました: {session_csv_filename}")
    print("通し番号を1から開始します")

def save_qr_to_csv(qr_number, qr_data, timestamp, preview_saved=False):
    """QRコードのデータをCSVファイルに保存する（完全な内容）"""
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Mac環境での文字化け対策を適用して完全な内容を保存
        processed_qr_data = process_text_for_mac(qr_data)
        writer.writerow([qr_number, processed_qr_data, timestamp, 'Yes' if preview_saved else 'No'])

def is_qr_content_saved_in_csv(qr_data):
    """CSVファイルでQRコード内容が既に保存されているかチェック"""
    if not os.path.exists(CSV_FILE):
        return False
    
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            # ヘッダー行をスキップ
            next(reader, None)
            
            processed_qr_data = process_text_for_mac(qr_data)
            for row in reader:
                if len(row) >= 2:
                    csv_content = row[1]  # QRコード内容の列
                    # 文字列の完全一致でチェック
                    if csv_content.strip() == processed_qr_data.strip():
                        return True
        return False
    except Exception as e:
        print(f"CSV重複チェックエラー: {e}")
        return False

def is_qr_preview_saved_in_csv(qr_data):
    """CSVファイルでQRコードの内容プレビューが既に保存されているかチェック"""
    if not os.path.exists(CSV_FILE):
        print(f"CSVファイルが存在しません: {CSV_FILE}")
        return False
    
    try:
        processed_qr_data = process_text_for_mac(qr_data)
        print(f"プレビュー保存チェック: 検索対象='{processed_qr_data}'")
        
        with open(CSV_FILE, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            # ヘッダー行をスキップ
            next(reader, None)
            
            for row_num, row in enumerate(reader, start=2):  # 行番号（ヘッダーを除く）
                if len(row) >= 4:  # プレビュー保存済み列も含む
                    csv_content = row[1]  # QRコード内容の列
                    preview_saved = row[3] if len(row) > 3 else 'No'  # プレビュー保存済み列
                    # 文字列の完全一致でチェック
                    if csv_content.strip() == processed_qr_data.strip() and preview_saved == 'Yes':
                        print(f"  行{row_num}: プレビュー保存済みを発見 - '{csv_content}' (プレビュー保存状態: {preview_saved})")
                        return True
                    elif csv_content.strip() == processed_qr_data.strip():
                        print(f"  行{row_num}: QRコード内容は一致するがプレビュー未保存 - '{csv_content}' (プレビュー保存状態: {preview_saved})")
        
        print(f"  プレビュー保存済みのQRコードが見つかりませんでした")
        return False
    except Exception as e:
        print(f"プレビュー保存チェックエラー: {e}")
        return False

def update_preview_saved_status(qr_data, saved=True):
    """CSVファイルでQRコードのプレビュー保存状態を更新"""
    if not os.path.exists(CSV_FILE):
        print(f"CSVファイルが存在しません: {CSV_FILE}")
        return False
    
    try:
        # CSVファイルを読み込み
        rows = []
        with open(CSV_FILE, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
        
        # 対象のQRコードの行を見つけて更新
        processed_qr_data = process_text_for_mac(qr_data)
        print(f"プレビュー保存状態更新: 対象='{processed_qr_data}', saved={saved}")
        
        updated = False
        for i, row in enumerate(rows):
            if i == 0:  # ヘッダー行はスキップ
                continue
            if len(row) >= 2 and row[1].strip() == processed_qr_data.strip():
                # プレビュー保存済み列を更新
                while len(row) < 4:
                    row.append('No')
                old_status = row[3] if len(row) > 3 else 'No'
                row[3] = 'Yes' if saved else 'No'
                updated = True
                print(f"  行{i+1}: プレビュー保存状態を更新 '{old_status}' → '{row[3]}'")
                break
        
        # 更新された内容をCSVファイルに書き戻し
        if updated:
            with open(CSV_FILE, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
            print(f"  CSVファイルを更新しました")
            return True
        else:
            print(f"  更新対象のQRコードが見つかりませんでした")
            return False
    except Exception as e:
        print(f"プレビュー保存状態更新エラー: {e}")
        return False

def open_url_in_browser(url):
    """URLをブラウザで開く"""
    try:
        webbrowser.open(url)
        print(f"URLをブラウザで開きました: {url}")
        return True
    except Exception as e:
        print(f"URLを開くのに失敗しました: {e}")
        return False


def get_qr_content_info(qr_data):
    """QRコードの内容から情報を取得"""
    content_info = {
        'type': 'text',
        'content': qr_data,
        'is_url': False,
        'is_image_url': False,
        'preview_image': None,
        'csv_info': None
    }
    
    # CSV情報を取得（常に最新の情報を取得）
    if qr_data in qr_manager:
        qr_number = qr_manager[qr_data]
        # CSVファイルから該当するQRコードの情報を取得
        try:
            if os.path.exists(CSV_FILE):
                with open(CSV_FILE, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    # 最新の情報を取得（最後に見つかった行）
                    latest_csv_info = None
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) >= 3 and parts[0] == str(qr_number):
                            latest_csv_info = {
                                'number': parts[0],
                                'timestamp': parts[2] if len(parts) > 2 else 'Unknown',
                                'content': parts[1] if len(parts) > 1 else qr_data  # CSVに保存された内容
                            }
                    if latest_csv_info:
                        content_info['csv_info'] = latest_csv_info
                        # QRコードから読み取った内容を表示用の内容として使用（CSVの内容は使用しない）
        except Exception as e:
            print(f"CSV情報取得エラー: {e}")
    
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

def save_qr_detection_images(frame, detected_qr_positions, frame_count, output_frame=None, increment_pause=True, force_save_preview=False):
    """QRコード検出画像を保存する関数（randoruto-wrs2.pyのファイル命名規則に準拠）"""
    global save_counter, image_save_counter, pause_counter, pause_sequence_counter
    
    # pause_counterを増加（自動保存時は増やさない）
    if increment_pause:
        pause_counter += 1
    
    # pause_sequence_counterは常に増加（通し番号のため）
    pause_sequence_counter += 1
    
    # ファイル名を生成（randoruto-wrs2.pyの命名規則に準拠）
    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    detection_count = len(detected_qr_positions)
    base_filename = f"capture_{pause_sequence_counter:06d}_{time_str}_frame{frame_count}_{detection_count}qr_codes"
    
    # 1. 左側動画画面を保存
    if output_frame is not None:
        # 左側の動画画面を切り抜き
        left_video_image = output_frame[0:VIDEO_AREA_HEIGHT, 0:LEFT_VIDEO_WIDTH]
        left_filepath = os.path.join(session_folder, f"{base_filename}_M_left_video.jpg")
        success = cv2.imwrite(left_filepath, left_video_image)
        if success:
            print(f"左側動画画面を保存しました: {base_filename}_M_left_video.jpg")
            save_counter += 1
            image_save_counter += 1
        else:
            print(f"エラー: 左側動画画面の保存に失敗しました - {left_filepath}")
    
    # 2. 内容プレビュー画面を保存（CSVベースの重複防止 + 強制保存オプション）
    if output_frame is not None:
        # 現在表示中のQRコードの内容プレビューが未保存の場合、または強制保存の場合のみ保存
        preview_already_saved = is_qr_preview_saved_in_csv(current_display_qr) if current_display_qr else False
        should_save_preview = (
            current_display_qr and 
            (not preview_already_saved or force_save_preview)
        )
        
        print(f"プレビュー保存判定: current_display_qr={current_display_qr}")
        print(f"  - preview_already_saved={preview_already_saved}")
        print(f"  - force_save_preview={force_save_preview}")
        print(f"  - should_save_preview={should_save_preview}")
        
        if should_save_preview:
            # 右側の内容プレビュー画面を切り抜き
            right_preview_image = output_frame[0:RIGHT_PREVIEW_HEIGHT, LEFT_VIDEO_WIDTH:WINDOW_WIDTH]
            right_filepath = os.path.join(session_folder, f"{base_filename}_A_content_preview.jpg")
            success = cv2.imwrite(right_filepath, right_preview_image)
            if success:
                print(f"内容プレビュー画面を保存しました: {base_filename}_A_content_preview.jpg")
                save_counter += 1
                image_save_counter += 1
                # CSVファイルでプレビュー保存状態を更新
                update_preview_saved_status(current_display_qr, saved=True)
            else:
                print(f"エラー: 内容プレビュー画面の保存に失敗しました - {right_filepath}")
        else:
            if is_qr_preview_saved_in_csv(current_display_qr):
                print("内容プレビュー画面は既に保存済みです（CSV確認済み）")
            else:
                print("保存する内容プレビューがありません")
    

def display_qr_history():
    """プログラム終了時に重複を除いたQRコードの履歴を表示する"""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, encoding='utf-8')
        df = df.drop_duplicates(subset=[df.columns[0]])  # 管理ナンバーで重複を削除
        print("\n==== QRコード履歴 ====")
        print(df.to_string(index=False))

def get_appropriate_font(size, text=""):
    """テキストに適したフォントを取得（Mac完全対応）"""
    try:
        # Mac環境での完全なフォント対応
        if sys.platform == 'darwin':  # Mac環境
            # Mac環境でのフォントパス設定（優先順位付き）
            mac_japanese_fonts = [
                "/System/Library/Fonts/Hiragino Sans GB.ttc",  # Mac標準日本語フォント（最優先）
                "/System/Library/Fonts/Hiragino Sans.ttc",     # Mac標準日本語フォント
                "/System/Library/Fonts/Hiragino Kaku Gothic ProN.ttc",  # Mac日本語フォント
                "/System/Library/Fonts/Hiragino Mincho ProN.ttc",       # Mac日本語フォント
                "/System/Library/Fonts/Yu Gothic Medium.otf",           # Mac日本語フォント
                "/System/Library/Fonts/Yu Gothic Bold.otf",             # Mac日本語フォント
                "/System/Library/Fonts/Arial Unicode MS.ttf",  # Mac Unicodeフォント
                "/Library/Fonts/Arial Unicode MS.ttf",         # Mac Unicodeフォント
            ]
            
            mac_english_fonts = [
                "/System/Library/Fonts/Helvetica.ttc",         # Mac標準フォント
                "/System/Library/Fonts/Arial.ttf",             # Mac Arialフォント
                "/System/Library/Fonts/Times.ttc",             # Mac Timesフォント
            ]
        else:
            mac_japanese_fonts = []
            mac_english_fonts = []
        
        # 日本語文字が含まれているかチェック（完全対応）
        has_japanese = any(
            '\u3040' <= char <= '\u309F' or  # ひらがな
            '\u30A0' <= char <= '\u30FF' or  # カタカナ
            '\u4E00' <= char <= '\u9FAF' or  # 漢字
            '\uFF00' <= char <= '\uFFEF' or  # 全角文字
            '\u3000' <= char <= '\u303F' or  # CJK記号・句読点
            '\u3400' <= char <= '\u4DBF' or  # CJK拡張A
            '\u20000' <= char <= '\u2A6DF'   # CJK拡張B
            for char in text
        )
        
        # 特殊記号や絵文字が含まれているかチェック
        has_special_chars = any(
            ord(char) > 127 or  # ASCII以外の文字
            char in '！？。、；：""''（）【】《》〈〉「」『』〔〕｛｝'  # 日本語記号
            for char in text
        )
        
        if has_japanese or has_special_chars:
            # Mac環境ではMac用日本語フォントを優先
            if sys.platform == 'darwin':
                for font_path in mac_japanese_fonts:
                    try:
                        return ImageFont.truetype(font_path, size)
                    except Exception as e:
                        print(f"Mac日本語フォント読み込み失敗: {font_path}, エラー: {e}")
                        continue
            
            # Windows環境またはMac用フォントが失敗した場合
            try:
                return ImageFont.truetype(FONT_PATH_JAPANESE, size)
            except:
                return ImageFont.load_default()
        else:
            # 英語フォントを試す
            if sys.platform == 'darwin':
                for font_path in mac_english_fonts:
                    try:
                        return ImageFont.truetype(font_path, size)
                    except:
                        continue
            
            try:
                return ImageFont.truetype(FONT_PATH_ENGLISH, size)
            except:
                try:
                    return ImageFont.truetype(FONT_PATH_JAPANESE, size)
                except:
                    return ImageFont.load_default()
    except Exception as e:
        print(f"フォント取得エラー: {e}")
        return ImageFont.load_default()

def draw_text_with_outline(img, text, position, font, text_color, outline_color=(0, 0, 0)):
    """文字を見やすくするために黒縁を適度に細くし、白字を適度に強調（Mac完全対応）"""
    try:
        # Mac環境での完全な文字化け対策
        safe_text = process_text_for_mac(text)
        
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        x, y = position
        
        # アウトラインを描画
        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in offsets:
            try:
                draw.text((x + dx, y + dy), safe_text, font=font, fill=outline_color)
            except Exception:
                # フォントエラーの場合はデフォルトフォントで再試行
                try:
                    default_font = ImageFont.load_default()
                    draw.text((x + dx, y + dy), safe_text, font=default_font, fill=outline_color)
                except:
                    pass
        
        # メインテキストを描画
        try:
            draw.text((x, y), safe_text, font=font, fill=text_color)
        except Exception:
            # フォントエラーの場合はデフォルトフォントで再試行
            try:
                default_font = ImageFont.load_default()
                draw.text((x, y), safe_text, font=default_font, fill=text_color)
            except:
                pass
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"文字描画エラー: {e}, テキスト: {repr(text)}")
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
                        qr_counter += 1  # 新しいQRコードの場合のみカウンターを増加
                    # QRコード内容情報を取得・保存（Mac対応）
                    processed_qr_data = process_text_for_mac(qr_data)
                    qr_contents[processed_qr_data] = get_qr_content_info(processed_qr_data)
                        
                        # URLの場合は自動的にブラウザで開かない（手動で'o'キーを押す必要がある）
                    
                    managed_data = f"[{qr_manager[qr_data]}] {qr_data}"
                    detected_qr_codes[managed_data] = time.time()
                    
                    # 最新検出のQRコードを自動表示
                    global current_display_qr, current_history_index
                    current_display_qr = qr_data
                    current_history_index = 0  # 新しいQRコード検出時は履歴インデックスをリセット
                    
                    # 新しいQRコードの場合は自動保存フラグを設定（一度だけ保存）
                    if is_new_qr:
                        # 自動保存フラグを設定
                        global auto_save_flag
                        auto_save_flag = True
                    
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
cap = cv2.VideoCapture(0)  # ← カメラ番号を変更する場合はここを編集してください（0: 内蔵カメラ、1: 外付けカメラ、2: OBS仮想カメラなど）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, LEFT_VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_AREA_HEIGHT)
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
print("  '1'キー: 履歴を順番に切り替え")
print("  '2'キー: 現在表示中のURLをブラウザで開く")
print("  '3-9'キー: 履歴から直接選択")
print("  'q'キー: 終了")

# システム起動時にCSVファイルの通し番号をリセット
reset_csv_counter()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ フレームの取得に失敗しました。")
        break

    frame_count += 1
    detected_qr_positions = detect_qr_code(frame)
    current_time = time.time()
    
    # 新しいレイアウト画面を作成
    output_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    
    # 左側：認識中の動画
    resized_frame = cv2.resize(frame, (LEFT_VIDEO_WIDTH, VIDEO_AREA_HEIGHT))
    output_frame[0:VIDEO_AREA_HEIGHT, 0:LEFT_VIDEO_WIDTH] = resized_frame
    
    # 検出されたQRコードに枠を描画
    for qr_data, rect in detected_qr_positions.items():
        x = int(rect.left * LEFT_VIDEO_WIDTH / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        y = int(rect.top * VIDEO_AREA_HEIGHT / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(rect.width * LEFT_VIDEO_WIDTH / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(rect.height * VIDEO_AREA_HEIGHT / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        # 日本語対応のテキスト描画
        try:
            # テキストを安全に処理
            safe_text = str(qr_data).encode('utf-8', errors='replace').decode('utf-8')
            # 長いテキストは短縮
            display_text = safe_text[:30] + "..." if len(safe_text) > 30 else safe_text
            
            # 日本語フォントを使用してテキストを描画
            font = get_appropriate_font(16, display_text)
            output_frame = draw_text_with_outline(output_frame, display_text, (x, y - 10), font, (0, 255, 0))
        except Exception as e:
            # エラーの場合はOpenCVのデフォルトフォントを使用
            cv2.putText(output_frame, str(qr_data)[:30], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 左側下部：履歴表示エリア
    cv2.rectangle(output_frame, (0, VIDEO_AREA_HEIGHT), (LEFT_VIDEO_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
    
    # カウンター情報を表示
    counter_y = VIDEO_AREA_HEIGHT + 20
    counter_font = get_appropriate_font(HISTORY_FONT_SIZE, "Pause Count")
    output_frame = draw_text_with_outline(output_frame, f"Pause Count: {pause_counter}", (20, counter_y), counter_font, (255, 255, 255))
    counter_y += 30
    output_frame = draw_text_with_outline(output_frame, f"Image Saves: {image_save_counter}", (20, counter_y), counter_font, (255, 255, 255))
    counter_y += 50
    
    # QRコード履歴を表示（最初の7文字のみ）
    y_offset = counter_y
    for i, qr_data in enumerate(reversed(qr_history[-MAX_HISTORY:])):
        text_color = (255, 255, 255)
        if qr_data in detected_qr_codes and (current_time - detected_qr_codes[qr_data]) < TIMEOUT:
            text_color = (255, 0, 0)  # 赤色：新しく検出されたQRコード
        elif i == current_history_index:
            text_color = (0, 255, 255)  # シアン色：現在選択中のQRコード
        
        # 管理番号部分を除去して元のQRコードデータを取得
        original_qr_data = qr_data.split('] ', 1)[1] if '] ' in qr_data else qr_data
        
        # Mac環境での文字化け対策
        processed_original_data = process_text_for_mac(original_qr_data)
        
        # 最初の7文字のみを表示
        short_content = processed_original_data[:7] + "..." if len(processed_original_data) > 7 else processed_original_data
        
        # 番号付きで表示
        display_text = f"{i+1}. {short_content}"
        history_font = get_appropriate_font(HISTORY_FONT_SIZE, display_text)
        output_frame = draw_text_with_outline(output_frame, display_text, (20, y_offset), history_font, text_color)
        y_offset += HISTORY_ENTRY_SPACING
    
    # 右側：内容プレビューエリア（大きく）
    cv2.rectangle(output_frame, (LEFT_VIDEO_WIDTH, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (40, 40, 40), -1)
    
    # 最新検出のQRコードを自動表示（選択されていない場合）
    if not current_display_qr and qr_history:
        latest_qr = qr_history[-1]
        latest_original_qr_data = latest_qr.split('] ', 1)[1] if '] ' in latest_qr else latest_qr
        current_display_qr = latest_original_qr_data
    
    if current_display_qr and current_display_qr in qr_contents:
        content_info = qr_contents[current_display_qr]
        content_y = 20
        
        # 簡潔なタイトルのみ表示
        title_text = f"QR#{qr_manager.get(current_display_qr, '?')}"
        title_font = get_appropriate_font(CONTENT_FONT_SIZE + 2, title_text)
        output_frame = draw_text_with_outline(output_frame, title_text, (LEFT_VIDEO_WIDTH + 20, content_y), title_font, (255, 255, 255))
        content_y += CONTENT_LINE_SPACING + 15
        
        # 内容テキスト（右側全体を活用してフル表示、vCard/JSON対応、Mac対応）
        content_text = process_text_for_mac(content_info['content'])
        
        # vCardやJSONの場合は改行を考慮した処理
        if content_text.startswith('BEGIN:VCARD') or content_text.startswith('{') or content_text.startswith('['):
            # vCardやJSONの場合は既存の改行を保持
            lines = content_text.split('\n')
            # 長すぎる行は分割
            processed_lines = []
            for line in lines:
                if len(line) > 40:
                    # 長い行を分割
                    for i in range(0, len(line), 40):
                        processed_lines.append(line[i:i+40])
                else:
                    processed_lines.append(line)
            lines = processed_lines
        else:
            # 通常のテキストの場合は文字数で分割
            max_chars_per_line = 40
            lines = [content_text[i:i+max_chars_per_line] for i in range(0, len(content_text), max_chars_per_line)]
        
        # フル表示（省略なし、vCard/JSON対応）
        for line in lines:
            if content_y < WINDOW_HEIGHT - 80:  # 画面からはみ出さないように制限（余裕を持たせる）
                line_font = get_appropriate_font(CONTENT_FONT_SIZE, line)
                output_frame = draw_text_with_outline(output_frame, line, (LEFT_VIDEO_WIDTH + 20, content_y), line_font, (255, 255, 255))
                content_y += CONTENT_LINE_SPACING + 5  # vCard/JSONの複雑な構造に対応して余裕を持たせる
            else:
                break
        
        # 右側全体をテキスト表示エリアとして使用（内容プレビューのみ）
        # 画像プレビューは表示せず、テキストのみに集中
    
    elif current_display_qr:
        # QRコード内容が取得できていない場合
        content_y = 20
        fallback_title_font = get_appropriate_font(CONTENT_FONT_SIZE + 2, "QRコード内容")
        output_frame = draw_text_with_outline(output_frame, "QRコード内容:", (LEFT_VIDEO_WIDTH + 20, content_y), fallback_title_font, (255, 255, 255))
        content_y += CONTENT_LINE_SPACING + 15
        fallback_loading_font = get_appropriate_font(CONTENT_FONT_SIZE, "内容を取得中")
        output_frame = draw_text_with_outline(output_frame, "内容を取得中...", (LEFT_VIDEO_WIDTH + 20, content_y), fallback_loading_font, (255, 255, 0))
        content_y += CONTENT_LINE_SPACING + 15
        
        # 内容を複数行に分割して表示（vCard/JSON対応、Mac対応）
        fallback_content = process_text_for_mac(current_display_qr)
        if len(fallback_content) > 40:
            fallback_lines = [fallback_content[i:i+40] for i in range(0, len(fallback_content), 40)]
            for line in fallback_lines:
                if content_y < WINDOW_HEIGHT - 80:
                    fallback_content_font = get_appropriate_font(CONTENT_FONT_SIZE, line)
                    output_frame = draw_text_with_outline(output_frame, line, (LEFT_VIDEO_WIDTH + 20, content_y), fallback_content_font, (255, 255, 255))
                    content_y += CONTENT_LINE_SPACING + 5
                else:
                    break
        else:
            fallback_content_font = get_appropriate_font(CONTENT_FONT_SIZE, fallback_content)
            output_frame = draw_text_with_outline(output_frame, fallback_content, (LEFT_VIDEO_WIDTH + 20, content_y), fallback_content_font, (255, 255, 255))
    
    cv2.imshow("QRコードトラッキング", output_frame)
    
    # プレビュー画像を保存（新しいQRコード検出時）
    
    # 自動保存（QRコード検出時、pause_counterを増やさない）
    if auto_save_flag and detected_qr_positions:
        print(f"自動保存を実行します: current_display_qr={current_display_qr}")
        save_qr_detection_images(frame, detected_qr_positions, frame_count, output_frame, increment_pause=False, force_save_preview=False)
        auto_save_flag = False  # フラグをリセット
        print(f"QRコード検出により自動保存しました (Total: {save_counter} files)")

    # キー操作の処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):  # sキー：QRコード検出画像を保存
        save_qr_detection_images(frame, detected_qr_positions, frame_count, output_frame, force_save_preview=True)
        pause_counter += 1
        print(f"QRコード検出画像を保存しました (Total: {save_counter} files)")
    elif key == ord('1'):  # 1キー：履歴を順番に切り替え
        if qr_history:
            # 履歴を順番に切り替え（最新から古い順）
            current_history_index = (current_history_index + 1) % len(qr_history)
            selected_qr = qr_history[-(current_history_index + 1)]
            # 管理番号部分を除去して元のQRコードデータを取得
            original_qr_data = selected_qr.split('] ', 1)[1] if '] ' in selected_qr else selected_qr
            current_display_qr = original_qr_data
            # 履歴切り替え時は内容プレビューの保存済みフラグをリセット（再保存可能にする）
            # CSVベースの管理では、強制保存オプションを使用して再保存を可能にする
            print(f"履歴切り替え ({current_history_index + 1}/{len(qr_history)}): {selected_qr}")
        else:
            print("履歴がありません")
    elif key == ord('2'):  # 2キー：現在表示中のURLをブラウザで開く
        if current_display_qr and current_display_qr.startswith(('http://', 'https://')):
            open_url_in_browser(current_display_qr)
        else:
            print("現在表示中のQRコードはURLではありません。")
    elif key >= ord('3') and key <= ord('9'):  # 3-9キー：履歴から直接選択
        selected_index = key - ord('3')  # 3キーは0番目、4キーは1番目...
        if 0 <= selected_index < len(qr_history):
            selected_qr = qr_history[-(selected_index+1)]
            # 管理番号部分を除去して元のQRコードデータを取得
            original_qr_data = selected_qr.split('] ', 1)[1] if '] ' in selected_qr else selected_qr
            current_display_qr = original_qr_data
            current_history_index = selected_index  # インデックスを更新
            # 直接選択時も内容プレビューの保存済みフラグをリセット（再保存可能にする）
            # CSVベースの管理では、強制保存オプションを使用して再保存を可能にする
            print(f"直接選択 ({selected_index + 1}/{len(qr_history)}): {selected_qr}")
    elif key == ord('q') or key == ord('Q'):  # qキー：終了
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 プログラムを終了しました。")
display_qr_history()
