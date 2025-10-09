import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
import math
#kome

# ========================================
# 設定値（一括変更可能）
# ========================================
# ブラックランドルト環の閾値（明度判定）
BLACK_THRESHOLD = 80

# ホワイトランドルト環の閾値（明度判定）
WHITE_THRESHOLD = 115

# カメラ設定
CAMERA_INDEX = 0  # カメラのインデックス（通常は0）
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ランドルト環検出設定
LANDOLT_MATCH_RATIO_THRESHOLD = 0.8  # ランドルト環色一致率の閾値（80%以上）
LANDOLT_NUM_CIRCLES = 20  # 同心円の数
LANDOLT_MIN_GAP_SIZE = 5.0  # 最小ギャップサイズ（度）
LANDOLT_OVERLAY_ALPHA = 0.6  # オーバーレイの透明度

# ----------------------------------------------------
# ランドルト環の穴検出用関数
# ----------------------------------------------------
def rgb_to_hsv_opencv(rgb_array):
    """RGB配列をHSV配列に変換（OpenCV使用）"""
    # rgb_array is (H, W, 3) with values 0-255
    if rgb_array.max() <= 1.5:  # 0-1 range
        rgb_array = (rgb_array * 255).astype(np.uint8)
    
    # OpenCVのHSVは H: 0-179, S: 0-255, V: 0-255
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    # H値を0-359に変換
    hsv[:,:,0] = hsv[:,:,0] * 2
    
    return hsv

def analyze_image_brightness(img):
    """画像全体の明度を分析して黒背景か白背景かを判定"""
    # BGRをRGBに変換してからHSVに変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = rgb_to_hsv_opencv(img_rgb)
    
    # V値（明度）の平均を計算
    v_channel = hsv_img[:, :, 2]
    mean_brightness = np.mean(v_channel)
    
    # 明度の平均が128より大きければ白背景、小さければ黒背景
    is_white_background = mean_brightness > 128
    
    return is_white_background, mean_brightness

def check_is_target_color(hsv_pixel, brightness_threshold=None, is_white_landolt=False):
    """HSV値がターゲット色に近いかをチェック（ランドルト環の色判定）"""
    h, s, v = hsv_pixel
    
    if is_white_landolt:
        # 白いランドルト環の場合：白い部分がランドルト環、黒い部分が穴
        # 白の認識範囲を広くする（閾値を下げる）
        return v >= WHITE_THRESHOLD  # 明るい部分をランドルト環として認識
    else:
        # 黒いランドルト環の場合：黒い部分がランドルト環、白い部分が穴
        threshold = brightness_threshold if brightness_threshold is not None else BLACK_THRESHOLD
        return v <= threshold  # 暗い部分をランドルト環として認識

def bilinear_sample_gray(gray, xs, ys):
    """グレースケール画像での双線形補間サンプリング"""
    H, W = gray.shape
    x0 = np.clip(np.floor(xs).astype(int), 0, W-1)
    y0 = np.clip(np.floor(ys).astype(int), 0, H-1)
    x1 = np.clip(x0 + 1, 0, W-1)
    y1 = np.clip(y0 + 1, 0, H-1)
    
    wa = (x1 - xs) * (y1 - ys)
    wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0)
    wd = (xs - x0) * (ys - y0)
    
    Ia = gray[y0, x0]
    Ib = gray[y0, x1]
    Ic = gray[y1, x0]
    Id = gray[y1, x1]
    
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def detect_landolt_gaps(img, center_x, center_y, num_circles=None, min_match_ratio=None):
    """
    画像中心を基準としてランドルト環の穴を検出（ブラックとホワイトの両方を同時検出）
    
    Parameters:
    - img: 入力画像 (BGR)
    - center_x, center_y: 画像中心座標
    - num_circles: 同心円の数（Noneの場合は設定値を使用）
    - min_match_ratio: 円周上でのランドルト環色一致率の最小閾値（Noneの場合は設定値を使用）
    
    Returns:
    - dict: 検出結果（ブラックとホワイトの両方の情報を含む）
    """
    # 設定値を使用
    if num_circles is None:
        num_circles = LANDOLT_NUM_CIRCLES
    if min_match_ratio is None:
        min_match_ratio = LANDOLT_MATCH_RATIO_THRESHOLD
    
    # 画像全体の明度を分析
    is_white_background, mean_brightness = analyze_image_brightness(img)
    
    # BGRをRGBに変換してからHSVに変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = rgb_to_hsv_opencv(img_rgb)
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[:2]
    
    # 画像の最小サイズに基づいて最大半径を決定
    max_radius = min(W, H) // 3
    min_radius = max_radius * 0.1
    
    # 同心円の半径を計算（外側から内側へ、外側3つを除外）
    all_radii = np.linspace(max_radius * 0.9, min_radius, num_circles + 3)
    radii = all_radii[3:]  # 外側の3つを除外
    
    # ブラックランドルト環用の変数
    black_valid_circles = []
    black_all_circles = []
    black_gap_results = {}
    
    # ホワイトランドルト環用の変数
    white_valid_circles = []
    white_all_circles = []
    white_gap_results = {}
    
    for radius in radii:
        # 円周上のサンプル点を取得
        n_samples = max(360, int(2 * np.pi * radius))
        theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        
        # 円周上の座標計算
        x_coords = center_x + radius * np.cos(theta)
        y_coords = center_y + radius * np.sin(theta)
        
        # 画像境界内の点のみを取得
        valid_mask = ((x_coords >= 0) & (x_coords < W) & 
                     (y_coords >= 0) & (y_coords < H))
        
        if not valid_mask.any():
            continue
            
        # 有効な座標のみを使用
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        theta_valid = theta[valid_mask]
        
        # 整数座標に変換（最近傍）
        x_int = np.clip(np.round(x_valid).astype(int), 0, W-1)
        y_int = np.clip(np.round(y_valid).astype(int), 0, H-1)
        
        # HSV値を取得
        hsv_samples = hsv_img[y_int, x_int]
        
        # ブラックランドルト環の検出（黒い部分がランドルト環、白い部分が穴）
        black_matches = []
        black_gap_angles = []
        
        # ホワイトランドルト環の検出（白い部分がランドルト環、黒い部分が穴）
        white_matches = []
        white_gap_angles = []
        
        for i, hsv_pixel in enumerate(hsv_samples):
            # ブラックランドルト環の判定
            is_black_landolt = check_is_target_color(hsv_pixel, brightness_threshold=BLACK_THRESHOLD, is_white_landolt=False)
            black_matches.append(is_black_landolt)
            
            if not is_black_landolt:
                # 非ブラックランドルト環色の部分はギャップとして記録
                angle_deg = (theta_valid[i] * 180.0 / np.pi) % 360.0
                black_gap_angles.append(angle_deg)
            
            # ホワイトランドルト環の判定
            is_white_landolt = check_is_target_color(hsv_pixel, brightness_threshold=BLACK_THRESHOLD, is_white_landolt=True)
            white_matches.append(is_white_landolt)
            
            if not is_white_landolt:
                # 非ホワイトランドルト環色の部分はギャップとして記録
                angle_deg = (theta_valid[i] * 180.0 / np.pi) % 360.0
                white_gap_angles.append(angle_deg)
        
        # ブラックランドルト環色一致率を計算
        black_match_ratio = np.mean(black_matches) if black_matches else 0.0
        
        # ホワイトランドルト環色一致率を計算
        white_match_ratio = np.mean(white_matches) if white_matches else 0.0
        
        # ブラックランドルト環の円の情報を記録
        black_circle_info = {
            'radius': radius,
            'match_ratio': black_match_ratio,
            'total_samples': len(black_matches),
            'matched_samples': np.sum(black_matches),
            'type': 'black'
        }
        black_all_circles.append(black_circle_info)
        
        # ホワイトランドルト環の円の情報を記録
        white_circle_info = {
            'radius': radius,
            'match_ratio': white_match_ratio,
            'total_samples': len(white_matches),
            'matched_samples': np.sum(white_matches),
            'type': 'white'
        }
        white_all_circles.append(white_circle_info)
        
        # ランドルト環色が設定値以上の円のみをランドルト環として認識
        landolt_ratio_threshold = min_match_ratio
        
        # ブラックランドルト環の有効性チェック
        if black_match_ratio >= landolt_ratio_threshold:
            black_valid_circles.append(black_circle_info)
            
            # ブラックランドルト環のギャップの連続区間を検出
            if black_gap_angles:
                gap_intervals = detect_gap_intervals(black_gap_angles)
                if gap_intervals:
                    black_gap_results[radius] = gap_intervals
        
        # ホワイトランドルト環の有効性チェック
        if white_match_ratio >= landolt_ratio_threshold:
            white_valid_circles.append(white_circle_info)
            
            # ホワイトランドルト環のギャップの連続区間を検出
            if white_gap_angles:
                gap_intervals = detect_gap_intervals(white_gap_angles)
                if gap_intervals:
                    white_gap_results[radius] = gap_intervals
    
    # 連続してランドルト環が多く並んでいる方を選択
    black_circle_count = len(black_valid_circles)
    white_circle_count = len(white_valid_circles)
    
    # 連続性を考慮した選択ロジック
    def calculate_continuity_score(circles):
        """連続性スコアを計算（連続して並んでいるランドルト環の最大長）"""
        if not circles:
            return 0
        
        # 半径でソート
        sorted_circles = sorted(circles, key=lambda x: x['radius'])
        max_continuity = 1
        current_continuity = 1
        
        for i in range(1, len(sorted_circles)):
            # 半径の差が小さい場合は連続とみなす
            radius_diff = sorted_circles[i]['radius'] - sorted_circles[i-1]['radius']
            if radius_diff < (max_radius - min_radius) / num_circles * 2:  # 2つ分以内の差
                current_continuity += 1
            else:
                max_continuity = max(max_continuity, current_continuity)
                current_continuity = 1
        
        max_continuity = max(max_continuity, current_continuity)
        return max_continuity
    
    # 連続性スコアを計算
    black_continuity_score = calculate_continuity_score(black_valid_circles)
    white_continuity_score = calculate_continuity_score(white_valid_circles)
    
    # 選択されたランドルト環タイプを決定（連続性スコアが高い方を選択）
    selected_type = None
    if black_continuity_score > white_continuity_score:
        selected_type = 'black'
    elif white_continuity_score > black_continuity_score:
        selected_type = 'white'
    elif black_circle_count > white_circle_count:
        # 連続性スコアが同じ場合は、円の数が多い方を選択
        selected_type = 'black'
    elif white_circle_count > black_circle_count:
        selected_type = 'white'
    # それでも同じ場合は、どちらも選択しない（None）
    
    # 選択されたタイプに基づいて結果を返す
    if selected_type == 'black':
        return {
            'black_valid_circles': black_valid_circles,
            'black_all_circles': black_all_circles,
            'black_gap_results': black_gap_results,
            'white_valid_circles': [],
            'white_all_circles': [],
            'white_gap_results': {},
            'center': (center_x, center_y),
            'mean_brightness': mean_brightness,
            'is_white_background': is_white_background,
            'selected_type': 'black',
            'black_circle_count': black_circle_count,
            'white_circle_count': white_circle_count,
            'black_continuity_score': black_continuity_score,
            'white_continuity_score': white_continuity_score
        }
    elif selected_type == 'white':
        return {
            'black_valid_circles': [],
            'black_all_circles': [],
            'black_gap_results': {},
            'white_valid_circles': white_valid_circles,
            'white_all_circles': white_all_circles,
            'white_gap_results': white_gap_results,
            'center': (center_x, center_y),
            'mean_brightness': mean_brightness,
            'is_white_background': is_white_background,
            'selected_type': 'white',
            'black_circle_count': black_circle_count,
            'white_circle_count': white_circle_count,
            'black_continuity_score': black_continuity_score,
            'white_continuity_score': white_continuity_score
        }
    else:
        # どちらも選択されない場合
        return {
            'black_valid_circles': [],
            'black_all_circles': [],
            'black_gap_results': {},
            'white_valid_circles': [],
            'white_all_circles': [],
            'white_gap_results': {},
            'center': (center_x, center_y),
            'mean_brightness': mean_brightness,
            'is_white_background': is_white_background,
            'selected_type': None,
            'black_circle_count': black_circle_count,
            'white_circle_count': white_circle_count,
            'black_continuity_score': black_continuity_score,
            'white_continuity_score': white_continuity_score
        }

def detect_gap_intervals(gap_angles, min_gap_size=None):
    """
    ギャップ角度のリストから連続する区間を検出
    
    Parameters:
    - gap_angles: ギャップの角度リスト（度）
    - min_gap_size: 最小ギャップサイズ（度）（Noneの場合は設定値を使用）
    
    Returns:
    - intervals: [(start_deg, end_deg), ...] のリスト
    """
    if min_gap_size is None:
        min_gap_size = LANDOLT_MIN_GAP_SIZE
    
    if not gap_angles:
        return []
    
    # 角度をソート
    angles = sorted(gap_angles)
    intervals = []
    
    if len(angles) == 1:
        return [(angles[0] - 2.5, angles[0] + 2.5)]
    
    current_start = angles[0]
    current_end = angles[0]
    
    for i in range(1, len(angles)):
        angle = angles[i]
        
        # 連続性をチェック（角度の循環も考慮）
        if angle - current_end <= min_gap_size:
            current_end = angle
        else:
            # 区間を確定
            if (current_end - current_start) >= min_gap_size or current_start == current_end:
                intervals.append((current_start, current_end))
            current_start = angle
            current_end = angle
    
    # 最後の区間
    if (current_end - current_start) >= min_gap_size or current_start == current_end:
        intervals.append((current_start, current_end))
    
    return intervals

def angle_to_clock_hour(angle_deg):
    """角度を時刻（0-11）に変換"""
    # 0度が3時の位置、時計回りに進む
    # 時計の12時を0度とするため90度回転
    hour_angle = (angle_deg + 90) % 360
    hour = int((hour_angle / 30) + 0.5) % 12
    return hour

def create_hsv_binary_visualization(img, center_x, center_y, num_circles=None, brightness_threshold=None):
    """
    HSVの分類結果を可視化する画像を作成（黒/白ランドルト環自動判別）
    
    Parameters:
    - img: 入力画像 (BGR)
    - center_x, center_y: 中心座標
    - num_circles: 同心円の数（Noneの場合は設定値を使用）
    - brightness_threshold: 明度判定の閾値（Noneの場合は設定値を使用）
    
    Returns:
    - binary_img: 分類結果の画像（ランドルト環色=青、穴色=赤、グレー=サンプルされていない領域）
    """
    # 設定値を使用
    if num_circles is None:
        num_circles = LANDOLT_NUM_CIRCLES
    if brightness_threshold is None:
        brightness_threshold = BLACK_THRESHOLD
    # 画像全体の明度を分析して白背景/黒背景を判定
    is_white_background, mean_brightness = analyze_image_brightness(img)
    is_white_landolt = is_white_background  # 白背景なら白いランドルト環
    
    # BGRをRGBに変換してからHSVに変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = rgb_to_hsv_opencv(img_rgb)
    
    H, W = img.shape[:2]
    
    # 初期化（グレーで埋める）
    binary_img = np.full((H, W, 3), 128, dtype=np.uint8)  # グレー背景
    
    # 画像の最小サイズに基づいて最大半径を決定
    max_radius = min(W, H) // 3
    min_radius = max_radius * 0.1
    
    # 同心円の半径を計算（外側から内側へ、外側3つを除外）
    all_radii = np.linspace(max_radius * 0.9, min_radius, num_circles + 3)
    radii = all_radii[3:]  # 外側の3つを除外
    
    for radius in radii:
        # 円周上のサンプル点を取得
        n_samples = max(360, int(2 * np.pi * radius))
        theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        
        # 円周上の座標計算
        x_coords = center_x + radius * np.cos(theta)
        y_coords = center_y + radius * np.sin(theta)
        
        # 画像境界内の点のみを取得
        valid_mask = ((x_coords >= 0) & (x_coords < W) & 
                     (y_coords >= 0) & (y_coords < H))
        
        if not valid_mask.any():
            continue
            
        # 有効な座標のみを使用
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        
        # 整数座標に変換（最近傍）
        x_int = np.clip(np.round(x_valid).astype(int), 0, W-1)
        y_int = np.clip(np.round(y_valid).astype(int), 0, H-1)
        
        # HSV値を取得
        hsv_samples = hsv_img[y_int, x_int]
        
        # ランドルト環分類結果を描画
        for i, hsv_pixel in enumerate(hsv_samples):
            is_landolt_color = check_is_target_color(hsv_pixel, brightness_threshold, is_white_landolt)
            x, y = x_int[i], y_int[i]
            
            if is_landolt_color:
                # ランドルト環色 → 青で描画
                cv2.circle(binary_img, (x, y), 2, (255, 0, 0), -1)  # 青
            else:
                # 穴色 → 赤で描画
                cv2.circle(binary_img, (x, y), 2, (0, 0, 255), -1)  # 赤
    
    # 中心点を描画
    cv2.circle(binary_img, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
    
    # 円を描画
    for radius in radii:
        cv2.circle(binary_img, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 1)
    
    return binary_img

def create_hsv_overlay_on_original(img, center_x, center_y, num_circles=None, brightness_threshold=None, alpha=None):
    """
    元画像にHSVの分類結果を半透明でオーバーレイした画像を作成（ブラックとホワイトの両方を同時表示）
    設定値以上のランドルト環色を持つランドルト環のみを表示し、色分けして表示
    
    Parameters:
    - img: 入力画像 (BGR)
    - center_x, center_y: 中心座標
    - num_circles: 同心円の数（Noneの場合は設定値を使用）
    - brightness_threshold: 明度判定の閾値（Noneの場合は設定値を使用）
    - alpha: オーバーレイの透明度（Noneの場合は設定値を使用）
    
    Returns:
    - overlay_img: 元画像にHSV分類結果をオーバーレイした画像
    """
    # 設定値を使用
    if num_circles is None:
        num_circles = LANDOLT_NUM_CIRCLES
    if brightness_threshold is None:
        brightness_threshold = BLACK_THRESHOLD
    if alpha is None:
        alpha = LANDOLT_OVERLAY_ALPHA
    # BGRをRGBに変換してからHSVに変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = rgb_to_hsv_opencv(img_rgb)
    
    H, W = img.shape[:2]
    
    # 元画像をコピー
    overlay_img = img.copy()
    
    # オーバーレイ用の透明レイヤーを作成
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 画像の最小サイズに基づいて最大半径を決定
    max_radius = min(W, H) // 3
    min_radius = max_radius * 0.1
    
    # 同心円の半径を計算（外側から内側へ、外側3つを除外）
    all_radii = np.linspace(max_radius * 0.9, min_radius, num_circles + 3)
    radii = all_radii[3:]  # 外側の3つを除外
    
    # 各円についてブラックとホワイトの両方をチェック
    black_valid_radii = []
    white_valid_radii = []
    
    for radius in radii:
        # 円周上のサンプル点を取得
        n_samples = max(360, int(2 * np.pi * radius))
        theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        
        # 円周上の座標計算
        x_coords = center_x + radius * np.cos(theta)
        y_coords = center_y + radius * np.sin(theta)
        
        # 画像境界内の点のみを取得
        valid_mask = ((x_coords >= 0) & (x_coords < W) & 
                     (y_coords >= 0) & (y_coords < H))
        
        if not valid_mask.any():
            continue
            
        # 有効な座標のみを使用
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        
        # 整数座標に変換（最近傍）
        x_int = np.clip(np.round(x_valid).astype(int), 0, W-1)
        y_int = np.clip(np.round(y_valid).astype(int), 0, H-1)
        
        # HSV値を取得
        hsv_samples = hsv_img[y_int, x_int]
        
        # ブラックランドルト環色一致率を計算
        black_landolt_count = 0
        white_landolt_count = 0
        total_count = len(hsv_samples)
        
        for hsv_pixel in hsv_samples:
            if check_is_target_color(hsv_pixel, brightness_threshold, is_white_landolt=False):
                black_landolt_count += 1
            if check_is_target_color(hsv_pixel, brightness_threshold, is_white_landolt=True):
                white_landolt_count += 1
        
        black_match_ratio = black_landolt_count / total_count if total_count > 0 else 0.0
        white_match_ratio = white_landolt_count / total_count if total_count > 0 else 0.0
        
        # 設定値以上のランドルト環色がある場合のみランドルト環として認識
        if black_match_ratio >= LANDOLT_MATCH_RATIO_THRESHOLD:
            black_valid_radii.append(radius)
        
        if white_match_ratio >= LANDOLT_MATCH_RATIO_THRESHOLD:
            white_valid_radii.append(radius)
    
    # 連続性を考慮した選択ロジック（可視化用）
    def calculate_continuity_score_vis(radii):
        """可視化用の連続性スコアを計算"""
        if not radii:
            return 0
        
        # 半径でソート
        sorted_radii = sorted(radii)
        max_continuity = 1
        current_continuity = 1
        
        for i in range(1, len(sorted_radii)):
            # 半径の差が小さい場合は連続とみなす
            radius_diff = sorted_radii[i] - sorted_radii[i-1]
            if radius_diff < (max_radius - min_radius) / num_circles * 2:  # 2つ分以内の差
                current_continuity += 1
            else:
                max_continuity = max(max_continuity, current_continuity)
                current_continuity = 1
        
        max_continuity = max(max_continuity, current_continuity)
        return max_continuity
    
    # 連続性スコアを計算
    black_continuity_score = calculate_continuity_score_vis(black_valid_radii)
    white_continuity_score = calculate_continuity_score_vis(white_valid_radii)
    
    # 選択されたタイプを決定（連続性スコアが高い方を選択）
    selected_type = None
    if black_continuity_score > white_continuity_score:
        selected_type = 'black'
    elif white_continuity_score > black_continuity_score:
        selected_type = 'white'
    elif len(black_valid_radii) > len(white_valid_radii):
        # 連続性スコアが同じ場合は、円の数が多い方を選択
        selected_type = 'black'
    elif len(white_valid_radii) > len(black_valid_radii):
        selected_type = 'white'
    # 円の数が同じ場合は、どちらも描画しない
    
    # 選択されたタイプのみを描画
    if selected_type == 'black':
        # ブラックランドルト環を描画（青色系）
        for radius in black_valid_radii:
            # 円周上のサンプル点を取得
            n_samples = max(360, int(2 * np.pi * radius))
            theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
            
            # 円周上の座標計算
            x_coords = center_x + radius * np.cos(theta)
            y_coords = center_y + radius * np.sin(theta)
            
            # 画像境界内の点のみを取得
            valid_mask = ((x_coords >= 0) & (x_coords < W) & 
                         (y_coords >= 0) & (y_coords < H))
            
            if not valid_mask.any():
                continue
                
            # 有効な座標のみを使用
            x_valid = x_coords[valid_mask]
            y_valid = y_coords[valid_mask]
            
            # 整数座標に変換（最近傍）
            x_int = np.clip(np.round(x_valid).astype(int), 0, W-1)
            y_int = np.clip(np.round(y_valid).astype(int), 0, H-1)
            
            # HSV値を取得
            hsv_samples = hsv_img[y_int, x_int]
            
            # ブラックランドルト環の部分と穴を描画
            for i, hsv_pixel in enumerate(hsv_samples):
                is_black_landolt = check_is_target_color(hsv_pixel, brightness_threshold, is_white_landolt=False)
                x, y = x_int[i], y_int[i]
                
                if is_black_landolt:
                    # ブラックランドルト環部分を青色でマーク
                    cv2.circle(overlay, (x, y), 3, (255, 0, 0), -1)  # 青
                else:
                    # ブラックランドルト環の穴部分を赤色でマーク
                    cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)  # 赤、少し大きく
        
        # ブラックランドルト環の円を描画
        for radius in black_valid_radii:
            cv2.circle(overlay, (int(center_x), int(center_y)), int(radius), (255, 0, 0), 2)  # 青い円
    
    elif selected_type == 'white':
        # ホワイトランドルト環を描画（緑色系）
        for radius in white_valid_radii:
            # 円周上のサンプル点を取得
            n_samples = max(360, int(2 * np.pi * radius))
            theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
            
            # 円周上の座標計算
            x_coords = center_x + radius * np.cos(theta)
            y_coords = center_y + radius * np.sin(theta)
            
            # 画像境界内の点のみを取得
            valid_mask = ((x_coords >= 0) & (x_coords < W) & 
                         (y_coords >= 0) & (y_coords < H))
            
            if not valid_mask.any():
                continue
                
            # 有効な座標のみを使用
            x_valid = x_coords[valid_mask]
            y_valid = y_coords[valid_mask]
            
            # 整数座標に変換（最近傍）
            x_int = np.clip(np.round(x_valid).astype(int), 0, W-1)
            y_int = np.clip(np.round(y_valid).astype(int), 0, H-1)
            
            # HSV値を取得
            hsv_samples = hsv_img[y_int, x_int]
            
            # ホワイトランドルト環の部分と穴を描画
            for i, hsv_pixel in enumerate(hsv_samples):
                is_white_landolt = check_is_target_color(hsv_pixel, brightness_threshold, is_white_landolt=True)
                x, y = x_int[i], y_int[i]
                
                if is_white_landolt:
                    # ホワイトランドルト環部分を緑色でマーク
                    cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)  # 緑
                else:
                    # ホワイトランドルト環の穴部分をマゼンタ色でマーク
                    cv2.circle(overlay, (x, y), 4, (255, 0, 255), -1)  # マゼンタ、少し大きく
        
        # ホワイトランドルト環の円を描画
        for radius in white_valid_radii:
            cv2.circle(overlay, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 2)  # 緑の円
    
    # 中心点を描画（黄色）
    cv2.circle(overlay, (int(center_x), int(center_y)), 5, (0, 255, 255), -1)
    
    # アルファブレンディングで重ね合わせ
    overlay_img = cv2.addWeighted(overlay_img, 1.0 - alpha, overlay, alpha, 0)
    
    return overlay_img

def draw_landolt_analysis(img, analysis_result):
    """
    ランドルト環解析結果を画像に描画（ブラックとホワイトの両方を表示）
    
    Parameters:
    - img: 入力画像
    - analysis_result: detect_landolt_gaps()の結果
    
    Returns:
    - annotated_img: 解析結果が描画された画像
    """
    annotated_img = img.copy()
    center_x, center_y = analysis_result['center']
    
    # ブラックランドルト環の情報を取得
    black_valid_circles = analysis_result.get('black_valid_circles', [])
    black_gap_results = analysis_result.get('black_gap_results', {})
    
    # ホワイトランドルト環の情報を取得
    white_valid_circles = analysis_result.get('white_valid_circles', [])
    white_gap_results = analysis_result.get('white_gap_results', {})
    
    # 中心点を描画
    cv2.circle(annotated_img, (int(center_x), int(center_y)), 3, (255, 255, 0), -1)
    
    # ブラックランドルト環の有効な円を描画（青色系）
    for circle_info in black_valid_circles:
        radius = int(circle_info['radius'])
        match_ratio = circle_info['match_ratio']
        
        # 円を描画（マッチ率に応じて色を変更）
        if match_ratio >= 0.9:
            color = (255, 0, 0)  # 青：90%以上（最高品質ブラックランドルト環）
        elif match_ratio >= 0.8:
            color = (255, 100, 0)  # 水色：80-90%（有効ブラックランドルト環）
        else:
            color = (128, 128, 128)  # グレー：80%未満（無効）
        
        cv2.circle(annotated_img, (int(center_x), int(center_y)), radius, color, 1)
        
        # ギャップがある場合は時刻を表示
        if radius in black_gap_results:
            for gap_start, gap_end in black_gap_results[radius]:
                gap_center = (gap_start + gap_end) / 2
                hour = angle_to_clock_hour(gap_center)
                
                # 時刻表示位置
                text_radius = radius + 15
                text_angle = math.radians(gap_center)
                text_x = int(center_x + text_radius * math.cos(text_angle))
                text_y = int(center_y + text_radius * math.sin(text_angle))
                
                # 時刻を表示（ブラックランドルト環用）
                cv2.putText(annotated_img, f"B{hour}", (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # ギャップの弧を青で描画
                start_angle = int(gap_start)
                end_angle = int(gap_end)
                cv2.ellipse(annotated_img, (int(center_x), int(center_y)), 
                           (radius, radius), 0, start_angle, end_angle, (255, 0, 0), 3)
    
    # ホワイトランドルト環の有効な円を描画（緑色系）
    for circle_info in white_valid_circles:
        radius = int(circle_info['radius'])
        match_ratio = circle_info['match_ratio']
        
        # 円を描画（マッチ率に応じて色を変更）
        if match_ratio >= 0.9:
            color = (0, 255, 0)  # 緑：90%以上（最高品質ホワイトランドルト環）
        elif match_ratio >= 0.8:
            color = (0, 255, 100)  # 薄緑：80-90%（有効ホワイトランドルト環）
        else:
            color = (128, 128, 128)  # グレー：80%未満（無効）
        
        cv2.circle(annotated_img, (int(center_x), int(center_y)), radius, color, 1)
        
        # ギャップがある場合は時刻を表示
        if radius in white_gap_results:
            for gap_start, gap_end in white_gap_results[radius]:
                gap_center = (gap_start + gap_end) / 2
                hour = angle_to_clock_hour(gap_center)
                
                # 時刻表示位置（ホワイトランドルト環用は少しずらす）
                text_radius = radius + 25
                text_angle = math.radians(gap_center)
                text_x = int(center_x + text_radius * math.cos(text_angle))
                text_y = int(center_y + text_radius * math.sin(text_angle))
                
                # 時刻を表示（ホワイトランドルト環用）
                cv2.putText(annotated_img, f"W{hour}", (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # ギャップの弧を緑で描画
                start_angle = int(gap_start)
                end_angle = int(gap_end)
                cv2.ellipse(annotated_img, (int(center_x), int(center_y)), 
                           (radius, radius), 0, start_angle, end_angle, (0, 255, 0), 3)
    
    return annotated_img

# ----------------------------------------------------
# 1. OBS仮想カメラの読み込み設定
# ----------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)  # 仮想カメラのインデックス。環境に合わせて変更してください
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

if not cap.isOpened():
    print("エラー: OBS仮想カメラを開けませんでした。")
    print("OBSで「仮想カメラを開始」しているか、正しいカメラインデックスか確認してください。")
    exit()

# randoruto-wrsフォルダーの作成
save_folder = "randoruto-wrs"
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
pause_sequence_counter = 0  # pauseごとの連番カウンター

# YOLOv8nモデルのパス - target2.ptを使用
# 学習済みYOLOモデルファイルを読み込み
# target2.pt: ターゲット検出専用の学習済みモデル
model = YOLO('target3.pt')  # target2.ptファイルを使用

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

def save_detection_images(frame, detections_with_confidence, frame_count, leftup_image=None, leftdown_image=None, save_leftup=False, save_leftdown=False):
    """一時停止時に検出画像を保存する関数（左上ストップ時：左上、右下、全体の3種類）"""
    global save_counter, image_save_counter, pause_sequence_counter
    
    # pauseを押すたびに連番を1増やす
    if save_leftup or save_leftdown:
        pause_sequence_counter += 1
    
    # 検出物体がない場合でも保存を継続
    
    # ファイル名を生成（撮影順序で並ぶように調整）
    detection_count = len(detections_with_confidence)
    # 現在の日時を取得
    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    # 連番でファイル名を生成（6桁のゼロパディング）
    base_filename = f"capture_{pause_sequence_counter:06d}_{time_str}_frame{frame_count}_{detection_count}objects"
    
    # 全体画像：全ての検出物体にバウンディングボックスを付けた画像を保存（左上ストップ時のみ）
    if save_leftup:
        annotated_img = frame.copy()
        for i, detection in enumerate(detections_with_confidence):
            x1, y1, x2, y2 = detection['box']
            label = detection['label']
            confidence = detection['confidence']
            
            # バウンディングボックスを描画
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ラベルと信頼度のテキストを追加
            text = f"{label}: {confidence:.3f}"
            cv2.putText(annotated_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 全体画像を保存
        whole_filepath = os.path.join(session_folder, f"{base_filename}_R_whole_detection.jpg")
        cv2.imwrite(whole_filepath, annotated_img)
        print(f"全体画像（検出結果）を保存しました: {base_filename}_R_whole_detection.jpg (検出物体数: {detection_count})")
        image_save_counter += 1
    
    # 左上画像：HSVオーバーレイ画像とオーバーレイなし画像を保存（save_leftup=Trueの場合のみ）
    if save_leftup:
        if leftup_image is not None:
            try:
                # 画像サイズを決定
                save_size = 512  # 保存用のサイズ
                resized_img = cv2.resize(leftup_image, (save_size, save_size))
                center_x = save_size // 2
                center_y = save_size // 2
                
                # 1. 左上画像（オーバーレイなし）を保存
                leftup_original_filepath = os.path.join(session_folder, f"{base_filename}_M_leftup_original.jpg")
                cv2.imwrite(leftup_original_filepath, resized_img)
                print(f"左上画像（オーバーレイなし）を保存しました: {base_filename}_M_leftup_original.jpg")
                image_save_counter += 1
                
                # 2. HSVオーバーレイ画像を作成
                hsv_overlay_img = create_hsv_overlay_on_original(resized_img, center_x, center_y)
                
                # 3. 左上画像（HSVオーバーレイ）を保存
                leftup_filepath = os.path.join(session_folder, f"{base_filename}_A_leftup_hsv.jpg")
                cv2.imwrite(leftup_filepath, hsv_overlay_img)
                print(f"左上画像（HSVオーバーレイ）を保存しました: {base_filename}_A_leftup_hsv.jpg")
                image_save_counter += 1
                
            except Exception as e:
                print(f"左上画像の保存に失敗しました: {e}")
        else:
            # leftup_imageがない場合は、no_targetファイル名で保存
            # オーバーレイなし画像
            leftup_original_filepath = os.path.join(session_folder, f"{base_filename}_M_leftup_original_no_target.jpg")
            empty_img = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imwrite(leftup_original_filepath, empty_img)
            print(f"左上画像（オーバーレイなし、No Target）を保存しました: {base_filename}_M_leftup_original_no_target.jpg")
            image_save_counter += 1
            
            # HSVオーバーレイ画像
            leftup_filepath = os.path.join(session_folder, f"{base_filename}_A_leftup_hsv_no_target.jpg")
            cv2.imwrite(leftup_filepath, empty_img)
            print(f"左上画像（HSVオーバーレイ、No Target）を保存しました: {base_filename}_A_leftup_hsv_no_target.jpg")
            image_save_counter += 1
    
    # 左下画像：セカンダリ検出画像を保存（save_leftdown=Trueの場合のみ）
    if save_leftdown:
        # 全体画像：全ての検出物体にバウンディングボックスを付けた画像を保存
        try:
            annotated_img = frame.copy()
            for i, detection in enumerate(detections_with_confidence):
                x1, y1, x2, y2 = detection['box']
                label = detection['label']
                confidence = detection['confidence']
                
                # バウンディングボックスを描画
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # ラベルと信頼度のテキストを追加
                text = f"{label}: {confidence:.3f}"
                cv2.putText(annotated_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 全体画像を保存
            whole_filepath = os.path.join(session_folder, f"{base_filename}_R_whole_detection.jpg")
            cv2.imwrite(whole_filepath, annotated_img)
            print(f"全体画像（検出結果）を保存しました: {base_filename}_R_whole_detection.jpg (検出物体数: {detection_count})")
            image_save_counter += 1
            
        except Exception as e:
            print(f"全体画像の保存に失敗しました: {e}")
        
        if leftdown_image is not None:
            try:
                # 左下画像をリサイズして保存
                save_size = 512
                resized_leftdown = cv2.resize(leftdown_image, (save_size, save_size))
                
                # 左下画像を保存
                leftdown_filepath = os.path.join(session_folder, f"{base_filename}_M_leftdown.jpg")
                cv2.imwrite(leftdown_filepath, resized_leftdown)
                print(f"左下画像（セカンダリ検出）を保存しました: {base_filename}_M_leftdown.jpg")
                image_save_counter += 1
                
            except Exception as e:
                print(f"左下画像の保存に失敗しました: {e}")
        else:
            # leftdown_imageがない場合は、no_targetファイル名で保存
            leftdown_filepath = os.path.join(session_folder, f"{base_filename}_M_leftdown_no_target.jpg")
            # 空の画像を作成（512x512の黒画像）
            empty_img = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imwrite(leftdown_filepath, empty_img)
            print(f"左下画像（No Target）を保存しました: {base_filename}_M_leftdown_no_target.jpg")
            image_save_counter += 1

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
            conf_text = f"{leftup_info.get('label','')}: {leftup_info['confidence']:.3f}"
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
            conf_text = f"{leftdown_info.get('label','')}: {leftdown_info['confidence']:.3f}"
            cv2.putText(panel_ld, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        panel_ld = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        cv2.putText(panel_ld, 'No Secondary Target', (40, half_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    
    # 左下パネルに色付きの縁取りを追加
    panel_ld = add_colored_border(panel_ld, leftdown_paused)

    # 右下：HSVオーバーレイ（ランドルト環の穴検出）
    if leftup_image is not None:
        try:
            # 正方形にリサイズして歪みを防ぐ
            square_size = min(half_w, half_h)
            resized_img = cv2.resize(leftup_image, (square_size, square_size))
            
            # パネルサイズに合わせて余白を追加（中央揃え）
            panel_rd = np.zeros((half_h, half_w, 3), dtype=np.uint8)
            start_x = (half_w - square_size) // 2
            start_y = (half_h - square_size) // 2
            
            # ランドルト環の穴検出を実行（正方形画像の中心を使用）
            center_x = square_size // 2
            center_y = square_size // 2
            
            # ランドルト環の穴検出
            landolt_result = detect_landolt_gaps(resized_img, center_x, center_y)
            
            # HSVオーバーレイ画像を作成（設定値以上のランドルト環のみ表示）
            hsv_overlay_img = create_hsv_overlay_on_original(resized_img, center_x, center_y)
            
            # HSVオーバーレイ画像をパネルの正しい位置に配置
            panel_rd[start_y:start_y+square_size, start_x:start_x+square_size] = hsv_overlay_img
            
            # 選択されたランドルト環タイプの情報を取得
            selected_type = landolt_result.get('selected_type', None)
            black_circle_count = landolt_result.get('black_circle_count', 0)
            white_circle_count = landolt_result.get('white_circle_count', 0)
            black_continuity_score = landolt_result.get('black_continuity_score', 0)
            white_continuity_score = landolt_result.get('white_continuity_score', 0)
            brightness = landolt_result.get('mean_brightness', 0)
            
            # 選択されたタイプに基づいて情報を表示
            if selected_type == 'black':
                # ブラックランドルト環の検出された穴の情報をテキストで表示
                black_gap_info = []
                for radius, gaps in landolt_result.get('black_gap_results', {}).items():
                    for gap_start, gap_end in gaps:
                        gap_center = (gap_start + gap_end) / 2
                        hour = angle_to_clock_hour(gap_center)
                        black_gap_info.append(f"r{int(radius)}:{hour}時")
                
                if black_gap_info:
                    gap_text = " ".join(black_gap_info[:2])  # 最大2つまで表示
                    cv2.putText(panel_rd, "Black Landolt:", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # 青色
                    cv2.putText(panel_rd, gap_text, (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # 青色
                    cv2.putText(panel_rd, f"Circles: {black_circle_count}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # 青色
                    cv2.putText(panel_rd, f"Continuity: {black_continuity_score}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # 青色
                else:
                    cv2.putText(panel_rd, "Black Landolt (No Gaps)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # 青色
                    cv2.putText(panel_rd, f"Circles: {black_circle_count}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # 青色
                    cv2.putText(panel_rd, f"Continuity: {black_continuity_score}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # 青色
            
            elif selected_type == 'white':
                # ホワイトランドルト環の検出された穴の情報をテキストで表示
                white_gap_info = []
                for radius, gaps in landolt_result.get('white_gap_results', {}).items():
                    for gap_start, gap_end in gaps:
                        gap_center = (gap_start + gap_end) / 2
                        hour = angle_to_clock_hour(gap_center)
                        white_gap_info.append(f"r{int(radius)}:{hour}時")
                
                if white_gap_info:
                    gap_text = " ".join(white_gap_info[:2])  # 最大2つまで表示
                    cv2.putText(panel_rd, "White Landolt:", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # 緑色
                    cv2.putText(panel_rd, gap_text, (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # 緑色
                    cv2.putText(panel_rd, f"Circles: {white_circle_count}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # 緑色
                    cv2.putText(panel_rd, f"Continuity: {white_continuity_score}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # 緑色
                else:
                    cv2.putText(panel_rd, "White Landolt (No Gaps)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # 緑色
                    cv2.putText(panel_rd, f"Circles: {white_circle_count}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # 緑色
                    cv2.putText(panel_rd, f"Continuity: {white_continuity_score}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # 緑色
            
            else:
                # どちらも選択されていない場合
                cv2.putText(panel_rd, "No Clear Landolt Ring", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)  # 黄色
                cv2.putText(panel_rd, f"Circles - B:{black_circle_count} W:{white_circle_count}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)  # 黄色
                cv2.putText(panel_rd, f"Continuity - B:{black_continuity_score} W:{white_continuity_score}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)  # 黄色
            
            # 明度情報を表示
            cv2.putText(panel_rd, f"Brightness: {brightness:.0f}", (10, half_h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
        except Exception as e:
            # エラーが発生した場合は認識情報を表示
            panel_rd = np.zeros((half_h, half_w, 3), dtype=np.uint8)
            cv2.putText(panel_rd, f"Landolt Error: {str(e)[:20]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # 基本的な認識情報も表示
            info_lines = []
            if detections_with_confidence:
                info_lines.append(f'検出数: {len(detections_with_confidence)}')
                if len(detections_with_confidence) >= 1:
                    info_lines.append(f'左上: {detections_with_confidence[0]["confidence"]:.3f}')
            for i, line in enumerate(info_lines[:3]):
                cv2.putText(panel_rd, line, (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        # 左上画像がない場合は従来の認識情報を表示
        panel_rd = np.zeros((half_h, half_w, 3), dtype=np.uint8)
        info_lines = []
        if detections_with_confidence:
            info_lines.append(f'Target Detections: {len(detections_with_confidence)}')
            if len(detections_with_confidence) >= 1:
                info_lines.append(f'Top-Left(1): {detections_with_confidence[0]["label"]} {detections_with_confidence[0]["confidence"]:.3f}')
            if len(detections_with_confidence) >= 2:
                info_lines.append(f'Bottom-Left(2): {detections_with_confidence[1]["label"]} {detections_with_confidence[1]["confidence"]:.3f}')
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
            # 左上一時停止時に画像を保存
            save_detection_images(frame, detections_with_confidence, leftup_frame_count, leftup_image, leftdown_image, save_leftup=True, save_leftdown=False)
            print(f"左上一時停止時の画像を保存しました (Total: {save_counter} files)")
            pause_counter += 1
        else:
            print("左上を再開しました")
    elif key == ord('2'):  # 2キー：左下の一時停止/再開
        leftdown_paused = not leftdown_paused
        if leftdown_paused:
            print("左下を一時停止しました")
            # 左下一時停止時に画像を保存
            save_detection_images(frame, detections_with_confidence, leftdown_frame_count, leftup_image, leftdown_image, save_leftup=False, save_leftdown=True)
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
            # 両方一時停止時に画像を保存
            save_detection_images(frame, detections_with_confidence, leftup_frame_count, leftup_image, leftdown_image, save_leftup=True, save_leftdown=True)
            print(f"両方一時停止時の画像を保存しました (Total: {save_counter} files)")
            pause_counter += 1
        else:
            print("左上と左下を同時に再開しました")

cap.release()
cv2.destroyAllWindows()
print("ターゲット検出を終了しました。")
print(f"保存された画像の総数: {save_counter}枚")
print(f"保存先フォルダー: {session_folder}")


  