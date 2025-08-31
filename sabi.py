import cv2
import numpy as np
import os
from datetime import datetime

def detect_rust_percentage(image_path, output_folder='sabi'):
    """
    画像全体に対する錆の割合を測定する関数
    
    Args:
        image_path (str): 入力画像のパス
        output_folder (str): 出力フォルダ名
    
    Returns:
        float: 錆の割合（%）
    """
    
    # 画像を読み込み
    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: 画像 {image_path} を読み込めませんでした。")
        return None
    
    # BGRからHSVに変換
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # デバッグ用：画像のHSV値を確認
    print("=== HSV値の確認 ===")
    print(f"画像のHSV値の範囲:")
    print(f"H: {img_hsv[:,:,0].min()} - {img_hsv[:,:,0].max()}")
    print(f"S: {img_hsv[:,:,1].min()} - {img_hsv[:,:,1].max()}")
    print(f"V: {img_hsv[:,:,2].min()} - {img_hsv[:,:,2].max()}")
    
    # RGB(85, 57, 45)のHSV値を計算
    rgb_test = np.array([[[85, 57, 45]]], dtype=np.uint8)
    bgr_test = cv2.cvtColor(rgb_test, cv2.COLOR_RGB2BGR)
    hsv_test = cv2.cvtColor(bgr_test, cv2.COLOR_BGR2HSV)
    print(f"RGB(85, 57, 45) → HSV({hsv_test[0,0,0]}, {hsv_test[0,0,1]}, {hsv_test[0,0,2]})")
    
    # 錆のHSV範囲を定義（提供されたHSV値に基づいて調整）
    # 提供されたHSV値: (8,132,89), (11,84,63), (11,76,71), (3,43,114), (7,102,89)
    # これらの値を網羅する範囲: H(3-11), S(43-132), V(63-114)
    # 少し余裕を持たせて設定
    rust_h_min, rust_s_min, rust_v_min = 2, 34, 40   # 錆の下限
    rust_h_max, rust_s_max, rust_v_max = 13, 145, 120   # 錆の上限
    
    print(f"現在の錆検出範囲: H({rust_h_min}-{rust_h_max}), S({rust_s_min}-{rust_s_max}), V({rust_v_min}-{rust_v_max})")
    
    # 錆のマスクを作成
    rust_lower = np.array([rust_h_min, rust_s_min, rust_v_min])
    rust_upper = np.array([rust_h_max, rust_s_max, rust_v_max])
    rust_mask = cv2.inRange(img_hsv, rust_lower, rust_upper)
    
    # ノイズ除去のためのモルフォロジー処理
    kernel = np.ones((3, 3), np.uint8)
    rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
    rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)
    
    # ピクセル数を計算
    total_pixels = img.shape[0] * img.shape[1]
    rust_pixels = cv2.countNonZero(rust_mask)
    
    # 錆の割合を計算
    rust_percentage = (rust_pixels / total_pixels) * 100
    
    # マーキング画像を作成
    marked_image = img.copy()
    
    # 錆の部分を赤色でマーキング
    marked_image[rust_mask > 0] = [0, 0, 255]  # 赤色で錆をマーキング
    
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # タイムスタンプ付きでファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"rust_detection_{timestamp}.jpg"
    output_path = os.path.join(output_folder, output_filename)
    
    # マーキングした画像を保存
    cv2.imwrite(output_path, marked_image)
    
    # 錆のマスクも保存
    rust_mask_filename = f"rust_mask_{timestamp}.jpg"
    rust_mask_path = os.path.join(output_folder, rust_mask_filename)
    cv2.imwrite(rust_mask_path, rust_mask)
    
    # HSV画像も保存（デバッグ用）
    hsv_filename = f"hsv_debug_{timestamp}.jpg"
    hsv_path = os.path.join(output_folder, hsv_filename)
    cv2.imwrite(hsv_path, img_hsv)
    
    print(f"画像全体に対する錆の割合: {rust_percentage:.2f}%")
    print(f"錆のピクセル数: {rust_pixels}")
    print(f"総ピクセル数: {total_pixels}")
    print(f"マーキング画像を保存しました: {output_path}")
    print(f"錆のマスクを保存しました: {rust_mask_path}")
    print(f"HSV画像を保存しました: {hsv_path}")
    
    return rust_percentage

def main():
    """メイン関数"""
    # sabi.jpgファイルの錆を検出
    image_path = 'sabi.jpg'
    
    if os.path.exists(image_path):
        print(f"画像 {image_path} の錆を検出中...")
        rust_percentage = detect_rust_percentage(image_path)
        
        if rust_percentage is not None:
            print(f"\n=== 結果 ===")
            print(f"画像全体に対する錆の割合: {rust_percentage:.2f}%")
        else:
            print("錆の検出に失敗しました。")
    else:
        print(f"エラー: ファイル {image_path} が見つかりません。")

if __name__ == "__main__":
    main()