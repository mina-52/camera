import cv2
import numpy as np

# === 画像読み込み ===
img = cv2.imread("sabi.jpg")

# === HSVに変換 ===
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# === 錆っぽい色範囲 (要調整) ===
lower_brown = np.array([5, 80, 40])    # H, S, V の下限
upper_brown = np.array([20, 255, 200]) # H, S, V の上限
mask = cv2.inRange(hsv, lower_brown, upper_brown)

# === ノイズ除去 ===
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# === 輪郭検出 ===
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    # 面積フィルタ (小さすぎ/大きすぎを除外)
    if area < 50 or area > 5000:
        continue

    # 円形度フィルタ
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity < 0.6:  # 円に近いかどうか
        continue

    valid_contours.append(cnt)

# === 板の実寸面積 (200×200 mm) ===
board_area = 200 * 200
rust_area = sum(cv2.contourArea(c) for c in valid_contours)
rust_ratio = rust_area / board_area * 100

print("錆の個数:", len(valid_contours))
print("錆の総面積(mm^2):", rust_area)
print("錆の割合(%):", rust_ratio)

# === 可視化 ===
for i, cnt in enumerate(valid_contours):
    cv2.drawContours(img, [cnt], -1, (0,0,255), 2)  # 赤枠
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        cv2.putText(img, str(i+1), (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,0,0), 2)  # 番号

cv2.imshow("Mask", mask)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()