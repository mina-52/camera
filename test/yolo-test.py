import cv2
from ultralytics import YOLO

# モデルのパス
model = YOLO('weights.pt')  # 必要に応じて変更

# カメラのデバイス番号（通常は0または1。OBS仮想カメラなら1）
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("YOLOv8n リアルタイム認識を開始します。'q'キーで終了します。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("カメラから映像を取得できませんでした。")
        break

    # YOLOで推論
    results = model(frame)
    # アノテーション付き画像を取得
    annotated_frame = results[0].plot()

    # 表示
    cv2.imshow("YOLOv8n RealTime", annotated_frame)

    # 'q'で終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("認識を終了しました。")
