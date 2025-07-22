import cv2
from ultralytics import YOLO

# モデルを読み込み（軽量版で速度優先：yolov8n）
model = YOLO("yolov8s.pt")  # yolov8s.pt や yolov8m.pt に変更も可

# 仮想カメラのデバイス番号（通常は0だが、異なる場合は試す）
cap = cv2.VideoCapture(1)

# 解像度指定（任意）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("映像を取得できませんでした。")
        break

    # YOLOで推論
    results = model(frame)

    # アノテーション付き画像を取得
    annotated_frame = results[0].plot()

    # 表示
    cv2.imshow("YOLOv8 Detection (Press 'q' to quit)", annotated_frame)

    # 'q'で終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
