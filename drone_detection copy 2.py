import cv2
from ultralytics import YOLO

# 再学習済みYOLOv8nモデルのパスを指定
model = YOLO("weights.pt")  # ここを自分のモデルファイル名に変更

# カメラのデバイス番号（必要に応じて変更）
cap = cv2.VideoCapture(0)

# 解像度指定（任意）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("リアルタイム検出を開始します。'q'キーで終了します。")

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
    cv2.imshow("YOLOv8n Detection (Press 'q' to quit)", annotated_frame)

    # 'q'で終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
print("検出を終了しました。")
