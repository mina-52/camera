import cv2
import numpy as np
from ultralytics import YOLO # YOLOv8を使用する場合

# ----------------------------------------------------
# 1. OBS仮想カメラの読み込み設定
# ----------------------------------------------------
# OBS仮想カメラのインデックスを確認する
# 通常は0から始まるが、環境によって異なる場合がある。
# 複数のカメラがある場合は、正しいインデックスを探す必要がある。
# 正しいインデックスが分からない場合は、0から順に試すか、
# cv2.VideoCapture(i) for i in range(10) などで試行錯誤する。
# Macの「システム情報」->「カメラ」で確認できる場合もあります。

# 上記のテストでOBS仮想カメラの正しいインデックスを見つけてください。S
# 例: もしテストでインデックス3がOBSカメラだと分かったら、下の行を cap = cv2.VideoCapture(3) に変更します。
# 自分で正しいインデックスを特定し、下の行の 1 をそのインデックスに置き換えてください。
# もし見つからなければ、found_obs_camera_index = あなたが見つけたインデックス に手動で設定してください。


cap = cv2.VideoCapture(1) # 仮想カメラのインデックス。環境に合わせて変更してください

if not cap.isOpened():
    print("エラー: OBS仮想カメラを開けませんでした。")
    print("OBSで「仮想カメラを開始」しているか、正しいカメラインデックスか確認してください。")
    exit()


# Roboflowからエクスポートしたモデル（通常は.ptファイル）のパスを指定
model = YOLO('yolov8n.pt') # ここをあなたのモデルのパスに置き換えてください

# ランドルト環のクラス名（Roboflowでのクラス名と一致させる）
# 例: class_names = ['landolt_c']
# モデルの学習時に指定したクラス名リストを正確に記述してください
class_names = ['randoruto'] # ここをあなたのクラス名リストに置き換えてください

print("リアルタイム検出を開始します。'q'キーで終了します。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを読み込めませんでした。")
        break

    # ----------------------------------------------------
    # 3. YOLOによる検出
    # ----------------------------------------------------
    # 推論を実行
    # conf=0.5 は信頼度スコアが0.5以上の検出のみ表示
    # iou=0.7 は重複するバウンディングボックスの除去（NMS）の閾値
    results = model(frame, conf=0.5, iou=0.7, verbose=False) # verbose=Falseでログ出力を抑える

    # 検出結果の描画
    annotated_frame = frame.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # バウンディングボックスの座標
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 信頼度スコア
            confidence = round(float(box.conf[0]), 2)
            # クラスID
            class_id = int(box.cls[0])
            # クラス名
            class_name = class_names[class_id] if class_id < len(class_names) else f'Unknown_{class_id}'

            # バウンディングボックスを描画
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 緑色の矩形

            # ラベル（クラス名と信頼度）を描画
            label = f'{class_name} {confidence}'
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ----------------------------------------------------
    # 4. 検出結果の表示
    # ----------------------------------------------------
    cv2.imshow('Landolt C Detection', annotated_frame)

    # 'q' キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------------------
# 5. リソースの解放
# ----------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("検出を終了しました。")
