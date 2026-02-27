import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def process_strawberry_3pins(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # 1. 青いピンを3つ探す
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue, upper_blue = np.array([100, 150, 50]), np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pins = []
    for cnt in blue_cnts:
        if cv2.contourArea(cnt) > 50: # 小さすぎるノイズを除外
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                pins.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

    # ピンが3つ未満の場合は処理できない
    if len(pins) < 3: return image_bytes 

    # 3点から4点目を推測するロジック
    # 3点を時計回りに並べ替える必要がある (簡易版として、x+yの順でソート)
    pins = sorted(pins, key=lambda p: p[0] + p[1])
    p1 = np.array(pins[0]) # 左上(と仮定)
    p2 = np.array(pins[1])
    p3 = np.array(pins[2]) # 右下(と仮定)

    # p1, p2, p3 の位置関係からL字の角を探す
    v1 = p2 - p1
    v2 = p3 - p2
    # ベクトルの内積が0に近いところが直角(角)
    if abs(np.dot(v1, v2)) < abs(np.dot(p2-p1, p3-p1)):
        corner = p2
        q1, q2 = p1, p3
    elif abs(np.dot(p2-p1, p3-p1)) < abs(np.dot(p3-p2, p1-p2)):
        corner = p1
        q1, q2 = p2, p3
    else:
        corner = p3
        q1, q2 = p1, p2

    # 4点目の推測: 2つのベクトルを足す
    p4 = q1 + q2 - corner
    all_pins = np.array([p1, p2, p3, p4], dtype="float32")
    
    # 4点を時計回りに整列 (射影変換用)
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # 左上
        rect[2] = pts[np.argmax(s)] # 右下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # 右上
        rect[3] = pts[np.argmax(diff)] # 左下
        return rect

    ordered_pins = order_points(all_pins)

    # --- 2. 射影変換（歪み補正） ---
    # 変換後の目標サイズ (320mm x 570mm に合わせたピクセル数)
    dst_w, dst_h = 640, 1140 
    dst_pts = np.array([
        [0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]
    ], dtype="float32")

    # 変換行列を計算して、画像を補正
    M_transform = cv2.getPerspectiveTransform(ordered_pins, dst_pts)
    warped_img = cv2.warpPerspective(img, M_transform, (dst_w, dst_h))

    # --- 3. 補正後の画像でグリッド解析 ---
    # 背景を白黒化
    gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    output_img = cv2.cvtColor(gray_warped, cv2.COLOR_GRAY2BGR)

    # 換算係数
    px_to_mm = 320 / dst_w
    
    cols, rows = 7, 13
    cell_w, cell_h = dst_w / cols, dst_h / rows

    # 赤色の抽出
    hsv_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50]) + np.array([170, 70, 50]) # 結合
    upper_red = np.array([10, 255, 255]) + np.array([180, 255, 255])
    # 注意: ここでHSVでの結合はmaskで行う必要がある。上記は概念。
    mask1 = cv2.inRange(hsv_warped, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv_warped, np.array([170, 70, 50]), np.array([180, 255, 255]))
    red_mask = mask1 + mask2

    # スキャンと判定
    for r in range(rows):
        for c in range(cols):
            center_x, center_y = int((c + 0.5) * cell_w), int((r + 0.5) * cell_h)
            
            roi_h, roi_w = int(cell_h * 0.7), int(cell_w * 0.7)
            y1, y2 = max(0, center_y - roi_h//2), min(dst_h, center_y + roi_h//2)
            x1, x2 = max(0, center_x - roi_w//2), min(dst_w, center_x + roi_w//2)
            
            roi_mask = red_mask[y1:y2, x1:x2]
            cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                largest_cnt = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(largest_cnt) > 200: # 補正後なので小さめで
                    _, (pw, ph), _ = cv2.minAreaRect(largest_cnt)
                    size_mm = max(pw, ph) * px_to_mm
                    
                    if size_mm >= 35: label, color = "L", (0, 0, 255)
                    elif size_mm >= 28: label, color = "M", (255, 0, 0)
                    else: label, color = "S", (0, 255, 0)

                    # --- 文字描画 ---
                    font = cv2.FONT_HERSHEY_DUPLEX
                    fs, thick = 3.5, 8
                    cv2.putText(output_img, label, (center_x-30, center_y+20), font, fs, (0,0,0), thick+4) # 縁
                    cv2.putText(output_img, label, (center_x-30, center_y+20), font, fs, color, thick)

    # 4. 最後に画像をjpegにして返す
    _, encoded_img = cv2.imencode(".jpg", output_img)
    return encoded_img.tobytes()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    result = process_strawberry_3pins(contents)
    return Response(content=result, media_type="image/jpeg")
