import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def process_strawberry_grid(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # 1. 背景を白黒化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. 青いピンを探してトレイの範囲を特定
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(blue_cnts) < 2:
        return image_bytes # ピンが見つからない場合はそのまま返す

    # ピンの範囲からグリッドを作成
    all_pts = np.concatenate(blue_cnts)
    bx, by, bw, bh = cv2.boundingRect(all_pts)
    
    # 換算係数
    px_to_mm = 320 / bw
    
    # 7列(横) x 13行(縦) の各マスの中心を計算
    cols = 7
    rows = 13
    cell_w = bw / cols
    cell_h = bh / rows

    # 赤色の抽出
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # 91マスを順番にスキャン
    for r in range(rows):
        for c in range(cols):
            # マスの中心座標
            center_x = int(bx + (c + 0.5) * cell_w)
            center_y = int(by + (r + 0.5) * cell_h)
            
            # マス目周囲の狭い範囲（ROI）を切り出す
            roi_size_h = int(cell_h * 0.8)
            roi_size_w = int(cell_w * 0.8)
            y1, y2 = max(0, center_y - roi_size_h//2), min(h, center_y + roi_size_h//2)
            x1, x2 = max(0, center_x - roi_size_w//2), min(w, center_x + roi_size_w//2)
            
            roi_mask = red_mask[y1:y2, x1:x2]
            cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                # そのマスの中で一番大きい赤い塊をイチゴとする
                largest_cnt = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(largest_cnt) > 500:
                    _, (pw, ph), _ = cv2.minAreaRect(largest_cnt)
                    size_mm = max(pw, ph) * px_to_mm
                    
                    # サイズ判定
                    if size_mm >= 35: label, color = "L", (0, 0, 255)
                    elif size_mm >= 28: label, color = "M", (255, 0, 0)
                    else: label, color = "S", (0, 255, 0)

                    # --- 巨大文字描画 (縁取り付き) ---
                    font = cv2.FONT_HERSHEY_PLAIN
                    font_scale = 5.0  # さらに大きく
                    thickness = 10
                    # 背景に黒い縁取り（視認性アップ）
                    cv2.putText(output_img, label, (center_x-40, center_y+20), font, font_scale, (0,0,0), thickness+5)
                    # メインの文字
                    cv2.putText(output_img, label, (center_x-40, center_y+20), font, font_scale, color, thickness)

    _, encoded_img = cv2.imencode(".jpg", output_img)
    return encoded_img.tobytes()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    result = process_strawberry_grid(contents)
    return Response(content=result, media_type="image/jpeg")
