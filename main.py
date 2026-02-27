import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def process_strawberry_final(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # --- 1. 背景を白黒化 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- 2. トレイの内側だけを抽出する（青いピンを利用） ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 青色（ピン）の範囲
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ピンが見つかれば、その範囲をマスク（見つからなければ全体）
    analysis_mask = np.zeros((h, w), dtype=np.uint8)
    if len(blue_cnts) >= 2:
        # ピンを結ぶ最小の外接矩形を作成
        all_pts = np.concatenate(blue_cnts)
        bx, by, bw, bh = cv2.boundingRect(all_pts)
        cv2.rectangle(analysis_mask, (bx, by), (bx+bw, by+bh), 255, -1)
    else:
        analysis_mask[:] = 255 # ピンがない場合は全体（念のため）

    # --- 3. イチゴの赤色を正確に抽出 ---
    # 赤色は0付近と180付近に分かれるため両方結合
    lower_red1, upper_red1 = np.array([0, 100, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 100, 70]), np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # トレイ内側のみに限定
    final_mask = cv2.bitwise_and(red_mask, analysis_mask)

    # ノイズ除去（膨張と収縮）
    kernel = np.ones((5,5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    # --- 4. 判定と描画 ---
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 換算係数（トレイの内側幅を320mmとする）
    # ピンが見つかっている場合はその幅(bw)を基準にする
    target_width_px = bw if len(blue_cnts) >= 2 else w
    px_to_mm = 320 / target_width_px

    for cnt in contours:
        if cv2.contourArea(cnt) < 1500: continue # 小さすぎるのは無視

        # 最小外接矩形でサイズ取得
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (width_px, height_px), angle = rect
        strawberry_size_mm = max(width_px, height_px) * px_to_mm

        # 判定
        if strawberry_size_mm >= 35:
            label, color = "L", (0, 0, 255) # 赤
        elif strawberry_size_mm >= 28:
            label, color = "M", (255, 0, 0) # 青
        else:
            label, color = "S", (0, 255, 0) # 緑

        # 文字描画
        font = cv2.FONT_HERSHEY_DUPLEX
        fs, thick = 3.0, 7
        t_size = cv2.getTextSize(label, font, fs, thick)[0]
        cv2.putText(output_img, label, (int(cx - t_size[0]/2), int(cy + t_size[1]/2)), 
                    font, fs, color, thick)

    _, encoded_img = cv2.imencode(".jpg", output_img)
    return encoded_img.tobytes()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    result = process_strawberry_final(contents)
    return Response(content=result, media_type="image/jpeg")
