import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

# GitHub Pagesからのアクセスを許可する設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_strawberry(image_bytes):
    # バイナリデータをOpenCV形式に変換
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. コンテナのサイズ基準 (32cm x 57cm)
    # 本来は青いピンで射影変換すべきですが、まずは簡易版として画像幅で計算
    CONTAINER_W_MM = 320
    px_to_mm = CONTAINER_W_MM / img.shape[1]

    # 2. 赤色を抽出 (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # 3. 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: continue # 小さすぎるゴミを除外

        x, y, w, h = cv2.boundingRect(cnt)
        width_mm = w * px_to_mm
        
        # サイズ判定
        if width_mm >= 35:
            label, color = f"L:{width_mm:.1f}", (0, 0, 255) # 赤
        elif width_mm >= 28:
            label, color = f"M:{width_mm:.1f}", (255, 0, 0) # 青
        else:
            label, color = f"S:{width_mm:.1f}", (0, 255, 0) # 緑

        # 描画
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # 4. 画像をJPEGに戻して返す
    _, encoded_img = cv2.imencode(".jpg", img)
    return encoded_img.tobytes()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    result_image = process_strawberry(contents)
    return Response(content=result_image, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
