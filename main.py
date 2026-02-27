import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# GitHub Pagesからのアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_strawberry_visual(image_bytes):
    # バイナリデータをOpenCV形式に変換
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # --- 1. 背景の白黒化 ---
    # 元画像をグレースケールにし、再度3チャンネルに戻す
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 基準となる換算係数 (32cm x 57cmトレイ前提)
    CONTAINER_W_MM = 320
    # 画像全体に対する換算係数を計算
    px_to_mm = CONTAINER_W_MM / img.shape[1]

    # --- 2. イチゴの検出 (元のカラー画像で行う) ---
    # 赤色抽出用のHSV変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 60, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 60, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 小さすぎる領域を除外 (ゴミなど)
        area = cv2.contourArea(cnt)
        if area < 1000: continue 

        # --- 3. サイズ判定 ---
        # 傾いたイチゴにも対応できる最小外接矩形を取得
        rect = cv2.minAreaRect(cnt)
        (x_c, y_c), (w, h), angle = rect
        
        # 横幅（長い方）を実寸に換算
        strawberry_width_mm = max(w, h) * px_to_mm
        
        # L/M/S 判定と文字色の設定
        if strawberry_width_mm >= 35:
            label, color = "L", (0, 0, 255)   # 赤色文字
        elif strawberry_width_mm >= 28:
            label, color = "M", (255, 0, 0)   # 青色文字
        else:
            label, color = "S", (0, 255, 0)   # 緑色文字

        # --- 4. 白黒画像への大きな文字描画 ---
        # 重心座標を取得
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # フォント設定 (大きな文字、太字)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3.5  # かなり大きく
            thickness = 8     # かなり太く
            
            # 文字のサイズを取得して中央寄せ
            text_size, _ = cv2.getTextSize(label, font_face, font_scale, thickness)
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            
            # 白黒の背景に文字を直接書き込む (枠は描画しない)
            cv2.putText(output_img, label, (text_x, text_y), 
                        font_face, font_scale, color, thickness)

    # --- 5. 画像をJPEGに戻して返す ---
    _, encoded_img = cv2.imencode(".jpg", output_img)
    return encoded_img.tobytes()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 画像ファイルを受け取る
    contents = await file.read()
    # 視覚化処理を実行
    result_image = process_strawberry_visual(contents)
    # 処理後の画像をResponseとして撃ち返す
    return Response(content=result_image, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    # ポート10000はRenderのデフォルト
    uvicorn.run(app, host="0.0.0.0", port=10000)
