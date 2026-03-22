from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_perspective_transform(image):
    # 1. Ön İşleme: Gri Tonlama ve Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # 2. Kontur Bulma (Sudoku Karesini Tespit Etme)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # En büyük dörtgeni bul
    biggest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)

    if len(approx) == 4:
        # Köşe noktalarını sırala (sol-üst, sağ-üst, sağ-alt, sol-alt)
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Kuşu bakışı perspektife çevir (450x450 px standart kare)
        dst = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, M, (450, 450))
        return warped
    return None

@app.post("/api/scan")
async def scan_sudoku(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Sudoku tahtasını bul ve düzelt
        warped = get_perspective_transform(img)
        
        if warped is None:
            return {"status": "error", "message": "Sudoku board not found. Please hold the camera steady."}

        # 2. 81 Hücreye Böl (9x9)
        grid = [[0 for _ in range(9)] for _ in range(9)]
        cell_size = 450 // 9

        # Şimdilik sadece bölüyoruz, bir sonraki adımda sayıları tanıyacağız
        # OpenCV ile her hücreyi analiz edip sayı olup olmadığını kontrol edeceğiz
        
        return {
            "status": "success", 
            "grid": grid, 
            "message": "Board detected! Ready for digit recognition."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}