import cv2
import numpy as np
import io
import os
import joblib
import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Zekâ Katmanı: Basit k-NN OCR ---
try:
    knn_model = joblib.load('api/knn_model.joblib')
    print("k-NN modeli başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenemedi: {e}")
    knn_model = None

def predict_digit(cell):
    # 1. Ön İşleme ve Normalizasyon
    cell_28 = cv2.resize(cell, (28, 28))
    cell_norm = cell_28 / 255.0 # 0-1 arası piksel değerleri
    
    # 2. Hücre Boş mu Kontrolü (Piksel Yoğunluğu)
    center_region = cell_norm[5:23, 5:23]
    if np.sum(center_region) < 15: # Bu eşik değeri kağıdın ışığına göre ayarlanabilir
        return 0 # Boş hücre
        
    if knn_model is None:
        return 5
    
    # 3. k-NN Tahmini
    # sklearn.datasets.load_digits 8x8 boyutlarındadır ve 0-16 arası değerler alır.
    cell_8x8 = cv2.resize(cell, (8, 8))
    cell_features = (cell_8x8 / 255.0 * 16.0).reshape(1, -1)
    
    prediction = knn_model.predict(cell_features)
    return int(prediction[0])

def get_perspective_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholding ile Sudoku çizgilerini patlat
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    biggest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)

    if len(approx) == 4:
        # Perspektif düzeltme (Warp Perspective)
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]

        dst = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, M, (450, 450))
        return warped
    return None

@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Sudoku Tahtasını Bul ve Düzelt (OpenCV)
        warped = get_perspective_transform(img)
        
        if warped is None:
            return {"status": "error", "message": "Sudoku tahtası bulunamadı. Lütfen dik ve sabit tutun."}

        # 2. Rakam Tanıma İçin İkinci Bir Threshold (Daha Net Rakamlar)
        # Rakam tanıma için siyah-beyaz (binary) görüntü şarttır.
        _, warped_thresh = cv2.threshold(warped, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. 81 Hücreye Böl ve Rakamları Oku (9x9)
        grid = []
        cell_size = 450 // 9

        for i in range(9):
            row = []
            for j in range(9):
                cell = warped_thresh[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                # Kenarlardaki çizgi gürültüsünü atmak için %10 crop
                h, w = cell.shape
                cell = cell[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
                
                digit = predict_digit(cell)
                row.append(digit)
            grid.append(row)
        
        return {"status": "success", "grid": grid}
    except Exception as e:
        return {"status": "error", "message": str(e)}