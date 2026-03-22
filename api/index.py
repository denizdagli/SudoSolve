import cv2
import numpy as np
import io
import os
import joblib # YENİ: Model yüklemek için
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Zekâ Katmanı: k-NN OCR Entegrasyonu ---
# Vercel sunucusunda model dosyasının yolunu bul (Relative path)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'knn_model.joblib')

# Modeli yüklüyoruz (Cachelemek için globalde tutuyoruz)
knn = None
if os.path.exists(MODEL_PATH):
    knn = joblib.load(MODEL_PATH)
    print("Model başarıyla yüklendi!")
else:
    print(f"HATA: Model dosyası bulunamadı! Yol: {MODEL_PATH}")

def predict_digit(cell):
    global knn
    # 1. Ön İşleme: Hücreyi modelin beklediği formata (28x28 grayscale) getir ve normalize et
    # 28x28 MNIST fontları için standart boyuttur.
    # Not: Modelin eğitim verisi (datasets.load_digits()) 8x8 veya 28x28 olabilir.
    # Bizim train_knn.py dosyamız load_digits() kullandığı için 8x8 resize yapmalıyız.
    cell = cv2.resize(cell, (8, 8)) # train_knn.py'daki verisetine göre (load_digits=8x8)
    cell_norm = (cell / 255.0) * 16.0 # load_digits veriseti 0-16 arası piksel değerleri bekler.
    
    # 2. Hücre Boş mu Kontrolü (Piksel Yoğunluğu)
    # MNIST fontları merkezde olduğu için hücrenin ortasına odaklanalım.
    center_region = cell_norm[1:7, 1:7]
    if np.sum(center_region) < 25: # Bu eşik değeri kağıdın ışığına göre ayarlanabilir
        return 0 # Boş hücre
    
    # 3. k-NN Tahmini
    if knn is None:
        # Model yoksa test amaçlı dolu hücreye 1 dönmeye devam edelim (Test için)
        return 1
    
    # Veriyi modelin beklediği (1, 64) formatına çevir
    cell_final = cell_norm.reshape(1, -1)
    
    # Tahmin yürüt
    prediction = knn.predict(cell_final)
    
    return int(prediction[0]) 

def get_perspective_transform(image):
    # (Önceki OpenCV Perspektif kodu aynen kalıyor)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
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
            return {"status": "error", "message": "Sudoku bulunamadı. Lütfen kamerayı sabit tutun."}

        # 2. Rakam Tanıma İçin İkinci Bir Threshold (Binary)
        # Rakam tanıma için siyah-beyaz görüntü şarttır.
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