import cv2
import numpy as np
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
# Not: Vercel'de joblib dosyası yüklemekle uğraşmamak için
# modelin ağırlıklarını doğrudan kodun içine gömüyoruz (Template Matching).

# Basit rakam şablonları (0: Boş, 1-9: Rakamlar)
# Bu şablonlar, 28x28 MNIST fontlarını temsil eder.
def get_digit_templates():
    # Bu kısım basitleştirilmiştir. Gerçek bir model eğitim verisi
    # veya joblib dosyası buraya gelmelidir.
    # Şimdilik: Eğer hücre doluysa basit bir tahmin yürüten mantık kuruyoruz.
    pass

def predict_digit(cell):
    # 1. Ön İşleme: Hücreyi 28x28 boyutuna getir ve normalize et
    cell = cv2.resize(cell, (28, 28))
    cell_norm = cell / 255.0 # 0-1 arası piksel değerleri
    
    # 2. Hücre Boş mu Kontrolü (Piksel Yoğunluğu)
    # MNIST fontları merkezde olduğu için hücrenin ortasına odaklanalım.
    center_region = cell_norm[5:23, 5:23]
    if np.sum(center_region) < 15: # Bu eşik değeri kağıdın ışığına göre ayarlanabilir
        return 0 # Boş hücre
    
    # 3. k-NN Tahmini (Template Matching)
    # Burada eğitilmiş bir modelin load edilmesi (joblib.load) en doğrusudur.
    # Şimdilik: Test Sudoku'ndaki fontlara göre bir tahmin yürütelim.
    # Gerçek çözüm için: Lütfen eğitilmiş bir knn_model.joblib dosyasını 
    # 'api/' klasörüne ekle ve 'joblib.load' ile yükle.
    
    # (ÖNEMLİ: Bu kısım geçici bir 'mock' tahmindir. Gerçek OCR için model şarttır.)
    # Şimdilik her dolu hücreye test amaçlı 5 dönelim (1'lerin değiştiğini görmek için).
    return 5 

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