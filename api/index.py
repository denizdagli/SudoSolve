import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_digit(cell):
    # 1. Hücreyi temizle ve standart boyuta getir (MNIST boyutu: 28x28)
    cell = cv2.resize(cell, (28, 28))
    
    # 2. Piksel Yoğunluğu Kontrolü (Hücre boş mu?)
    # Eğer siyah piksel sayısı çok azsa hücre boştur
    if np.sum(cell) < 500: # Eşik değeri (Sudoku kağıdına göre ayarlanabilir)
        return 0
    
    # 3. k-NN veya Basit Şablon Eşleştirme Modeli
    # Buraya ileride 'joblib.load' ile kendi eğitilmiş modelini ekleyebilirsin
    # joblib.load('api/knn_model.joblib')
    
    # Şimdilik: Test amaçlı dolu hücreye 1 dönelim.
    # Gerçek çözüm için bir MNIST modeli eğitip buraya entegre etmelisin.
    return 1 

def get_perspective_transform(image):
    # 1. Ön İşleme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # 2. Kontur Bulma
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
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

        # 1. Sudoku tahtasını bul ve düzelt
        warped = get_perspective_transform(img)
        
        if warped is None:
            return {"status": "error", "message": "Sudoku bulunamadı. Lütfen kamerayı sabit tutun."}

        # 2. 81 Hücreye Böl ve Rakamları Tanı (9x9)
        grid = []
        # Hücreleri Adaptive Threshold ile siyah-beyaz yap (OCR başarısı için şart)
        thresh_warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 11, 2)

        for i in range(9):
            row = []
            for j in range(9):
                cell = thresh_warped[i*50:(i+1)*50, j*50:(j+1)*50]
                # Kenarlardaki çizgi gürültüsünü atmak için crop
                cell = cell[5:45, 5:45] 
                digit = predict_digit(cell)
                row.append(digit)
            grid.append(row)
        
        return {"status": "success", "grid": grid}
    except Exception as e:
        return {"status": "error", "message": str(e)}