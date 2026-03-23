import cv2
import numpy as np
import io
import os
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_cell(cell):
    """
    Hücreyi temizle: Kenar gürültülerini kaldır, rakamı ortala ve Tesseract için hazırla.
    """
    # 1. Kenarlardaki gürültüleri (ızgara çizgileri) temizlemek için %15 crop
    h, w = cell.shape
    margin_h, margin_w = int(h * 0.15), int(w * 0.15)
    cell = cell[margin_h:h-margin_h, margin_w:w-margin_w]

    # 2. Bağlı bileşen analizi ile en büyük 'rakam' olabilecek parçayı bul
    # (Bu, küçük gürültüleri tamamen temizler)
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # En büyük konturu bul (muhtemelen rakam)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w_c, h_c = cv2.boundingRect(cnt)

    # Çok küçük parçalar rakam değildir
    if w_c < 5 or h_c < 10:
        return None

    # Rakamı kesip al
    digit_crop = cell[y:y+h_c, x:x+w_c]

    # 3. Rakamı 28x28 veya benzeri bir kareye ortala (Beyaz padding ile)
    # Tesseract için etrafında biraz boşluk olması iyidir
    pad = 10
    digit_padded = cv2.copyMakeBorder(digit_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    # Görüntüyü tersine çevir (Tesseract beyaz zemin üzerinde siyah rakam sever)
    digit_final = cv2.bitwise_not(digit_padded)
    
    return digit_final

def predict_digit(cell):
    """
    Tesseract kullanarak rakamı tanı.
    """
    cleaned = clean_cell(cell)
    if cleaned is None:
        return 0

    # Tesseract konfigürasyonu: Sadece rakamlar (1-9), tek karakter modu (PSM 10)
    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
    
    try:
        text = pytesseract.image_to_string(cleaned, config=custom_config)
        text = text.strip()
        if text and text.isdigit():
            return int(text[0])
    except Exception as e:
        print(f"OCR Hatası: {e}")
        
    return 0

def get_perspective_transform(image):
    """
    Görüntüdeki en büyük kareyi (Sudoku tahtası) bul ve perspektifini düzelt.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptif threshold ile ızgarayı belirginleştir
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Alan büyüklüğüne göre sırala
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            # 4 köşeli bir yapı bulduk (Sudoku tahtası)
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            # Köşeleri sırala: [sol-üst, sağ-üst, sağ-alt, sol-alt]
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            side = 450
            dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(gray, M, (side, side))
            return warped
            
    return None

@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Sudoku Tahtasını Bul ve Düzelt
        warped = get_perspective_transform(img)
        
        if warped is None:
            return {"status": "error", "message": "Sudoku bulunamadı. Lütfen tahtayı tam kareye alın."}

        # 2. Rakam Tanıma İçin Threshold
        # Otsu thresholding daha iyi sonuç verebilir
        blur_warped = cv2.GaussianBlur(warped, (3, 3), 0)
        thresh_warped = cv2.adaptiveThreshold(blur_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)

        # 3. 81 Hücreye Böl ve Rakamları Oku (9x9)
        grid = []
        cell_size = 450 // 9

        for i in range(9):
            row = []
            for j in range(9):
                # Hücreyi kes
                cell = thresh_warped[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                digit = predict_digit(cell)
                row.append(digit)
            grid.append(row)
        
        return {"status": "success", "grid": grid}
    except Exception as e:
        return {"status": "error", "message": str(e)}