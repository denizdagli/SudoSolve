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
    Hücreyi temizle: Izgara çizgilerini kaldır, rakamı izole et ve Tesseract için hazırla.
    """
    # 1. Kenarlardan %15 kırp (Çizgi gürültüsünü azaltmak için)
    h, w = cell.shape
    margin_h, margin_w = int(h * 0.15), int(w * 0.15)
    cell = cell[margin_h:h-margin_h, margin_w:w-margin_w]

    # 2. Morfolojik işlemler: Küçük gürültüleri sil, rakam çizgilerini birleştir
    kernel = np.ones((2, 2), np.uint8)
    cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel) # Gürültü silme
    cell = cv2.dilate(cell, kernel, iterations=1) # Rakamı kalınlaştır

    # 3. En büyük bağlı bileşeni bul (Rakam)
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w_c, h_c = cv2.boundingRect(cnt)

    # İstatistiksel kontrol: Çok küçük veya orantısız yapılar rakam olamaz
    if w_c < 5 or h_c < 10 or h_c < w_c * 0.5:
        return None

    # Rakamı kes
    digit_crop = cell[y:y+h_c, x:x+w_c]

    # 4. Kareye ortala ve padding ekle (Tesseract için kritik)
    # Rakamın etrafında beyaz boşluk olması tanıman başarısını %50 artırır
    side = max(w_c, h_c) + 20
    digit_final = np.zeros((side, side), dtype=np.uint8)
    
    # Rakamı merkeze yerleştir
    start_x = (side - w_c) // 2
    start_y = (side - h_c) // 2
    digit_final[start_y:start_y+h_c, start_x:start_x+w_c] = digit_crop

    # Tesseract siyah zemin üzerinde beyaz rakamda zorlanabilir, tersine çevir
    digit_final = cv2.bitwise_not(digit_final)
    
    # 5. Keskinleştirme
    digit_final = cv2.GaussianBlur(digit_final, (3, 3), 0)
    
    return digit_final

def predict_digit(cell):
    """
    Tesseract kullanarak rakamı tanı. Farklı PSM modlarını deneyerek başarıyı artırır.
    """
    cleaned = clean_cell(cell)
    if cleaned is None:
        return 0

    # Farklı konfigürasyonları dene
    # PSM 10: Tek karakter, PSM 7: Tek satır (bazen daha iyi sonuç verir)
    configs = [
        r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789',
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=123456789'
    ]
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(cleaned, config=config)
            text = text.strip()
            if text and text.isdigit():
                val = int(text[0])
                if 1 <= val <= 9:
                    return val
        except:
            continue
            
    return 0

def get_perspective_transform(image):
    """
    Görüntüdeki Sudoku tahtasını bul ve 900x900 çözünürlüğe getir.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Izgarayı bulmak için adaptif threshold
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)

    # Kenarları biraz kalınlaştırarak ızgara çizgilerini birleştir
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000: # Çok küçük alanlar sudoku olamaz
            continue
            
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            # Daha yüksek çözünürlük (900px) OCR başarısını artırır
            side = 900 
            dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(gray, M, (side, side))
            return warped
            
    return None

@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    """
    Kamera veya yüklenen fotoğraftan Sudoku gridini oku.
    """
    try:
        # Görüntüyü oku
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Sudoku karesini bul ve perspektifi düzelt (900x900)
        warped = get_perspective_transform(img)
        if warped is None:
            return {"status": "error", "message": "Sudoku tahtası rsimde bulunamadı."}

        # 2. Rakam tanıma için ön işleme (Eşikleme)
        warped_blur = cv2.GaussianBlur(warped, (5, 5), 0)
        warped_thresh = cv2.adaptiveThreshold(warped_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)

        # 3. 81 hücreye böl ve rakamları tanı
        grid = []
        c_count = 0
        
        for r in range(9):
            row = []
            for c in range(9):
                # Hücre koordinatlarını belirle (900/9 = 100px)
                cell = warped_thresh[r*100 : (r+1)*100, c*100 : (c+1)*100]
                
                # Rakam tahmini yap
                val = predict_digit(cell)
                if val > 0:
                    c_count = c_count + 1
                row.append(val)
            grid.append(row)
        
        return {
            "status": "success",
            "grid": grid,
            "debug": {
                "detected": c_count,
                "res": "900x900"
            }
        }
    except Exception as err:
        print(f"HATA: {str(err)}")
        return {"status": "error", "message": str(err)}