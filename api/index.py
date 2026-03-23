import cv2
import numpy as np
import os
import json
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Gemini Yapılandırması
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Kullanıcı tarafından belirtilen özel model
MODEL_NAME = "gemini-3.1-pro-preview"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_perspective_transform(image):
    """
    Görüntüdeki Sudoku tahtasını bul ve perspektifini düzelt.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000:
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

            side = 900 
            dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (side, side)) # Orijinal renkli görüntüyü warp et
            return warped
            
    return None

async def gemini_ocr(image_bytes):
    """
    Gemini API kullanarak Sudoku gridini tanı.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        prompt = """
        Bu bir Sudoku bulmacası görüntüsüdür. Lütfen bu görüntüdeki Sudoku gridini analiz et ve rakamları bir JSON dizisi olarak döndür.
        Boş hücreler için 0 kullan.
        Sadece 9x9'luk bir integer dizisi (list of lists) içeren saf bir JSON döndür. 
        Örnek format: [[5,3,0,0,7,0,0,0,0], ...] 
        Markdown veya açıklama ekleme, sadece JSON.
        """
        
        # Görüntüyü hazırla
        contents = [
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
        ]
        
        response = model.generate_content(contents)
        text_response = response.text.strip()
        
        # Markdown kod bloklarını temizle (eğer varsa)
        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        elif "```" in text_response:
            text_response = text_response.split("```")[1].split("```")[0].strip()
            
        grid = json.loads(text_response)
        return grid
    except Exception as e:
        print(f"Gemini OCR Hatası: {e}")
        return None

@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    """
    Kamera veya yüklenen fotoğraftan Sudoku gridini Gemini ile oku.
    """
    try:
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Sudoku karesini bul ve perspektifi düzelt
        # (Gemini'nin daha odaklı çalışması için görüntüyü crop ediyoruz)
        warped = get_perspective_transform(img)
        
        # Eğer perspektif bulunamazsa orijinal görüntüyü kullan (Gemini yine de çözebilir)
        if warped is not None:
            _, buffer = cv2.imencode(".jpg", warped)
        else:
            _, buffer = cv2.imencode(".jpg", img)
            
        # 2. Gemini ile Tanıma
        grid = await gemini_ocr(buffer.tobytes())
        
        if grid is None:
            return {"status": "error", "message": "Gemini rakamları tanıyamadı."}

        # Kaç adet rakam okundu?
        detected = sum(1 for row in grid for d in row if d > 0)
        
        return {
            "status": "success",
            "grid": grid,
            "debug": {
                "detected": detected,
                "model": MODEL_NAME
            }
        }
    except Exception as err:
        print(f"HATA: {str(err)}")
        return {"status": "error", "message": str(err)}