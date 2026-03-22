from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io

app = FastAPI()

# Frontend ile iletişim için CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_for_sudoku(image):
    # 1. Gri tonlama
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur ile gürültü temizleme
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Adaptive Thresholding (Sayıları ve çizgileri belirginleştirir)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

@app.post("/api/scan")
async def scan_sudoku(file: UploadFile = File(...)):
    try:
        # Fotoğrafı oku
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"status": "error", "message": "Image could not be decoded"}

        # Görüntü işleme adımını çalıştır
        processed = preprocess_for_sudoku(image)
        
        # NOT: Burada ileride perspektif düzeltme ve 81 hücreye bölme yapılacak.
        # Şimdilik frontend'e boş bir grid dönüyoruz ki sistemin çalıştığını görelim.
        empty_grid = [[0 for _ in range(9)] for _ in range(9)]
        
        return {
            "status": "success",
            "grid": empty_grid,
            "message": "Python backend received the image successfully!"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/health")
def health():
    return {"status": "working"}