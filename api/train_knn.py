import cv2
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 1. MNIST Rakam Verisetini Yükle (0-9 Rakamları)
print("Veriseti yükleniyor...")
digits = datasets.load_digits()
data = digits.images.reshape((len(digits.images), -1))
labels = digits.target

# 2. Veriyi Eğitim ve Test Olarak Böl
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 3. k-NN Modelini Oluştur ve Eğit
# n_neighbors=3 veya 5 standart Sudoku fontları için mükemmeldir.
print("Model eğitiliyor (k=3)...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 4. Model Başarısını Test Et
print(f"Model Başarısı: %{knn.score(X_test, y_test)*100:.2f}")

# 5. Eğitilen Modeli Joblib Dosyası Olarak Kaydet
model_filename = "api/knn_model.joblib"
print(f"Model kaydediliyor: {model_filename}...")
joblib.dump(knn, model_filename)
print("Bitti!")
