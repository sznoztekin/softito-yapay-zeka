"""RNN.ipynb

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

# ==========================================
# 1. VERİ SETİ VE ÖN İŞLEME
# ==========================================
yorumlar = [
    "kargo çok hızlı geldi paketleme harikaydı", "kumaşı çok kalitesiz dikişleri sökük geldi",
    "ürün güzel ama bedeni uymadı iade edeceğim", "tam bir fiyat performans ürünü çok beğendim",
    "hayatımda gördüğüm en kötü paketleme kalitesiz ürün", "beklediğimden çok daha iyi çıktı kesinlikle tavsiye ederim",
    "tasarımı güzel ama malzemesi çok basit", "hızlı teslimat ve kaliteli ürün teşekkürler",
    "rezalet bir hizmet bir daha asla almam", "fiyatı pahalı ama kalitesi idare eder"
]
# 1: Olumlu, 0: Olumsuz
etiketler = np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.float32)

vocab_size = 1000
max_length = 8
embedding_dim = 16

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(yorumlar)
sequences = tokenizer.texts_to_sequences(yorumlar)
X_pad = pad_sequences(sequences, maxlen=max_length, padding='post')

# ==========================================
# 2. RNN MODEL MİMARİSİ
# ==========================================
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    SimpleRNN(units=32, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 3. MODELİN EĞİTİLMESİ (GRAFİK İÇİN GÜNCELLENDİ)
# ==========================================
print("\n--- Model Eğitiliyor ve Geçmiş Kaydediliyor ---")
# Epoch sayısını grafikte değişimi net görebilmek için 30'a çıkardık
# 'history' değişkeni eğitim boyunca oluşan metrikleri saklar
history = model.fit(X_pad, etiketler, epochs=30, batch_size=2, verbose=0)
print("Eğitim tamamlandı!")

# ==========================================
# 4. MATPLOTLIB İLE GRAFİKLERİN OLUŞTURULMASI
# ==========================================
# Yan yana iki grafik oluşturuyoruz: Biri Kayıp (Loss), diğeri Doğruluk (Accuracy)
plt.figure(figsize=(14, 5))

# --- Grafik 1: Loss (Kayıp) Grafiği ---
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı (Loss)', color='red', linewidth=2)
plt.title('Epoch Başına Model Kaybı (Loss)')
plt.xlabel('Epoch (Adım)')
plt.ylabel('Kayıp (Loss)')
plt.legend()
plt.grid(True)

# --- Grafik 2: Accuracy (Doğruluk) Grafiği ---
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu (Accuracy)', color='blue', linewidth=2)
plt.title('Epoch Başına Model Doğruluğu')
plt.xlabel('Epoch (Adım)')
plt.ylabel('Doğruluk (Accuracy)')
plt.legend()
plt.grid(True)

# Grafikleri ekrana yansıtma
plt.tight_layout()
plt.show()