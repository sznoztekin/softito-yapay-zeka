"""Spotify LSTM Sınıflandırıcı

"""

# ==============================================================================
# 🔐 SPOTIFY VERİ SETİ İLE PYTORCH LSTM SINIFLANDIRICI (BÖLÜM 5 & 6)
# ==============================================================================
# Bu kod, data.csv veri setinizi yükler, LSTM modelinin beklediği formatta
# hazırlar, modeli kurar, eğitir ve sonuçları görselleştirir.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 📊 ADIM 1: Veri Yükleme ve Önişleme (Preprocessing)
# ------------------------------------------------------------------------------
print("1. Veri seti yükleniyor ve hazırlanıyor...")

# data.csv dosyasını oku
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Hata: 'data.csv' dosyası bulunamadı! Lütfen dosyanın bu script ile aynı dizinde olduğundan emin olun.")
    exit()

# Eğitilecek sayısal öznitelikler (Özellikler)
feature_cols = [
    'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'key', 'liveness', 'loudness',
    'mode', 'speechiness', 'tempo', 'time_signature', 'valence'
]

X = df[feature_cols].values
y = df['target'].values  # 0: Şarkıyı sevmedi, 1: Şarkıyı sevdi

# Train-Test Ayrımı (%80 Eğitim, %20 Test)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standartlaştırma (Standardization)
# duration_ms veya tempo gibi çok büyük sayısal değerlerin, acousticness gibi
# 0-1 arası küçük değerleri ezmesini önlemek için veriyi ölçeklendiriyoruz.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# PyTorch Tensor formatına dönüştürme
# LSTM'in beklediği girdi şekli: (Batch Boyutu, Dizi Uzunluğu, Öznitelik Sayısı)
# Her şarkı tekil bir veri noktası olduğundan Dizi Uzunluğu'nu (seq_len) = 1 yapıyoruz.
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)

y_train_t = torch.tensor(y_train_raw, dtype=torch.long)
y_test_t = torch.tensor(y_test_raw, dtype=torch.long)

print(f" -> Eğitim Girdisi Şekli (X_train): {X_train_t.shape}") # (Batch, 1, 13)
print(f" -> Eğitim Etiket Şekli (y_train): {y_train_t.shape}")  # (Batch,)
print(f" -> Test Girdisi Şekli   (X_test) : {X_test_t.shape}")  # (Batch, 1, 13)
print("-" * 70)


# ------------------------------------------------------------------------------
# 🚀 ADIM 2: PyTorch nn.LSTM Sınıflandırıcı Modeli (Bölüm 5)
# ------------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """
    Şarkı özniteliklerini alıp kullanıcının beğenip beğenmeyeceğini tahmin eden LSTM modeli.
    Mimari: Girdi -> LSTM (2 Katman) -> Dropout -> Lineer Sınıflandırıcı
    """
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.3):
        super().__init__()

        # nn.LSTM parametreleri:
        # input_size : her adımda modele giren özellik sayısı (bizde 13)
        # hidden_size: LSTM hücrelerinin iç boyutu
        # num_layers : yığılmış katman sayısı (alt katmanın çıktısı üst katmana girer)
        # batch_first: True ise girdiler (batch, seq, feature) formatındadır
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0
        )

        # Eğitimde aşırı öğrenmeyi (overfitting) engellemek için Dropout katmanı
        self.dropout = nn.Dropout(dropout)

        # Son LSTM katmanının gizli durumunu (hidden_size) sınıf skorlarına (num_classes) çevirir
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # output : (batch, seq_len, hidden) -> her zaman adımının h_t çıktısı
        # h_n    : (num_layers, batch, hidden) -> en son adımın gizli durumu (hidden state)
        # c_n    : (num_layers, batch, hidden) -> en son adımın hücre durumu (cell state)
        output, (h_n, c_n) = self.lstm(x)

        # En son LSTM katmanının son zaman adımındaki gizli durumunu (h_T) seçiyoruz
        # h_n[-1] şekli: (batch, hidden_size) olur ve tüm seriyi/şarkıyı temsil eder.
        last_hidden = h_n[-1]

        # Dropout uygula (yalnızca model.train() modunda aktiftir)
        last_hidden = self.dropout(last_hidden)

        # Logit (ham skor) değerlerini hesapla
        logits = self.fc(last_hidden)
        return logits


# ------------------------------------------------------------------------------
# 🔄 ADIM 3: Eğitim Kurulumu ve Hiperparametreler (Bölüm 6)
# ------------------------------------------------------------------------------
INPUT_SIZE  = len(feature_cols)  # Girdi boyutu = 13
HIDDEN_SIZE = 64                 # LSTM gizli katman boyutu
NUM_CLASSES = 2                  # Sınıf sayısı = 2 (Beğendi / Beğenmedi)
EPOCHS      = 100                # Eğitim turu sayısı
LR          = 0.005              # Öğrenme oranı (Learning Rate)

# Modeli oluşturma
model_lstm = LSTMClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES,
    num_layers=2,
    dropout=0.2
)

# Kayıp fonksiyonu (İçeride otomatik Softmax içerir)
criterion = nn.CrossEntropyLoss()

# Adam optimizasyon algoritması (L2 regularizasyonu için weight_decay=1e-4)
optimizer = optim.Adam(model_lstm.parameters(), lr=LR, weight_decay=1e-4)

# StepLR: Her 25 epoch'ta öğrenme oranını %50 azaltır
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

# Metrikleri kaydetmek için geçmiş takibi
history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}


# ------------------------------------------------------------------------------
# 🔄 ADIM 4: Eğitim Döngüsü
# ------------------------------------------------------------------------------
print("Eğitim başlatılıyor...")
print(f"{'Epoch':>5} | {'Eğitim Kaybı':>12} | {'Eğitim Doğr.':>12} | {'Test Kaybı':>11} | {'Test Doğr.':>11} | {'LR':>8}")
print("─" * 75)

for epoch in range(1, EPOCHS + 1):

    # ① Modeli eğitim moduna al (Dropout aktifleşir)
    model_lstm.train()

    # ② Gradyanları sıfırla (PyTorch varsayılan olarak biriktirir)
    optimizer.zero_grad()

    # ③ İleri geçiş (Forward pass)
    logits = model_lstm(X_train_t)

    # ④ Kaybı hesapla (logits vs y_train)
    loss = criterion(logits, y_train_t)

    # ⑤ Geriye yayılım (Backpropagation)
    loss.backward()

    # ⑥ Gradient clipping (Patlayan gradyanları önlemek için kritik limit)
    nn.utils.clip_grad_norm_(model_lstm.parameters(), max_norm=1.0)

    # ⑦ Ağırlıkları güncelle
    optimizer.step()

    # ⑧ Öğrenme oranı zamanlayıcısını güncelle
    scheduler.step()

    # Değerlendirme (Validation / Testing)
    model_lstm.eval() # Modeli değerlendirme moduna al (Dropout kapanır)
    with torch.no_grad(): # Autograd (gradyan takibi) devre dışı bırakılır

        # Eğitim metrikleri
        train_preds = logits.argmax(dim=1)
        train_acc = (train_preds == y_train_t).float().mean().item() * 100

        # Test metrikleri
        test_logits = model_lstm(X_test_t)
        test_loss = criterion(test_logits, y_test_t).item()
        test_preds = test_logits.argmax(dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item() * 100

    # Metrikleri kaydet
    current_lr = scheduler.get_last_lr()[0]
    history["train_loss"].append(loss.item())
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    history["lr"].append(current_lr)

    # Her 10 epoch'ta bir gelişmeleri yazdır
    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:5d} | {loss.item():12.4f} | {train_acc:11.1f}% | {test_loss:11.4f} | {test_acc:10.1f}% | {current_lr:.5f}")

print("\nEğitim tamamlandı! ✓")


# ------------------------------------------------------------------------------
# 📈 ADIM 5: Grafik Çizimi (Eğitim Eğrileri)
# ------------------------------------------------------------------------------
print("Sonuçların grafiği çıkarılıyor...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Kayıp (Loss) Grafiği
ax1.plot(history["train_loss"], label="Eğitim Kaybı", color="#534AB7", lw=2)
ax1.plot(history["test_loss"], label="Test Kaybı", color="#E24B4A", lw=2)
ax1.set_title("CrossEntropy Kayıp Geçmişi", fontsize=12)
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Kayıp (Loss)", fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Doğruluk (Accuracy) Grafiği
ax2.plot(history["train_acc"], label="Eğitim Doğruluğu", color="#1D9E75", lw=2)
ax2.plot(history["test_acc"], label="Test Doğruluğu", color="#4AB7E2", lw=2)
ax2.axhline(50, color="gray", ls="--", alpha=0.6, label="Rastgele Tahmin (%50)")
ax2.set_title(f"Doğruluk (%) Geçmişi (En İyi Test Doğr.: {max(history['test_acc']):.1f}%)", fontsize=12)
ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("Doğruluk (%)", fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Spotify Veri Seti - LSTM Eğitim Metrikleri", fontsize=14, fontweight='bold')
plt.tight_layout()

# Grafiği yerel dizine kaydet
plt.savefig("spotify_lstm_results.png", dpi=120)
print("Eğitim grafiği 'spotify_lstm_results.png' olarak başarıyla kaydedildi! ✓")
plt.show()