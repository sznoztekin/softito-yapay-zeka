"""görüntü_işleme_panda.ipynb

"""

# Gerekli kütüphaneleri içe aktarıyoruz
import numpy as np                          # Sayısal dizi işlemleri için temel kütüphane
import matplotlib.pyplot as plt             # Grafik ve görüntü çizimi için
import matplotlib.gridspec as gridspec      # Karmaşık subplot düzeni için

import cv2                                  # OpenCV — görüntü işlemenin standart kütüphanesi
from PIL import Image, ImageFilter          # Pillow — Python'ın görüntü kütüphanesi
from PIL import ImageEnhance               # Parlaklık/kontrast ayarları için

from scipy import datasets as scd           # Scipy'den gerçek test görüntüleri
from scipy.ndimage import (                 # Scipy ile görüntü filtresi işlemleri
    gaussian_filter,
    sobel,
    median_filter
)

from sklearn.datasets import load_digits    # Sklearn'den el yazısı rakam veri seti
from sklearn.model_selection import train_test_split   # Eğitim/test bölme
from sklearn.preprocessing import StandardScaler        # Özellik normalizasyonu
from sklearn.neighbors import KNeighborsClassifier      # KNN sınıflandırıcı
from sklearn.metrics import classification_report, confusion_matrix  # Değerlendirme

import warnings
warnings.filterwarnings('ignore')           # Gereksiz uyarı mesajlarını gizle

# Matplotlib ayarları — retina ekranlar için yüksek DPI
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'DejaVu Sans'

print("✅ Tüm kütüphaneler başarıyla yüklendi!")
print(f"   NumPy:      {np.__version__}")
print(f"   OpenCV:     {cv2.__version__}")

# 1. Adım: panda.jpeg resmini yükle ve gri tonlamaya (L modu) çevir
resim_yolu = "panda.jpeg"
img_raw = Image.open(resim_yolu).convert("L")

# 2. Adım: Resmi filtreleme işlemleri için numpy float64 matrisine dönüştür
img_gray = np.array(img_raw).astype(np.float64)

# 3. Adım: Yüklenen pandanın özelliklerini kontrol et
print("--- Panda Resmi Bilgileri ---")
print(f"Çözünürlük (Piksel Sayısı) : {img_gray.shape}")  # Satır x Sütun (Yükseklik x Genişlik)
print(f"Veri Tipi                  : {img_gray.dtype}")  # float64
print(f"En Karanlık Nokta (Min)    : {img_gray.min()}")
print(f"En Aydınlık Nokta (Max)    : {img_gray.max()}")

# 4. Adım: Resmi Ekrana Çizdir
plt.figure(figsize=(6, 6))
plt.imshow(img_gray, cmap="gray")
plt.title("Orijinal Panda Resminiz (Gri Tonlamalı)")
plt.axis("off")  # Kenardaki koordinat çizgilerini kapatır
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Sol Grafik: Orijinal Panda Resminiz ───────────────────────────────────────
axes[0].imshow(img_gray, cmap="gray")  # Gri tonlama için cmap='gray'
axes[0].set_title(
    f"🐼 Orijinal Panda Resmi\n{img_gray.shape[0]}×{img_gray.shape[1]} px, Gri Tonlama",
    fontsize=11,
)
axes[0].axis("off")  # Eksen çizgilerini gizle

# ── Sağ Grafik: Resmin Piksel Histogramı (Görüntü İşleme İçin Çok Önemlidir) ──
# 2 boyutlu resim matrisini tek boyuta (ravel) döküp piksellerin (0-255 arası) dağılımına bakıyoruz
axes[1].hist(img_gray.ravel(), bins=256, range=(0, 256), color="black", alpha=0.7)
axes[1].set_title("📊 Piksel Yoğunluk Dağılımı (Histogram)", fontsize=11)
axes[1].set_xlabel("Piksel Değeri (0: Siyah, 255: Beyaz)")
axes[1].set_ylabel("Piksel Sayısı")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

plt.suptitle(
    "📂 Yüklenen Panda Resmi ve Analizi", fontsize=14, fontweight="bold", y=1.05
)
plt.tight_layout()
plt.show()

# ── Görüntü boyutunu incele ───────────────────────────────────────────────────
# Gri resimlerde shape sadece (Yükseklik, Genişlik) döndürür.
H, W = img_gray.shape

print("=== Panda Görüntü Boyut Bilgisi ===")
print(f"Yükseklik (H): {H} piksel")    # Satır sayısı
print(f"Genişlik  (W): {W} piksel")    # Sütun sayısı
print(f"Kanal sayısı : 1 (Gri Tonlamalı - Tek Kanal)")
print(f"Toplam piksel: {H * W:,}")     # Gri resimde toplam piksel = H * W
print(f"Veri tipi    : {img_gray.dtype}")   # float64 (filtrelere hazır format)
print(f"Min değer    : {img_gray.min():.1f}")   # En karanlık piksel (0'a yakın)
print(f"Max değer    : {img_gray.max():.1f}")   # En parlak piksel (255'e yakın)
print(f"Ortalama     : {img_gray.mean():.1f}")  # Resmin genel ortalama parlaklığı

resim_yolu = "panda.jpeg"
img_color = np.array(Image.open(resim_yolu).convert("RGB")).astype(np.float64)

# 2. Adım: Grafik alanını oluştur (1 satır, 4 sütun)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# ── 1. Grafik: Orijinal Görüntü ──────────────────────────────────────────────
# imshow() float64 verileri gösterirken 0-1 aralığı bekler veya uint8 ister.
# Düzgün görünmesi için geçici olarak uint8'e çevirip çizdiriyoruz.
axes[0].imshow(img_color.astype(np.uint8))
axes[0].set_title("🎨 Orijinal Panda (RGB)")
axes[0].axis("off")

# ── 2, 3 ve 4. Grafikler: Kanalları Ayrıştır ──────────────────────────────────
# Renkli resimde: [satır, sütun, kanal_indeksi]
channel_info = [
    (img_color[:, :, 0], "Reds", "R (Kırmızı) Kanalı"),  # İndeks 0 = Red
    (img_color[:, :, 1], "Greens", "G (Yeşil) Kanalı"),  # İndeks 1 = Green
    (img_color[:, :, 2], "Blues", "B (Mavi) Kanalı"),  # İndeks 2 = Blue
]

for ax, (channel, cmap, title) in zip(axes[1:], channel_info):
    ax.imshow(channel, cmap=cmap)  # Tek kanalı ilgili renk haritasıyla göster
    ax.set_title(f"{title}\nOrtalama Parlaklık: {channel.mean():.1f}")
    ax.axis("off")

plt.suptitle(
    "Panda RGB Kanal Ayrıştırması — Her kanal bağımsız bir gri matrisidir",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# ── 1. Adım: Dinamik Olarak Koordinatları Belirle ────────────────────────────
H, W, C = img_color.shape  # Pandanın güncel yükseklik ve genişliğini al

mid_h, mid_w = int(H / 2), int(W / 2)  # Merkez piksel koordinatları
max_h, max_w = H - 1, W - 1  # Sağ alt köşe (indeksler 0'dan başladığı için -1)

# ── 2. Adım: Belirli Piksel Değerlerini İncele ────────────────────────────────
print("=== Panda Resminin Belirli Piksellerindeki RGB Değerleri ===")
print(
    f"  Sol üst köşe [0, 0]        : R={img_color[0, 0, 0]:.0f}, G={img_color[0, 0, 1]:.0f}, B={img_color[0, 0, 2]:.0f}"
)
print(
    f"  Merkez       [{mid_h}, {mid_w}] : R={img_color[mid_h, mid_w, 0]:.0f}, G={img_color[mid_h, mid_w, 1]:.0f}, B={img_color[mid_h, mid_w, 2]:.0f}"
)
print(
    f"  Sağ alt köşe [{max_h}, {max_w}]: R={img_color[max_h, max_w, 0]:.0f}, G={img_color[max_h, max_w, 1]:.0f}, B={img_color[max_h, max_w, 2]:.0f}"
)

# ── 3. Adım: Pandadan Dinamik Bir Bölge Kırp (Zoom Effect) ────────────────────
# Sabit değerler yerine resmin merkezinden 50x50 piksellik güvenli bir alan kırpalım
r_start_h, r_end_h = mid_h - 25, mid_h + 25
r_start_w, r_end_w = mid_w - 25, mid_w + 25

region = img_color[
    r_start_h:r_end_h, r_start_w:r_end_w
]  # [satır_aralığı, sütun_aralığı]
print(f"\nKırpılan panda bölgesi şekli: {region.shape}")

# ── 4. Adım: Görselleştirme ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Sol Grafik: Orijinal resim ve üzerine çizilen kırpma kutusu
ax1.imshow(img_color.astype("uint8"))
# Rectangle parametreleri: (Sol alt köşe X, Sol alt köşe Y), Genişlik, Yükseklik
rect = plt.Rectangle(
    (r_start_w, r_start_h),
    50,
    50,
    linewidth=2,
    edgecolor="yellow",
    facecolor="none",
)
ax1.add_patch(rect)  # Sarı kutuyu orijinal resmin üzerine ekle
ax1.set_title("Orijinal Panda — Sarı bölge kırpılacak")
ax1.axis("off")

# Sağ Grafik: Sadece kırpılan (zoom yapılan) bölge
ax2.imshow(region.astype("uint8"))
ax2.set_title(f"Kırpılmış Bölge\n{region.shape[0]}×{region.shape[1]} piksel")
ax2.axis("off")

plt.tight_layout()
plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Önceki adımlardan gelen img_color'ın boyutlarını alalım
H, W, C = img_color.shape

# ── 1. Kırpma (Cropping) ──────────────────────────────────────────────────────
# Resmin tam ortasından dikeyde %50, yatayda %50'lik bir alanı dinamik olarak kırpalım
y_start, y_end = int(H * 0.25), int(H * 0.75)
x_start, x_end = int(W * 0.25), int(W * 0.75)

crop = img_color[y_start:y_end, x_start:x_end]
print(f"Kırpma sonrası şekil: {crop.shape}")

# ── 2. Yeniden Boyutlandırma (Resize) ─────────────────────────────────────────
# Orijinal en-boy oranını korumak için resmi yarı yarıya küçültüp, 1.5 kat büyütebiliriz
img_small = cv2.resize(
    img_color, (int(W * 0.5), int(H * 0.5))
)  # cv2.resize önce Genişlik (W), sonra Yükseklik (H) bekler!
img_large = cv2.resize(img_color, (int(W * 1.5), int(H * 1.5)))
print(f"Küçültülmüş şekil : {img_small.shape}")
print(f"Büyütülmüş şekil  : {img_large.shape}")

# ── 3. Döndürme (Rotation) ────────────────────────────────────────────────────
# OpenCV fonksiyonları float64 ile çalışırken uint8 girdiler bekleyebilir, o yüzden güvenli dönüşüm yapıyoruz
img_uint8 = img_color.astype(np.uint8)
img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)  # RGB → BGR (OpenCV için)

center = (W // 2, H // 2)  # Döndürme merkezi (Genişlik/2, Yükseklik/2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45 derece döndür, ölçek aynı kalsın
img_rotated_bgr = cv2.warpAffine(img_bgr, M, (W, H))
img_rotated = cv2.cvtColor(
    img_rotated_bgr, cv2.COLOR_BGR2RGB
)  # Tekrar Matplotlib için RGB'ye çevir

# ── 4. Aynalamalar (Flip) ─────────────────────────────────────────────────────
img_flip_h = np.fliplr(img_color)  # Yatay çevirme (Ayna efekti)
img_flip_v = np.flipud(img_color)  # Dikey çevirme (Ters takla efekti)

# ── 5. Hepsini Görselleştir ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Satır: Orijinal, Kırpılmış ve Küçültülmüş Görüntüler
axes[0, 0].imshow(img_color.astype("uint8"))
axes[0, 0].set_title("Orijinal Panda")
axes[0, 0].axis("off")

axes[0, 1].imshow(crop.astype("uint8"))
axes[0, 1].set_title(f"Dinamik Kırpma\n({crop.shape[1]}x{crop.shape[0]})")
axes[0, 1].axis("off")

axes[0, 2].imshow(img_small.astype("uint8"))
axes[0, 2].set_title(f"Küçültülmüş Panda\n({img_small.shape[1]}x{img_small.shape[0]})")
axes[0, 2].axis("off")

# 2. Satır: Döndürülmüş ve Aynalanmış Görüntüler
axes[1, 0].imshow(img_rotated.astype("uint8"))
axes[1, 0].set_title("45° Döndürülmüş Panda")
axes[1, 0].axis("off")

axes[1, 1].imshow(img_flip_h.astype("uint8"))
axes[1, 1].set_title("Yatay Ayna (Sol-Sağ)")
axes[1, 1].axis("off")

axes[1, 2].imshow(img_flip_v.astype("uint8"))
axes[1, 2].set_title("Dikey Ayna (Yukarı-Aşağı)")
axes[1, 2].axis("off")

plt.suptitle(
    "🐼 Panda Resmi Üzerinde Temel Görüntü Manipülasyonları",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── 0. Veri Tipi Hazırlığı ────────────────────────────────────────────────────
# OpenCV renk dönüşümleri için img_color'ı uint8 formatına çekiyoruz
img_uint8 = img_color.astype(np.uint8)

# ── 1. RGB → Grayscale Dönüşümü ──────────────────────────────────────────────
# Formül: Gray = 0.2989*R + 0.5870*G + 0.1140*B
gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
print("=== Renk Uzayı Boyut Bilgileri ===")
print(f"Grayscale (Gri) Şekli: {gray.shape} (Kanal yok, tek matris)")

# Manuel hesaplama (img_color float64 olduğu için hassas hesap yapar):
gray_manual = (
    0.2989 * img_color[:, :, 0]
    + 0.5870 * img_color[:, :, 1]
    + 0.1140 * img_color[:, :, 2]
)
# İki yöntem arasındaki maksimum farkı kontrol edelim (Neredeyse 0 olmalı)
diff = abs(gray.astype(float) - gray_manual).max()
print(f"Manuel ve OpenCV Grayscale maksimum farkı: {diff:.2f}")

# ── 2. RGB → HSV Dönüşümü ────────────────────────────────────────────────────
# H=Ton (Renk özü), S=Doygunluk (Rengin canlılığı), V=Parlaklık
img_bgr_orig = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)  # OpenCV BGR ister
hsv = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2HSV)
print(f"HSV Görüntü Şekli   : {hsv.shape}")

# ── 3. RGB → LAB Dönüşümü ────────────────────────────────────────────────────
# L=Aydınlık (Lightness), A=Kırmızı-Yeşil ekseni, B=Sarı-Mavi ekseni
lab = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2LAB)
print(f"LAB Görüntü Şekli   : {lab.shape}\n")

# ── 4. Görselleştirme ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))

# Üst Satır: Ana Dönüşümler
axes[0, 0].imshow(img_uint8)
axes[0, 0].set_title("RGB — Orijinal Panda")
axes[0, 0].axis("off")

axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Grayscale (Siyah-Beyaz)")
axes[0, 1].axis("off")

axes[0, 2].imshow(hsv[:, :, 0], cmap="hsv")
axes[0, 2].set_title("HSV — Ton (H) Kanalı")
axes[0, 2].axis("off")

axes[0, 3].imshow(lab[:, :, 0], cmap="gray")
axes[0, 3].set_title("LAB — Parlaklık (L) Kanalı")
axes[0, 3].axis("off")

# Alt Satır: Kanalların Detaylı İncelenmesi
axes[1, 0].imshow(hsv[:, :, 0], cmap="hsv")
axes[1, 0].set_title("H — Ton (Renk Dağılımı)")
axes[1, 0].axis("off")

axes[1, 1].imshow(hsv[:, :, 1], cmap="gray")
axes[1, 1].set_title("S — Doyma (Canlılık)")
axes[1, 1].axis("off")

axes[1, 2].imshow(hsv[:, :, 2], cmap="gray")
axes[1, 2].set_title("V — Parlaklık (Işık)")
axes[1, 2].axis("off")

# LAB renk uzayının A kanalı Kırmızı-Yeşil skalasını görselleştirmek için 'RdYlGn'

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── 1. Adım: Önceki Adımdan Kalan HSV Görüntüsünü Kontrol Et ──────────────────
# hsv matrisinin tanımlı olduğundan emin oluyoruz (uint8 formatında)
img_uint8 = img_color.astype(np.uint8)
img_bgr_orig = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2HSV)

# ── 2. Adım: Maskelenecek Renk Aralığını Belirle ──────────────────────────────
# Eğer panda resminde yeşil yapraklar/bambular varsa bu aralık onları yakalar:
lower_green = np.array([35, 40, 40])  # Alt sınır (Ton, Doyma, Parlaklık)
upper_green = np.array([85, 255, 255])  # Üst sınır

# 💡 ALTERNATİF (Eğer yeşil yoksa ve pandanın SİYAH tüylerini maskelemek istersen):
# lower_black = np.array([0, 0, 0])
# upper_black = np.array([180, 255, 50]) # Düşük parlaklık (V) siyahı yakalar
# mask = cv2.inRange(hsv, lower_black, upper_black)

# ── 3. Adım: Maskeyi Oluştur ve Uygula ────────────────────────────────────────
# Belirtilen aralıktaki pikselleri 255 (beyaz), diğerlerini 0 (siyah) yapar
mask = cv2.inRange(hsv, lower_green, upper_green)

# bitwise_and işlemi: Maskedeki beyaz bölgeleri orijinal resimden "kesip" alır
masked_bgr = cv2.bitwise_and(img_bgr_orig, img_bgr_orig, mask=mask)
masked_rgb = cv2.cvtColor(
    masked_bgr, cv2.COLOR_BGR2RGB
)  # Matplotlib için tekrar RGB'ye çevir

# ── 4. Adım: Görselleştirme ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Grafik: Orijinal Görüntü
axes[0].imshow(img_uint8)
axes[0].set_title("Orijinal Panda (RGB)")
axes[0].axis("off")

# 2. Grafik: İksir Maskesi (Siyah-Beyaz Matris)
axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Yeşil Renk Maskesi\n(Beyaz = Yakalanan Bölgeler)")
axes[1].axis("off")

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── 0. Veri Tipi ve Boyut Hazırlığı ──────────────────────────────────────────
img_uint8 = img_color.astype(np.uint8)
gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
H, W = gray.shape

# ── 1. Gri Tonlamalı Histogram Hesaplama ──────────────────────────────────────
hist_gray, bins = np.histogram(gray.flatten(), 256, [0, 256])

# ── 2. Histogram Eşitleme İçin Dinamik Bölge Seçimi ───────────────────────────
# Sabit koordinatlar yerine, pandanın merkezinden dinamik bir bölge seçiyoruz.
# Genelde bu bölge kontrast iyileştirmesini test etmek için harika bir örnektir.
y_start, y_end = int(H * 0.3), int(H * 0.7)
x_start, x_end = int(W * 0.2), int(W * 0.6)

dark_region = img_uint8[y_start:y_end, x_start:x_end]
dark_gray = cv2.cvtColor(dark_region, cv2.COLOR_RGB2GRAY)

# Histogram Eşitleme: Kontrastı otomatik olarak maksimize eder
equalized = cv2.equalizeHist(dark_gray)

# ── 3. Görselleştirme Matrisi (2 satır, 3 sütun) ──────────────────────────────
fig = plt.figure(figsize=(16, 8))

# Sol Üst: Orijinal Renkli Panda
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(img_uint8)
ax1.set_title("Orijinal Görüntü (RGB)")
ax1.axis("off")

# Sol Alt: Gri Tonlamalı Panda
ax2 = fig.add_subplot(2, 3, 4)
ax2.imshow(gray, cmap="gray")
ax2.set_title("Gri Görüntü (Grayscale)")
ax2.axis("off")

# Orta Üst: Gri Histogram Eğrisi
ax3 = fig.add_subplot(2, 3, 2)
ax3.plot(hist_gray, color="gray", linewidth=1.5)  # Histogram eğrisi
ax3.fill_between(
    range(256), hist_gray, alpha=0.3, color="gray"
)  # Eğrinin altını doldur
ax3.set_title("Gri Histogram Dağılımı")
ax3.set_xlabel("Piksel Değeri (0:Siyah - 255:Beyaz)")
ax3.set_ylabel("Piksel Sayısı")
ax3.set_xlim([0, 256])
ax3.grid(axis="y", linestyle="--", alpha=0.5)

# Orta Alt: RGB Kanallarının Ayrı Ayrı Histogramları
ax4 = fig.add_subplot(2, 3, 5)
colors = ["red", "green", "blue"]
channel_names = ["Kırmızı (R) Kanalı", "Yeşil (G) Kanalı", "Mavi (B) Kanalı"]

for i, (color, name) in enumerate(zip(colors, channel_names)):
    # Her bir renk katmanını tek boyuta indirgeyip histogramını çıkarıyoruz
    hist_c, _ = np.histogram(img_uint8[:, :, i].flatten(), 256, [0, 256])
    ax4.plot(hist_c, color=color, alpha=0.6, linewidth=1.5, label=name)

ax4.set_title("RGB Kanalları Dağılımı")
ax4.set_xlabel("Piksel Değeri (0-255)")
ax4.set_xlim([0, 256])
ax4.legend(fontsize=9)
ax4.grid(axis="y", linestyle="--", alpha=0.5)

# Sağ Üst: Seçilen Orijinal Bölge
ax5 = fig.add_subplot(2, 3, 3)
ax5.imshow(dark_gray, cmap="gray")
ax5.set_title(f"Seçilen Bölge (Orijinal)\nBoyut: {dark_gray.shape[1]}x{dark_gray.shape[0]}")
ax5.axis("off")

# Sağ Alt: Histogram Eşitleme Sonucu
ax6 = fig.add_subplot(2, 3, 6)
ax6.imshow(equalized, cmap="gray")
ax6.set_title("Histogram Eşitleme Sonrası\n(Dengelenmiş Kontrast)")
ax6.axis("off")

plt.suptitle(
    "🐼 Panda Resmi Histogram Analizi ve Kontrast İyileştirme",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
# Supatitle'ın grafiklerin üstüne binmesini engellemek için küçük bir boşluk payı
plt.subplots_adjust(top=0.90)
plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── 0. Veri Seti Hazırlığı ────────────────────────────────────────────────────
# İlk adımlarda Pillow ile yüklediğimiz ham gri resmi (uint8) temel alıyoruz
# Eğer img_raw yoksa: img_raw = Image.open("panda.jpeg").convert("L") ile np.array yapabilirsin.
img_gray_raw = np.array(img_raw).astype(np.uint8)

# Test için görüntüye gürültü ekle
rng = np.random.default_rng(42)  # Tekrar üretilebilir rastgele sayı üretici
noise = rng.integers(0, 60, img_gray_raw.shape, dtype=np.uint8)  # 0-60 arası gürültü
noisy = np.clip(img_gray_raw.astype(int) + noise, 0, 255).astype(np.uint8)

# ── 1. Ortalama (Box) Filtre ──────────────────────────────────────────────────
# 5x5'lik penceredeki tüm piksellerin aritmetik ortalamasını alır, görüntüyü bulanıklaştırır
kernel_box = np.ones((5, 5), np.float32) / 25
blur_box = cv2.filter2D(noisy, -1, kernel_box)

# ── 2. Gaussian Bulanıklaştırma ───────────────────────────────────────────────
# Merkeze yakın piksellere çan eğrisi (Gaussian dağılımı) formülüne göre daha çok ağırlık verir
blur_gauss = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.5)

# ── 3. Medyan Filtre (Median Filter) ──────────────────────────────────────────
# Penceredeki pikselleri sıralar ve ortadaki (medyan) değeri seçer. Noktasal gürültülerde mükemmeldir
blur_median = cv2.medianBlur(noisy, 5)

# ── 4. Bilateral Filtre (Kenar Koruyucu Bulanıklaştırma) ──────────────────────
# Hem piksellerin birbirine olan uzaklığına hem de renk (yoğunluk) benzerliğine bakar.
# Gürültüyü temizlerken nesnelerin keskin kenarlarını asla bozmaz.
bilateral = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)

# ── 5. Keskinleştirme (Sharpening) ────────────────────────────────────────────
# Orijinal (gürültüsüz) resimdeki kenar detaylarını ön plana çıkarmak için kullanılır
kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharp = cv2.filter2D(img_gray_raw, -1, kernel_sharp)

# ── 6. Görselleştirme (2 satır, 3 sütun = 6 Görsel) ───────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Listeyi 6 görsel olacak şekilde güncelledik (Keskinleştirilmiş resmi de dahil ettik)
imgs = [img_gray_raw, noisy, blur_box, blur_gauss, blur_median, bilateral]
titles = [
    "1. Orijinal Gri Panda",
    "2. Gürültülü Panda (+Rastgele Gürültü)",
    "3. Box Filter (Ortalama 5×5)",
    "4. Gaussian Blur (σ=1.5)",
    "5. Medyan Filtre (5x5)",
    "6. Bilateral Filtre (Kenar Koruyucu)",
]

# Görselleştirme döngüsü
for ax, img, title in zip(axes.flatten(), imgs, titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

plt.suptitle(
    "🐼 Panda Resmi Üzerinde Uzamsal Filtreleme ve Gürültü Azaltma Karşılaştırması",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

# ── Ekstra: Keskinleştirme Sonucunu Ayrıca Görelim ───────────────────────────
fig_sharp, axes_sharp = plt.subplots(1, 2, figsize=(11, 5))
axes_sharp[0].imshow(img_gray_raw, cmap="gray")
axes_sharp[0].set_title("Orijinal Gri Panda")
axes_sharp[0].axis("off")

axes_sharp[1].imshow(sharp, cmap="gray")
axes_sharp[1].set_title("Keskinleştirilmiş (Sharpened) Panda")
axes_sharp[1].axis("off")

plt.suptitle("Laplacian Tabanlı Maske ile Görüntü Keskinleştirme")
plt.tight_layout()
plt.show()

import numpy as np

# ── 1. PSNR Hesaplama Fonksiyonu ──────────────────────────────────────────────
def calculate_psnr(original, filtered):
    """
    PSNR hesapla: Orijinal gürültüsüz görüntü ile filtrelenmiş görüntüyü karşılaştırır.
    Formül: PSNR = 20 * log10(MAX_I / sqrt(MSE))
    """
    # İki matris arasındaki farkların karelerinin ortalaması (Mean Squared Error)
    mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)

    if mse == 0:
        return float('inf')  # Sıfır hata varsa sonsuz başarı (mükemmel eşleşme)

    max_pixel = 255.0  # 8-bit görüntüler için maksimum piksel değeri
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))  # dB cinsinden PSNR sonucu
    return psnr

# ── 2. Filtre Sonuçlarını Sözlük (Dictionary) Yapısında Toplama ───────────────
# Önceki kod hücresinde ürettiğimiz filtre sonuçlarını buraya bağlıyoruz
filters = {
    'Box Filter (Ortalama)' : blur_box,
    'Gaussian Blur'         : blur_gauss,
    'Medyan Filtre'         : blur_median,
    'Bilateral Filtre'      : bilateral,
}

# ── 3. Sonuçları Ekranda Karşılaştırmalı ve Grafik Çubuklu Listeleme ──────────
print("=== 🐼 Panda Görüntüsü İçin PSNR Karşılaştırması (dB) ===")
print(f"{'Uygulanan Filtre':<24} {'PSNR (dB)':>10}")
print("-" * 45)

for name, filtered in filters.items():
    # img_gray_raw: Senin yüklediğin orijinal gürültüsüz gri panda resmi
    psnr = calculate_psnr(img_gray_raw, filtered)

    # Konsolda tatlı bir grafik çubuğu (progress bar) oluşturmak için:
    # Örneğin 28 dB için 14 tane "█" karakteri basar
    bar = "█" * int(psnr / 2)

    print(f"{name:<24} {psnr:>8.2f} dB  {bar}")

print("-" * 45)
print("💡 Önemli Not: Daha yüksek PSNR = Orijinal resme daha yakın ve başarılı filtreleme.")

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── 0. Görüntü Ön İşleme (Pre-processing) ────────────────────────────────────
# Görüntüdeki yüksek frekanslı sahte gürültülerin kenar olarak algılanmaması için
# kenar tespitinden önce mutlaka hafif bir Gaussian Blur uygulanır.
test_img = cv2.GaussianBlur(img_gray_raw, (3, 3), 0)

# ── 1. Sobel Kenar Dedektörü (Birinci Türev Tabanlı) ─────────────────────────
# Sobel X: Yatay gradyanı hesaplar, dikey kenarlara (örn. ağaç gövdeleri, dik çizgiler) duyarlıdır
sobel_x = cv2.Sobel(test_img, cv2.CV_64F, 1, 0, ksize=3)

# Sobel Y: Dikey gradyanı hesaplar, yatay kenarlara (örn. yer çizgileri, ufuk çizgisi) duyarlıdır
sobel_y = cv2.Sobel(test_img, cv2.CV_64F, 0, 1, ksize=3)

# Toplam Sobel: İki yönün Pisagor formülüyle birleşimi (Gradyan Büyüklüğü)
sobel_total = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_total = np.clip(sobel_total, 0, 255).astype(np.uint8)

# ── 2. Laplacian Kenar Dedektörü (İkinci Türev Tabanlı) ──────────────────────
# İkinci türev yön bağımsız çalışır; her yöndeki ani değişimleri tek seferde yakalar
laplacian = cv2.Laplacian(test_img, cv2.CV_64F, ksize=3)
laplacian_abs = np.abs(laplacian).astype(np.uint8)

# ── 3. Canny Kenar Dedektörü (Çok Aşamalı Akıllı Algoritma) ────────────────────
# Gürültü azaltma, gradyan bulma, inceltme (non-max suppression) ve çift eşikleme aşamalarından oluşur.
# threshold1: Zayıf kenar eşiği, threshold2: Güçlü kenar eşiği
canny_strict = cv2.Canny(test_img, threshold1=100, threshold2=200)

# ── 4. Görselleştirme (2 satır, 3 sütun) ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Satır: Giriş ve Sobel Yön Analizleri
axes[0, 0].imshow(test_img, cmap="gray")
axes[0, 0].set_title("Orijinal Filtrelenmiş Panda (Gri)")
axes[0, 0].axis("off")

axes[0, 1].imshow(np.abs(sobel_x).astype(np.uint8), cmap="gray")
axes[0, 1].set_title("Sobel X\n(Dikey Kenarlar)")
axes[0, 1].axis("off")

axes[0, 2].imshow(np.abs(sobel_y).astype(np.uint8), cmap="gray")
axes[0, 2].set_title("Sobel Y\n(Yatay Kenarlar)")
axes[0, 2].axis("off")

# 2. Satır: Birleşik ve İleri Seviye Kenar Detektörleri
axes[1, 0].imshow(sobel_total, cmap="gray")
axes[1, 0].set_title("Sobel Toplam\n√(Gx² + Gy²)")
axes[1, 0].axis("off")

axes[1, 1].imshow(laplacian_abs, cmap="gray")
axes[1, 1].set_title("Laplacian (İkinci Türev)")
axes[1, 1].axis("off")

axes[1, 2].imshow(canny_strict, cmap="gray")
axes[1, 2].set_title("Canny Algoritması (100/200)\n(İnceltilmiş Temiz Sınırlar)")
axes[1, 2].axis("off")

plt.suptitle(
    "🐼 Panda Resmi Kenar Tespiti Matematiksel Karşılaştırması",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── 0. Giriş ve İkili (Binary) Görüntü Hazırlığı ──────────────────────────────
# Önceki adımdan gelen pürüzsüzleştirilmiş test_img'i kullanıyoruz.
# Canny bize morfolojik işlemler için harika bir siyah-beyaz (binary) matris verir.
binary = cv2.Canny(test_img, 50, 150)

# ── 1. Yapısal Element (Structuring Element / Kernel) ─────────────────────────
# Matrisin üzerinde gezinerek morfolojik dönüşümü yapacak şekil şablonu.
# 3x3 boyutunda kare (rectangular) bir filtre penceresi tanımlıyoruz.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# ── 2. Morfolojik Matematiksel İşlemler ───────────────────────────────────────

# Erosion (Aşındırma): Beyaz bölgeleri kenarlarından kemirerek küçültür.
# İnce sahte çizgileri ve gürültü piksellerini yok etmek için idealdır.
erosion = cv2.erode(binary, kernel, iterations=1) # Farkı net görmek için iterasyonu 1 yapabilirsin

# Dilation (Genişletme): Beyaz bölgeleri kenarlarından genişleterek büyütür.
# Nesne üzerindeki kopuklukları veya çatlakları birleştirir.
dilation = cv2.dilate(binary, kernel, iterations=1)

# Opening (Açma): Önce Erosion + Sonra Dilation yapar.
# Çizgilerin genel kalınlığını bozmadan, arka plandaki küçük beyaz noktaları (gürültüleri) siler.
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing (Kapama): Önce Dilation + Sonra Erosion yapar.
# Çizgilerin kalınlığını bozmadan, nesne içindeki siyah boşlukları ve küçük delikleri kapatır.
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Morphological Gradient: Dilation matrisinden Erosion matrisini çıkarır.
# Geriye sadece nesnenin en dıştaki ince kontur (sınır) çizgisi kalır.
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

# ── 3. Görselleştirme (2 satır, 3 sütun) ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Satır: Giriş, Aşındırma ve Genişletme
axes[0, 0].imshow(binary, cmap="gray")
axes[0, 0].set_title("Orijinal Canny Binary")
axes[0, 0].axis("off")

axes[0, 1].imshow(erosion, cmap="gray")
axes[0, 1].set_title("Erosion (Aşındırma)\n[Çizgiler İnceltildi]")
axes[0, 1].axis("off")

axes[0, 2].imshow(dilation, cmap="gray")
axes[0, 2].set_title("Dilation (Genişletme)\n[Çizgiler Kalınlaştırıldı]")
axes[0, 2].axis("off")

# 2. Satır: Açma, Kapama ve Gradyan Konturları
axes[1, 0].imshow(opening, cmap="gray")
axes[1, 0].set_title("Opening (Açma)\n[Dış Gürültüler Silindi]")
axes[1, 0].axis("off")

axes[1, 1].imshow(closing, cmap="gray")
axes[1, 1].set_title("Closing (Kapama)\n[İç Boşluklar Kapatıldı]")
axes[1, 1].axis("off")

axes[1, 2].imshow(gradient, cmap="gray")
axes[1, 2].set_title("Morphological Gradient\n[Sınır Konturları]")
axes[1, 2].axis("off")

plt.suptitle(
    "🐼 Panda Kenar Haritası Üzerinde Morfolojik İşlemler",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── 0. Dinamik Bölge Seçimi (ROI - Region of Interest) ────────────────────────
# Önceki adımlardan gelen img_gray_raw (uint8 gri panda) matrisini alıyoruz
img_gray_raw = np.array(img_raw).astype(np.uint8)
H, W = img_gray_raw.shape

# Sabit sınırlar yerine pandanın merkezine yakın %50'lik bir ilgi alanı kırpalım
y_start, y_end = int(H * 0.2), int(H * 0.8)
x_start, x_end = int(W * 0.2), int(W * 0.8)

test = img_gray_raw[y_start:y_end, x_start:x_end]
print(f"Eşikleme yapılacak dinamik bölge boyutu: {test.shape[1]}x{test.shape[0]}")

# ── 1. Global (Manuel) Eşikleme ─────────────────────────────────────────────
# Tüm resim için tek bir eşik değeri (127) kullanılır. Üstü beyaz, altı siyah olur.
_, thresh_global = cv2.threshold(test, 127, 255, cv2.THRESH_BINARY)

# ── 2. Otsu Yöntemi (Otomatik Global Eşikleme) ───────────────────────────────
# Resmin histogramını analiz ederek iki sınıfı (arka plan/ön plan)
# en iyi ayıran optimum eşik değerini matematiksel olarak kendi hesaplar.
otsu_val, thresh_otsu = cv2.threshold(
    test, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(f"Otsu algoritmasının panda için hesapladığı ideal eşik değeri: {otsu_val:.0f}")

# ── 3. Adaptif Ortalama Eşikleme (Yerel Eşikleme) ─────────────────────────────
# Resimde homojen olmayan bir ışık/gölge varsa global eşikler patlar.
# Bu yöntem her pikselin 11x11'lik komşuluğunun ortalamasına bakarak yerel eşik belirler.
thresh_adapt_mean = cv2.adaptiveThreshold(
    test,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,  # Komşuluk pencere boyutu (Tek sayı olmalı)
    C=2,  # Ortalamadan çıkarılacak dengeleme sabiti
)

# ── 4. Adaptif Gaussian Eşikleme ─────────────────────────────────────────────
# Ortalama yerine yerel bölgeye Gaussian filtresi (merkeze yakın pikseller baskın)
# uygulayarak eşik belirler. Çizgilerde daha az karıncalanma yapar.
thresh_adapt_gauss = cv2.adaptiveThreshold(
    test, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2
)

# ── 5. Görselleştirme Matrisi (2 satır, 3 sütun) ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Üst Satır: Global Yöntemler
axes[0, 0].imshow(test, cmap="gray")
axes[0, 0].set_title("Orijinal Kırpılmış Panda (Gri)")
axes[0, 0].axis("off")

axes[0, 1].imshow(thresh_global, cmap="gray")
axes[0, 1].set_title("Global Eşik (Sabit: 127)")
axes[0, 1].axis("off")

axes[0, 2].imshow(thresh_otsu, cmap="gray")
axes[0, 2].set_title(f"Otsu Yöntemi (Otomatik Eşik: {otsu_val:.0f})")
axes[0, 2].axis("off")

# Alt Satır: Adaptif Yöntemler ve Grafik
axes[1, 0].imshow(thresh_adapt_mean, cmap="gray")
axes[1, 0].set_title("Adaptif Ortalama (Yerel)")
axes[1, 0].axis("off")

axes[1, 1].imshow(thresh_adapt_gauss, cmap="gray")
axes[1, 1].set_title("Adaptif Gaussian (Yerel)")
axes[1, 1].axis("off")

# Sağ Alt: Histogram Üzerinde Eşik Çizgilerini Gösterme
axes[1, 2].hist(test.flatten(), 256, [0, 256], color="steelblue", alpha=0.7)
axes[1, 2].axvline(
    x=otsu_val,
    color="red",
    linewidth=2,
    linestyle="--",
    label=f"Otsu Eşiği ({otsu_val:.0f})",
)
axes[1, 2].axvline(
    x=127, color="orange", linewidth=2, linestyle="--", label="Manuel Eşik (127)"
)
axes[1, 2].set_title("Piksel Histogramı ve Eşik Analizi")
axes[1, 2].set_xlabel("Piksel Değeri (0-255)")
axes[1, 2].set_ylabel("Piksel Sayısı (Frekans)")
axes[1, 2].legend(fontsize=9)
axes[1, 2].grid(axis="y", linestyle="--", alpha=0.3)

plt.suptitle(
    "🐼 Panda Resmi Üzerinde Görüntü Eşikleme (Thresholding) Teknikleri",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── 0. Dinamik Çalışma Bölgesi Seçimi (ROI) ──────────────────────────────────
# Önceki adımlardan gelen img_gray_raw (uint8 gri panda) matrisini alıyoruz
img_gray_raw = np.array(img_raw).astype(np.uint8)
H, W = img_gray_raw.shape

# Sabit sınırlar yerine resmin merkez etrafındaki %70'lik güvenli alanını seçelim
y_start, y_end = int(H * 0.15), int(H * 0.85)
x_start, x_end = int(W * 0.15), int(W * 0.85)

prep = img_gray_raw[y_start:y_end, x_start:x_end]
print("=== 🐼 Panda Nesne Analizi ===")
print(f"Çalışma Bölgesi Çözünürlüğü: {prep.shape[1]}x{prep.shape[0]}")

# ── 1. Görüntüyü Kontur Bulma İçin Hazırla ───────────────────────────────────
# Konturların net çıkması için önce pürüzleri Gaussian ile eziyoruz, sonra Otsu ile binary yapıyoruz
prep_blur = cv2.GaussianBlur(prep, (5, 5), 0)
_, prep_thresh = cv2.threshold(
    prep_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# ── 2. Kontur Bulma (Contour Detection) ──────────────────────────────────────
# contours: Bulunan her bir konturun nokta koordinatlarını içeren bir listedir.
contours, hierarchy = cv2.findContours(
    prep_thresh,
    cv2.RETR_EXTERNAL,  # Sadece en dıştaki kapsayıcı konturları al
    cv2.CHAIN_APPROX_SIMPLE,  # Doğrusal hatlardaki gereksiz noktaları silerek hafızayı korur
)
print(f"Bulunan ham toplam kontur sayısı: {len(contours)}")

# ── 3. Alan Filtrelemesi (Gürültü Temizleme) ──────────────────────────────────
# Resimdeki çok küçük lekelerin kontur olarak alınmaması için alan eşiği uyguluyoruz
min_area = 200  # Piksel kare cinsinden minimum alan
large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
print(f"Alanı {min_area} px² değerinden büyük olan güçlü konturlar: {len(large_contours)}")

# ── 4. Konturları ve Geometrik Özellikleri Çiz ───────────────────────────────
canvas = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)  # Gri resmi, renkli çizimler için BGR yapıyoruz

# Yeşil renkli (0, 255, 0) ve 2 piksel kalınlığında ana kontur çizgilerini çiz
cv2.drawContours(canvas, large_contours, -1, (0, 255, 0), 2)

# İlk 10 büyük konturun geometrik merkezini ve kapsayıcı kutusunu (Bounding Box) bulalım
for i, cnt in enumerate(large_contours[:10]):
    # Nesnenin sınır diktörtgen koordinatlarını (Sol üst X, Sol üst Y, Genişlik, Yükseklik) al
    x, y, w, h = cv2.boundingRect(cnt)
    # Mavi renkli (255, 0, 0) kapsayıcı dikdörtgen kutuyu çiz
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # İstatistiksel Momentleri kullanarak nesnenin ağırlık merkezini hesapla
    M_cnt = cv2.moments(cnt)
    if M_cnt["m00"] != 0:  # Paydanın sıfır olmasını engelle (Alan sıfır değilse)
        cx = int(M_cnt["m10"] / M_cnt["m00"])  # X koordinat merkezi
        cy = int(M_cnt["m01"] / M_cnt["m00"])  # Y koordinat merkezi
        # Kırmızı renkli (0, 0, 255) içi dolu nokta koy
        cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)

# OpenCV'nin BGR formatını Matplotlib'in RGB formatına çeviriyoruz
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# ── 5. Görselleştirme (1 satır, 3 sütun) ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(prep, cmap="gray")
axes[0].set_title("Orijinal Kırpılmış Bölge")
axes[0].axis("off")

axes[1].imshow(prep_thresh, cmap="gray")
axes[1].set_title("Eşiklenmiş (Otsu Binary)")
axes[1].axis("off")

axes[2].imshow(canvas_rgb)
axes[2].set_title(
    f"Analiz Edilen Konturlar ({len(large_contours)} Adet)\n🟢Kontur  🔵BBox  🔴Ağırlık Merkezi"
)
axes[2].axis("off")

plt.suptitle(
    "🐼 Panda Resmi Üzerinde Kontur Tespiti ve Geometrik Nesne Analizi",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()