"""görüntü_işleme.ipynb

"""

# ── Tüm kütüphaneleri içe aktar ───────────────────────────────────────────────
import numpy as np                          # Matris işlemleri
import matplotlib.pyplot as plt             # Görselleştirme
import matplotlib.colors as mcolors        # Renk haritası yardımcıları
from matplotlib.gridspec import GridSpec    # Karmaşık subplot düzeni

from sklearn.datasets import fetch_openml   # Orijinal MNIST'i indirmek için
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA       # Boyut indirgeme
from sklearn.manifold import TSNE           # Görselleştirme için boyut indirgeme
from sklearn.svm import SVC                 # Destek Vektör Makinesi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier   # Çok Katmanlı Algılayıcı (Sinir Ağı)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns                       # Karmaşıklık matrisini güzel çizmek için

import warnings
warnings.filterwarnings('ignore')

# ── 1. MNIST Veri Setini Yükle ────────────────────────────────────────────────
print("⏳ MNIST Veri seti OpenML üzerinden indiriliyor (Bu işlem internet hızınıza bağlı olarak 1-2 dk sürebilir)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X_all = mnist.data      # (70000, 784) -> Düzleştirilmiş piksel değerleri (0-255)
y_all = mnist.target.astype(int) # (70000,) -> Hedef etiketler (0-9)

print("✅ MNIST Başarıyla Yüklendi!")
print(f"   Orijinal Veri Boyutu: {X_all.shape[0]} örnek | Özellik sayısı: {X_all.shape[1]} (28×28 piksel)")

# ── 2. Performans İçin Örneklem Seçimi (Opsiyonel) ─────────────────────────────
# 70.000 örneğin tamamında SVM eğitmek çok uzun sürebilir.
# Testi hızlandırmak için rastgele 5000 örnek seçiyoruz.
sample_size = 5000
random_indices = np.random.choice(X_all.shape[0], sample_size, replace=False)
X_flat = X_all[random_indices]
y_digits = y_all[random_indices]

# 2D orijinal görüntü formunu elde etme (Görselleştirme için)
X_digits = X_flat.reshape(-1, 28, 28)

# ── 3. Eğitim / Test Bölmesi ve Ölçeklendirme ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_digits, test_size=0.25, random_state=42, stratify=y_digits
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # Eğitimde fit+transform
X_test_sc  = scaler.transform(X_test)        # Testte sadece transform

print(f"\n⚡ Analiz Başlıyor (Örneklem Kümesi):")
print(f"   Eğitim: {X_train.shape[0]} örnek  |  Test: {X_test.shape[0]} örnek")

# ── 4. İlk 5 Rakamı Görselleştirme (Doğrulama) ────────────────────────────────
plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_digits[i], cmap='gray')
    plt.title(f"Etiket: {y_digits[i]}")
    plt.axis('off')
plt.suptitle("MNIST Veri Setinden Örnek Görüntüler (28x28)", y=1.05)

import numpy as np
import cv2
import matplotlib.pyplot as plt

def augment_image_mnist(img_2d, seed=None):
    """
    Tek bir 28x28 MNIST görüntüsüne rastgele augmentation uygula.
    Veri aralığı: 0-255 arası float/int girdiler için uygundur.
    """
    rng = np.random.default_rng(seed)          # Tekrar üretilebilir rastgelelik

    # Görüntüyü OpenCV işlemleri için doğrudan standart uint8 formatına getiriyoruz
    img_uint8 = np.clip(img_2d, 0, 255).astype(np.uint8)
    h, w  = img_uint8.shape

    results = {}
    results['original'] = img_uint8.copy().astype(float) # Orijinali sakla

    # ── 1. Döndürme (Rotation) ──────────────────────────────────────────────
    angle = rng.uniform(-20, 20)               # Rakamlar için ideal: -20° ile +20° arası
    M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img_uint8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    results['rotation'] = rotated.astype(float)

    # ── 2. Kaydırma (Translation) ──────────────────────────────────────────
    # 28x28 boyuta uygun olarak piksel kaydırma limitini ±3.5 piksele çıkardık
    tx = rng.uniform(-3.5, 3.5)
    ty = rng.uniform(-3.5, 3.5)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(img_uint8, M_trans, (w, h), borderMode=cv2.BORDER_REPLICATE)
    results['translation'] = shifted.astype(float)

    # ── 3. Yatay Ayna (Horizontal Flip) ────────────────────────────────────
    # Not: Rakamlarda anlam bozulmasına yol açabilir (Örn: 3, 5, 6 ters döner).
    # Ancak pipeline yapısını korumak için ölçeğiyle korundu.
    flipped = np.fliplr(img_uint8)
    results['flip'] = flipped.astype(float)

    # ── 4. Gaussian Gürültü ─────────────────────────────────────────────────
    # 0-255 skalasına uygun gürültü şiddeti
    sigma  = rng.uniform(10, 30)
    noise  = rng.normal(0, sigma, img_uint8.shape).astype(np.int16)
    noisy  = np.clip(img_uint8.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    results['noise'] = noisy.astype(float)

    # ── 5. Parlaklık Değiştirme (Brightness) ───────────────────────────────
    factor = rng.uniform(0.5, 1.5)
    bright = np.clip(img_uint8.astype(float) * factor, 0, 255).astype(np.uint8)
    results['brightness'] = bright.astype(float)

    # ── 6. Zoom (Kırp + Yeniden Boyutlandır) ───────────────────────────────
    # 28x28 pikselde kararlı bir zoom için oran %80-%95 olarak güncellendi
    zoom   = rng.uniform(0.8, 0.95)
    crop_h = int(h * zoom)
    crop_w = int(w * zoom)
    y0     = rng.integers(0, h - crop_h + 1)
    x0     = rng.integers(0, w - crop_w + 1)
    cropped  = img_uint8[y0:y0+crop_h, x0:x0+crop_w]
    zoomed   = cv2.resize(cropped, (w, h))
    results['zoom'] = zoomed.astype(float)

    return results

# ── MNIST Örneği Üzerinde Augmentation Uygulaması ─────────────────────────────
# Önceki kodda seçilen rastgele alt kümeden (X_digits) bir örnek alıyoruz
sample_idx = 0  # İlk indexteki rakamı alalım
sample_img = X_digits[sample_idx]
aug_results = augment_image_mnist(sample_img, seed=42)

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
axes = axes.flatten()

titles = ['Orijinal', 'Döndürme\n(±20°)', 'Kaydırma\n(±3.5px)',
          'Yatay Ayna', 'Gaussian\nGürültü', 'Parlaklık\nDeğiştirme',
          'Zoom\n(Kırp+Yeniden Boyutlandır)', 'Çakışma\n(Hepsi Birden)']

for i, (key, img) in enumerate(aug_results.items()):
    # MNIST standart çizimi için genellikle interpolation='bilinear' veya 'nearest' tercih edilir
    axes[i].imshow(img, cmap='gray_r', interpolation='bilinear')
    axes[i].set_title(titles[i], fontsize=11, fontweight='bold')
    axes[i].set_xlabel(f"Boyut: {img.shape}\nMin:{img.min():.1f} Max:{img.max():.1f}", fontsize=9)
    axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 28x28 boyuta uygun daha seyrek piksel ızgarası kılavuz çizgileri
    axes[i].set_xticks(np.arange(-0.5, 28, 4), minor=True)
    axes[i].set_yticks(np.arange(-0.5, 28, 4), minor=True)
    axes[i].grid(which='minor', color='gray', linewidth=0.3, alpha=0.3)

# Son panel: Tüm varyasyonların ortalamasını alarak harmanlanmış (combined) çıktı üretme
combined = np.mean([v for v in aug_results.values()], axis=0)
axes[7].imshow(combined, cmap='gray_r', interpolation='bilinear')
axes[7].set_title(titles[7], fontsize=11, fontweight='bold')
axes[7].set_xlabel(f"Boyut: {combined.shape}\nMin:{combined.min():.1f} Max:{combined.max():.1f}", fontsize=9)
axes[7].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.suptitle(f"MNIST için Veri Artırma (Data Augmentation) — Gerçek Etiket: {y_digits[sample_idx]}",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Augmentation ile veri setini büyüt ve sınıflandırma etkisini göster ───────

def build_augmented_dataset_mnist(X_img, y, n_aug=3, seed=0):
    """
    Her 28x28 MNIST görüntüsüne n_aug kez augmentation uygulayarak veri setini büyütür.
    Döner: (X_aug, y_aug) — orijinaller dahil, düzleştirilmiş (flattened) matrisler.
    """
    # Orijinal verileri düzleştirerek (28x28 -> 784) başlangıç listesine ekle
    X_list = [X_img.reshape(len(X_img), -1)]
    y_list = [y]
    rng = np.random.default_rng(seed)

    for aug_round in range(n_aug):
        X_new = []
        for img in X_img:
            s = rng.integers(0, 10000)
            # Bir önceki adımda yazdığımız MNIST uyumlu fonksiyonu çağırıyoruz
            aug = augment_image_mnist(img, seed=int(s))

            # Her turda farklı bir augmentation tipi seç ('original' hariç sırayla gider)
            key = list(aug.keys())[1 + (aug_round % 6)]
            X_new.append(aug[key].flatten())              # 28×28 → 784 özellik

        X_list.append(np.array(X_new))
        y_list.append(y)

    return np.vstack(X_list), np.concatenate(y_list)

# ── 1. Veriyi Temiz İndislerle Bölme ──────────────────────────────────────────
# X_digits (28x28 görüntü) ve X_flat (784 özellik) dizilerimizi bölüyoruz
train_idx, test_idx, _, _ = train_test_split(
    np.arange(len(y_digits)), y_digits, test_size=0.25, random_state=42, stratify=y_digits
)

# Eğitim ve test setlerini oluşturma
X_tr_imgs = X_digits[train_idx]       # Eğitim için 2D görüntüler (Augmentation için)
y_tr      = y_digits[train_idx]

X_te_flat = X_flat[test_idx]          # Test için düzleştirilmiş görüntüler (Tahmin için)
y_te      = y_digits[test_idx]

# ── 2. Veri Setini Büyütme (Data Augmentation) ────────────────────────────────
print("⏳ MNIST Eğitim verisine augmentation uygulanıyor, veri seti büyütülüyor...")
X_aug, y_aug = build_augmented_dataset_mnist(X_tr_imgs, y_tr, n_aug=3, seed=42)

print(f"✅ Orijinal eğitim seti  : {len(y_tr)} örnek")
print(f"✅ Augmented eğitim seti : {len(y_aug)} örnek  (4× büyüdü)")
print(f"   Yeni veri matrisi boyutu: {X_aug.shape} (Her örnek {X_aug.shape[1]} özellik)")

# ── 3. Ölçeklendirme (Scaling) ────────────────────────────────────────────────
sc = StandardScaler()
X_tr_sc  = sc.fit_transform(X_flat[train_idx])     # Orijinal eğitim verisi (Ölçeklenmiş)
X_aug_sc = sc.fit_transform(X_aug)                  # Büyütülmüş eğitim verisi (Ölçeklenmiş)
X_te_sc  = sc.transform(X_te_flat)                  # Test verisi (Sadece transform)

# ── 4. Çok Katmanlı Algılayıcı (MLP) Modellerinin Eğitimi ──────────────────────
print("\n🤖 Yapay Sinir Ağları (MLP) eğitiliyor (Bu işlem 1-2 dakika sürebilir)...")

# MNIST 784 özelliğe sahip olduğundan katman genişliklerini (256, 128) olarak koruyoruz
mlp_orig = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
mlp_aug  = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)

# Modelleri fit etme
mlp_orig.fit(X_tr_sc, y_tr)
print("   [1/2] Orijinal veriyle kurulan model eğitildi.")

mlp_aug.fit(X_aug_sc, y_aug)
print("   [2/2] Artırılmış (Augmented) veriyle kurulan model eğitildi.")

# ── 5. Sonuçların Karşılaştırılması ──────────────────────────────────────────
acc_orig = accuracy_score(y_te, mlp_orig.predict(X_te_sc))
acc_aug  = accuracy_score(y_te, mlp_aug.predict(X_te_sc))

print("\n📊 DENEY SONUÇLARI")
print("-" * 40)
print(f"Test Doğruluğu — Orijinal veri : %{acc_orig*100:.2f}")
print(f"Test Doğruluğu — Augmented veri: %{acc_aug*100:.2f}")
print(f"Augmentation Net Kazancı       : {(acc_aug - acc_orig)*100:+.2f} puan")
print("-" * 40)

# ── PCA ile Boyut İndirgeme (MNIST UYARLAMASI) ───────────────────────────────
pca_full = PCA()                               # Tüm bileşenleri hesapla
pca_full.fit(X_flat)                           # 784 bileşenin hepsini öğren (MNIST için)

# Kümülatif açıklanan varyans — kaç bileşen yeterli?
cumvar = np.cumsum(pca_full.explained_variance_ratio_)   # Her bileşenin katkısını topla
n90  = np.searchsorted(cumvar, 0.90) + 1       # %90 için gereken bileşen sayısı
n95  = np.searchsorted(cumvar, 0.95) + 1       # %95 için
n99  = np.searchsorted(cumvar, 0.99) + 1       # %99 için

print(f"📊 MNIST PCA ANALİZ SONUÇLARI:")
print("-" * 45)
print(f"   %90 varyans için gerekli bileşen: {n90}")
print(f"   %95 varyans için gerekli bileşen: {n95}")
print(f"   %99 varyans için gerekli bileşen: {n99}")
print(f"   Orijinal boyut: 784  →  Sıkıştırma oranı (%90): {784/n90:.1f}×")
print("-" * 45)

# ── Eigendigits (PCA bileşenlerini görselleştir) ──────────────────────────────
# Her öz bileşen, veri setindeki bir 'temel şekil'i temsil eder
fig, axes = plt.subplots(3, 7, figsize=(18, 9))

# İlk satır: varyans grafiği (MNIST için ilk 50 bileşeni göstermek daha anlamlıdır)
ax_var = fig.add_axes([0.05, 0.68, 0.90, 0.25])  # [left, bottom, width, height]
show_components = 50  # Grafikte gösterilecek bileşen sayısı

ax_var.bar(range(1, show_components + 1), pca_full.explained_variance_ratio_[:show_components]*100,
           color='steelblue', alpha=0.7, label='Tekil bileşen')
ax_var.plot(range(1, show_components + 1), cumvar[:show_components]*100,
            'ro-', markersize=3, linewidth=1.2, label='Kümülatif')

ax_var.axhline(y=90, color='orange', linestyle='--', linewidth=1, label='%90 Varyans')
ax_var.axhline(y=95, color='red',    linestyle='--', linewidth=1, label='%95 Varyans')

ax_var.set_xlabel("Bileşen Numarası")
ax_var.set_ylabel("Açıklanan Varyans (%)")
ax_var.set_title("MNIST PCA — İlk 50 Bileşenin Katkısı", fontweight='bold')
ax_var.legend(loc='lower right', fontsize=8)
ax_var.set_xlim(0.5, show_components + 0.5)
ax_var.grid(True, alpha=0.2)

# Eigendigit görüntüleri (her bileşen bir 28×28 görüntüye reshape edilir)
# 3x7 = toplam 21 alt grafik (axes.flatten'ın ilk 21 elemanını kullanıyoruz)
for i, ax in enumerate(axes.flatten()):
    if i < 21:
        # MNIST özellikleri için 784 -> 28x28 dönüşümü yapılıyor
        eigenimg = pca_full.components_[i].reshape(28, 28)

        # vmin ve vmax değerleri özvektörlerin yoğunluğuna göre ayarlandı
        im = ax.imshow(eigenimg, cmap='RdBu_r',             # Kırmızı-Mavi renk haritası
                       vmin=-0.15, vmax=0.15)
        ax.set_title(f"PC {i+1}\n({pca_full.explained_variance_ratio_[i]*100:.1f}%)",
                     fontsize=8, fontweight='bold')
        ax.axis('off')
    else:
        # Boşta kalan (21'den büyük) eksenleri gizle
        ax.axis('off')

# Colorbar (Renk Skalası)
fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04,
             label='Bileşen ağırlığı (Mavi: Pozitif Katkı ↔ Kırmızı: Negatif Katkı)')

plt.suptitle("Eigendigits — MNIST PCA Temel Bileşenleri\n(Her 28x28 görüntü rakamları oluşturan soyut bir kalıbı temsil eder)",
             fontsize=13, fontweight='bold', y=0.66)
plt.show()

# ── PCA ile Görüntü Sıkıştırma ve Yeniden Yapılandırma (MNIST UYARLAMASI) ────
# n bileşenle sıkıştır → orijinal boyuta geri döndür ve kaliteyi karşılaştır

# MNIST (784 özellik) için anlamlı bileşen sayıları listesi
n_components_list = [2, 10, 30, 80, 150, 784]
sample_idx = 42                               # Gösterilecek örnek indeksi
original   = X_flat[sample_idx]              # Orijinal düzleştirilmiş görüntü (784,)

# Grafik boyutunu grid düzenine göre ayarlıyoruz
fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(20, 4))

# Orijinal görüntünün çizilmesi (28x28 olarak reshape edildi)
axes[0].imshow(original.reshape(28, 28), cmap='gray_r', interpolation='bilinear')
axes[0].set_title(f"Orijinal\n(784 özellik)\nRakam: {y_digits[sample_idx]}", fontsize=10, fontweight='bold')
axes[0].axis('off')

for j, n in enumerate(n_components_list):
    # n bileşenli PCA modeli
    pca_n = PCA(n_components=n)
    pca_n.fit(X_flat)                              # Veri setine fit et

    # Seçilen örneği sıkıştır ve geri yapılandır
    compressed = pca_n.transform(X_flat[sample_idx:sample_idx+1])  # Boyut indirgeme
    reconstructed = pca_n.inverse_transform(compressed)            # Eski boyuta geri dön

    # Yeniden yapılandırma hatası (Mean Squared Error - MSE)
    mse = np.mean((original - reconstructed[0])**2)
    # n adet bileşenin toplam açıklanan varyans yüzdesi
    var_exp = np.sum(pca_n.explained_variance_ratio_) * 100

    # Geri yapılandırılan görüntüyü 28x28 olarak çizdirme
    axes[j+1].imshow(reconstructed[0].reshape(28, 28), cmap='gray_r', interpolation='bilinear')

    # Son varyans adımı %100 olduğunda veya n=784 olduğunda başlığı optimize et
    if n == 784:
        axes[j+1].set_title(f"n={n} (Tam Boyut)\n%{var_exp:.1f} varyans\nMSE={mse:.2f}", fontsize=9)
    else:
        axes[j+1].set_title(f"n={n}\n%{var_exp:.0f} varyans\nMSE={mse:.2f}", fontsize=9)

    axes[j+1].axis('off')

plt.suptitle("MNIST için PCA Görüntü Sıkıştırma — Bileşen Sayısı (n) Arttıkça Detaylar ve Kalite Yükselir",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# ── 1. Adım: PCA ile Ön İndirgeme (t-SNE'yi hızlandırmak için) ────────────────
print(f"Adım 1: PCA ile 30 boyuta indiriliyor (Girdi boyutu: {X_flat.shape})...")
pca30 = PCA(n_components=30, random_state=42)
X_pca30 = pca30.fit_transform(X_flat)         # (Örnek Sayısı, 784) → (Örnek Sayısı, 30)
print(f"  PCA sonrası yeni boyut: {X_pca30.shape}")

# ── 2. Adım: t-SNE ile 2 Boyuta İndirgeme ─────────────────────────────────────
print("Adım 2: t-SNE ile 2 boyuta indiriliyor (Bu işlem verinin büyüklüğüne göre biraz sürebilir)...")
tsne = TSNE(
    n_components=2,       # 2D düzleme yansıt
    perplexity=30,        # Yerel komşu sayısı (MNIST için 30-50 arası idealdir)
    max_iter=1000,        # İterasyon sayısı
    random_state=42,      # Tekrar üretilebilirlik
    learning_rate='auto', # Otomatik öğrenme hızı
    init='pca'            # PCA başlangıcı → Kümelenmeyi daha kararlı yapar
)
X_tsne = tsne.fit_transform(X_pca30)          # (Örnek Sayısı, 30) → (Örnek Sayısı, 2)
print(f"  t-SNE sonrası yeni boyut: {X_tsne.shape}")

# ── 3. Adım: Görselleştirme Hazırlığı ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))   # 10 farklı rakam sınıfı için 10 renk

# --- SOL GRAFİK: t-SNE Scatter Plot ---
for digit in range(10):
    mask = y_digits == digit                    # İlgili rakama ait noktaların maskesi
    axes[0].scatter(
        X_tsne[mask, 0],                        # t-SNE 1. Eksen
        X_tsne[mask, 1],                        # t-SNE 2. Eksen
        c=[colors[digit]],
        label=str(digit),
        alpha=0.5,                              # Noktaların üst üste bindiği yerleri görebilmek için
        s=12,                                   # Nokta boyutu
        edgecolors='none'
    )

axes[0].set_title("MNIST t-SNE — Tüm Alt Küme Dağılımı\n(Her renk bir rakam sınıfını temsil eder)",
                  fontsize=12, fontweight='bold')
axes[0].legend(title="Rakamlar", ncol=2, fontsize=9, loc='upper right')
axes[0].set_xlabel("t-SNE Boyutu 1")
axes[0].set_ylabel("t-SNE Boyutu 2")
axes[0].set_facecolor('#f9f9f9')
axes[0].grid(True, alpha=0.3)

# --- SAĞ GRAFİK: t-SNE Üzerine Gerçek MNIST Rakamlarını Yerleştirme ---
axes[1].set_facecolor('#1a1a2e')  # Koyu arka plan teması
axes[1].set_title("MNIST t-SNE — Gerçek Görüntü İzdüşümleri\n(Her kutu 28x28 boyutunda gerçek bir piksel örneğidir)",
                  fontsize=12, fontweight='bold', color='white')

# Grafiğin kalabalıklaşıp çorba olmaması için her sınıftan rastgele 12'şer örnek seçiyoruz
rng = np.random.default_rng(42)
for digit in range(10):
    mask_idx = np.where(y_digits == digit)[0]
    sample_size = min(12, len(mask_idx))  # Görsel netlik için örnek sayısı 12 olarak optimize edildi
    sample_indices = rng.choice(mask_idx, size=sample_size, replace=False)

    for idx in sample_indices:
        img = X_digits[idx]                        # 28×28 boyutundaki MNIST görüntüsü
        x, y_pos = X_tsne[idx, 0], X_tsne[idx, 1]  # Noktanın t-SNE üzerindeki koordinatları

        # Sınıfa özel renk paleti ayarı (RGB)
        colored = plt.cm.tab10(digit / 10)[:3]

        # !! KRİTİK DEĞİŞİKLİK !! 28x28 görüntü için zoom oranı 0.45 yapıldı (Kutular taşmasın diye)
        imagebox = OffsetImage(img, zoom=0.45, cmap='gray_r')

        ab = AnnotationBbox(imagebox, (x, y_pos),
                            frameon=True,
                            bboxprops=dict(
                                boxstyle='round,pad=0.05',
                                facecolor=(*colored, 0.25),   # Hafif şeffaf sınıf rengi arka planı
                                edgecolor=(*colored, 0.8),    # Belirgin kenarlık rengi
                                linewidth=1.0
                            ))
        axes[1].add_artist(ab)                       # Resmi sağdaki grafiğe yerleştir

# Dinamik eksen sınırlandırması (Kutuların kenarlara çarparak kırpılmasını önlemek için marj ±5 yapıldı)
axes[1].set_xlim(X_tsne[:, 0].min() - 5, X_tsne[:, 0].max() + 5)
axes[1].set_ylim(X_tsne[:, 1].min() - 5, X_tsne[:, 1].max() + 5)

axes[1].tick_params(colors='white')
for spine in axes[1].spines.values():
    spine.set_edgecolor('white')

plt.tight_layout()
plt.show()

print("\n💡 t-SNE Analiz Notu: MNIST uzayında benzer yazılış stillerine sahip rakamların "
      "(örneğin 3, 5 ve 8) t-SNE haritasında birbirine yakın veya komşu adacıklar oluşturduğunu, "
      "0 ve 1 gibi karakteristik rakamların ise tamamen bağımsız net kümeler halinde ayrıştığını gözlemleyebilirsiniz.")