"""word2vec-glove-fasttext.ipynb

"""

import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec, FastText
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

print("✅ Kütüphaneler yüklendi!")

sentences = [
    ["kral", "erkek", "lider", "taht", "saray"],
    ["kraliçe", "kadın", "lider", "taç", "saray"],
    ["adam", "erkek", "çalışıyor", "koşuyor", "güçlü"],
    ["kadın", "çalışıyor", "koşuyor", "güçlü"],
    ["çocuk", "oynuyor", "park", "mutlu"],
    ["kız", "çocuk", "oynuyor", "gülüyor"],
    ["erkek", "çocuk", "oynuyor", "koşuyor"],
    ["köpek", "havlıyor", "koşuyor", "hızlı"],
    ["kedi", "miyavlıyor", "uyuyor", "sessiz"],
    ["kuş", "uçuyor", "gökyüzü", "özgür"],
    ["paris", "fransa", "şehir", "avrupa"],
    ["berlin", "almanya", "şehir", "avrupa"],
    ["ankara", "türkiye", "şehir", "başkent"],
    ["istanbul", "türkiye", "şehir", "boğaz"],
    ["doktor", "hastane", "tedavi", "hasta"],
    ["hemşire", "hastane", "bakım", "hasta"],
    ["öğretmen", "okul", "ders", "öğrenci"],
    ["öğrenci", "okul", "öğreniyor", "ders"],
    ["mühendis", "şirket", "proje", "teknoloji"],
    ["yazılımcı", "bilgisayar", "kod", "teknoloji"],
    ["araba", "yol", "sürücü", "hız"],
    ["otobüs", "şehir", "yolcu", "ulaşım"],
    ["uçak", "havaalanı", "seyahat", "gökyüzü"],
    ["tren", "istasyon", "yolculuk", "ray"]
]

model_sg=Word2Vec(
    sentences,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,
    epochs=200,
    seed=42
)
print("✅ Modeller eğitildi!")
print(f"Kelime hazinesi boyutu: {len(model_sg.wv)}")
print(f"Vektör boyutu: {model_sg.wv.vector_size}")

kelime = "doktor"
vektor = model_sg.wv[kelime]
print(f"'{kelime}' kelimesinin vektörü (ilk 10 eleman):")
print(np.round(vektor[:10], 3))
print(f"\nVektör şekli: {vektor.shape}")

# En benzer kelimeler
print("\n--- 'doktor' kelimesine en benzer kelimeler ---")
benzerler = model_sg.wv.most_similar("doktor", topn=5)
for kelime, skor in benzerler:
    print(f"  {kelime:15s}  benzerlik: {skor:.4f}")

ciftler = [
    ("doktor", "hemşire"),
    ("öğretmen", "öğrenci"),
    ("paris", "berlin"),
    ("kedi", "köpek"),
    ("doktor", "uçak")
]

print("Kelime çiftleri arasındaki benzerlik:")
print(f"{'Çift':<30} {'Benzerlik':>10}")
print("-" * 42)
for k1, k2 in ciftler:
    skor = model_sg.wv.similarity(k1, k2)
    bar = "█" * int(skor * 20)
    print(f"({k1}, {k2}){'':<15} {skor:>6.4f}  {bar}")

# Vektör aritmetiği
print("=== Vektör Aritmetiği ===\n")

# Kral - Erkek + Kadın = ?
sonuc = model_sg.wv.most_similar(
    positive=["doktor", "hemşire"],  # ekle
    negative=["erkek"],          # çıkar
    topn=3
)
print("doktor − Erkek + Kadın = ?")
for kelime, skor in sonuc:
    print(f"  → {kelime:15s}  ({skor:.4f})")

print()

# Paris - Fransa + Türkiye = ?
sonuc2 = model_sg.wv.most_similar(
    positive=["paris", "türkiye"],
    negative=["fransa"],
    topn=3
)
print("Paris − Fransa + Türkiye = ?")
for kelime, skor in sonuc2:
    print(f"  → {kelime:15s}  ({skor:.4f})")

# Vektör uzayını 2D'de görselleştir (PCA ile boyut indirgeme)
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Görselleştirilecek kelimeler (senin korpusuna uygun)
kelimeler = [
    "doktor", "hemşire", "hasta", "hastane",
    "öğretmen", "öğrenci", "okul", "ders",
    "paris", "berlin", "ankara", "istanbul",
    "kedi", "köpek"
]

vektorler = np.array([model_sg.wv[k] for k in kelimeler])

# PCA ile 2 boyuta indir
pca = PCA(n_components=2, random_state=42)
vek_2d = pca.fit_transform(vektorler)

# Renk grupları
renkler = {
    "saglik":   ("#D85A30", ["doktor", "hemşire", "hasta", "hastane"]),
    "egitim":   ("#1D9E75", ["öğretmen", "öğrenci", "okul", "ders"]),
    "sehir":    ("#185FA5", ["paris", "berlin", "ankara", "istanbul"]),
    "hayvan":   ("#BA7517", ["kedi", "köpek"]),
}

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor("#f8f8f6")
fig.patch.set_facecolor("#f8f8f6")

for grup, (renk, grup_kelimeleri) in renkler.items():
    for kw in grup_kelimeleri:
        idx = kelimeler.index(kw)
        x, y = vek_2d[idx]

        ax.scatter(x, y, color=renk, s=120, zorder=5)
        ax.annotate(
            kw,
            (x, y),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=11,
            color=renk,
            fontweight="bold"
        )

# Analoji okları (örnek ilişkiler)
for k1, k2 in [
    ("doktor", "hemşire"),
    ("öğretmen", "öğrenci"),
]:
    i1, i2 = kelimeler.index(k1), kelimeler.index(k2)
    ax.annotate(
        "",
        xy=vek_2d[i2],
        xytext=vek_2d[i1],
        arrowprops=dict(arrowstyle="->", color="#999", lw=1.2, linestyle="dashed")
    )

ax.set_title("Kelime Vektörleri — 2D PCA Görünümü", fontsize=13, pad=12)
ax.set_xlabel("Bileşen 1")
ax.set_ylabel("Bileşen 2")
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("vektor_uzayi.png", dpi=150, bbox_inches="tight")
plt.show()

print("✅ Grafik kaydedildi: vektor_uzayi.png")

# ── GloVe'u simüle eden küçük örnek (Türkçe) ──

print("GloVe ile 'doktor' için beklenen benzerler:")
demo_sonuclar = [
    ("hemşire",   0.902),
    ("hasta",     0.885),
    ("hastane",   0.867),
    ("tedavi",    0.842),
    ("bakım",     0.821),
]

for kw, skor in demo_sonuclar:
    bar = "█" * int(skor * 25)
    print(f"  {kw:20s} {skor:.3f}  {bar}")

print()
print("Word2Vec vs GloVe Karşılaştırması (Türkçe korpus):")
print(f"{'Özellik':<30} {'Word2Vec':^15} {'GloVe':^15}")
print("-" * 62)

satirlar = [
    ("Eğitim yöntemi",    "yerel pencere",   "global matris"),
    ("Hız",               "hızlı",           "orta"),
    ("Küçük veri",        "iyi",             "zayıf"),
    ("Büyük veri",        "orta",            "çok iyi"),
    ("Anlamsal ilişki",   "iyi",             "çok iyi"),
]

for ozellik, wv, gl in satirlar:
    print(f"  {ozellik:<28} {wv:^15} {gl:^15}")

sentences_tr = [
    ["futbol", "oynamak", "güzel", "spor"],
    ["futbolcu", "çok", "hızlı", "sahada"],
    ["basketbol", "oynamak", "eğlenceli", "spor"],
    ["basketbolcu", "zıplıyor", "basket", "attı"],
    ["yüzmek", "spor", "havuz", "yüzücü"],
    ["koşmak", "hızlı", "maraton", "koşucu"],
    ["kitap", "okumak", "öğrenmek", "bilgi"],
    ["kitaplık", "kitaplar", "raflar", "kütüphane"],
]

# FastText modeli
ft_model = FastText(
    sentences_tr,
    vector_size=50,
    window=3,
    min_count=1,
    epochs=200,
    min_n=2,       # minimum n-gram uzunluğu
    max_n=5,       # maksimum n-gram uzunluğu
    seed=42
)

print("✅ FastText modeli eğitildi!")
print(f"Kelime hazinesi: {len(ft_model.wv)} kelime\n")

# Bilinen kelimeler
print("'futbol' için en benzer kelimeler:")
for kw, skor in ft_model.wv.most_similar("futbol", topn=4):
    print(f"  {kw:20s} {skor:.4f}")

# FastText'in süper gücü: bilinmeyen kelimeler!
print("=== FastText'in Süper Gücü: OOV Kelimeleri ===\n")

test_kelimeleri = [
    ("futbolculuk",    "eğitimde YOK — ama alt-kelimelerden tahmin eder"),
    ("basketbolsever", "eğitimde YOK — ama tahmin eder"),
    ("futbolcular",    "eğitimde YOK — ama tahmin eder"),
    ("futbol",         "eğitimde VAR"),
]

for kw, aciklama in test_kelimeleri:
    try:
        vektor = ft_model.wv[kw]
        norm = np.linalg.norm(vektor)
        print(f"✅ '{kw}'")
        print(f"   {aciklama}")
        print(f"   Vektör normu: {norm:.3f}  (sıfır değil = anlamlı vektör)")
        print()
    except KeyError:
        print(f"❌ '{kw}' bulunamadı\n")

# Word2Vec aynı kelimeyi bulamaz
print("--- Word2Vec için aynı test ---")
try:
    _ = model_sg.wv["futbolculuk"]
    print("✅ 'futbolculuk' bulundu")
except KeyError:
    print("❌ Word2Vec 'futbolculuk' kelimesini bilmiyor (eğitimde yoktu)")

# N-gram karşılaştırma görseli
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor("#f8f8f6")

# Sol: FastText n-gram parçaları
ax = axes[0]
ax.set_facecolor("#f8f8f6")
kelime_ornek = "futbol"
ngrams = []
n = 3
padded = "<" + kelime_ornek + ">"
for i in range(len(padded) - n + 1):
    ngrams.append(padded[i:i+n])

renkler_ng = plt.cm.Purples(np.linspace(0.3, 0.9, len(ngrams)))
for i, (ng, renk) in enumerate(zip(ngrams, renkler_ng)):
    ax.barh(i, 1, color=renk, edgecolor="white", linewidth=2)
    ax.text(0.5, i, f'"{ng}"', va="center", ha="center", fontsize=13,
            fontweight="bold", color="white")

ax.set_xlim(0, 1.2)
ax.set_yticks([])
ax.set_title(f'"{kelime_ornek}" → 3-gram parçaları', fontsize=12)
ax.set_xlabel("FastText bu parçaların hepsini öğrenir")
ax.spines[["top","right","left","bottom"]].set_visible(False)
ax.set_xticks([])

# Sağ: Model karşılaştırma tablosu (bar chart)
ax2 = axes[1]
ax2.set_facecolor("#f8f8f6")

modeller = ["Word2Vec", "GloVe", "FastText"]
ozellikler = {
    "Hız":               [9, 6, 6],
    "Büyük corpus":      [8, 9, 8],
    "Yeni kelimeler":    [0, 0, 9],
    "Eklemeli dil":      [4, 4, 9],
}

x = np.arange(len(modeller))
genislik = 0.18
renkler_bar = ["#7F77DD", "#1D9E75", "#D85A30", "#185FA5"]

for i, (ozellik, skorlar) in enumerate(ozellikler.items()):
    offset = (i - 1.5) * genislik
    bars = ax2.bar(x + offset, skorlar, genislik * 0.9,
                   label=ozellik, color=renkler_bar[i], alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels(modeller, fontsize=11)
ax2.set_yticks([])
ax2.set_ylim(0, 12)
ax2.set_title("Model Karşılaştırması (10 = en iyi)", fontsize=12)
ax2.legend(loc="upper left", fontsize=9)
ax2.spines[["top","right","left"]].set_visible(False)
ax2.grid(axis="y", alpha=0.2, linestyle="--")

plt.tight_layout()
plt.savefig("model_karsilastirma.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Grafik kaydedildi: model_karsilastirma.png")