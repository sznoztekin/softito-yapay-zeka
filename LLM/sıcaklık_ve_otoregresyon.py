"""sıcaklık-ve-otoregresyon.ipynb

"""

import re                        # metni temizlemek ve kelimelere ayırmak için düzenli ifadeler
import numpy as np                # softmax, olasılık vektörleri ve rastgele örnekleme için
import matplotlib.pyplot as plt   # olasılık dağılımlarını çubuk grafikle çizmek için

# Tekrar üretilebilirlik için sabit tohum: aynı kodu tekrar çalıştırınca
# rastgele sayılar (örnekleme adımlarında) hep aynı çıksın, sonuçlar şansa bağlı görünmesin.
rng_global = np.random.default_rng(42)

# Grafiklerin görünümünü sadeleştiren ayarlar (üst ve sağ çizgileri kaldırır, çözünürlüğü artırır)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

def softmax_with_temperature(logits, T=1.0):
    """Logit vektörünü, T sıcaklığıyla ölçekleyip olasılık dağılımına çevirir."""
    logits = np.asarray(logits, dtype=float) / T   # 1. adım: her logiti T'ye böl (sıcaklık ölçekleme)
    logits = logits - logits.max()                 # 2. adım: en büyük değeri çıkar (sayısal kararlılık, e^büyük_sayı taşmasını engeller)
    exp_logits = np.exp(logits)                     # 3. adım: her ölçeklenmiş logitin e tabanındaki üssünü al
    return exp_logits / exp_logits.sum()             # 4. adım: toplama böl -> toplamı 1 olan bir olasılık dağılımı elde et


# Örnek: modelin 5 kelime için ürettiği varsayımsal ham logitler
kelimeler = ["özgürlük","kuantum","melodi","ışık", "volkan"]
logits = [2.4, 1.8, 0.7, 0.1, -0.64]   # logitler keyfi sayılardır, henüz olasılık değildir (negatif de olabilir, toplamları 1 değildir)

# Başlık satırını yazdır: sütun başına bir kelime adı
print(f"{'Sıcaklık':<10}" + "".join(f"{k:>10}" for k in kelimeler))

# Aynı logitleri farklı sıcaklıklarla deneyip her satırda bir T değerinin sonucunu yazdır
for T in [0.3, 0.6, 1.2, 1.8, 3.5]:
    probs = softmax_with_temperature(logits, T)               # bu T için olasılık dağılımını hesapla
    print(f"T={T:<8}" + "".join(f"{p:>10.1%}" for p in probs))  # yüzde formatında satır satır yazdır

temperatures = [0.2, 1.0, 3.0]                                       # karşılaştırılacak üç sıcaklık değeri
fig, axes = plt.subplots(1, len(temperatures), figsize=(13, 3.5), sharey=True)  # yan yana 3 grafik alanı oluştur, y eksenini paylaştır

for ax, T in zip(axes, temperatures):
    probs = softmax_with_temperature(logits, T)                       # bu sıcaklık için olasılıkları hesapla
    bars = ax.bar(kelimeler, probs, color="#5DCAA5")                    # her kelime için bir çubuk çiz
    ax.set_title(f"T = {T}")
    ax.set_ylim(0, 1)                                                   # y eksenini 0-1 (yüzde 0-100) aralığında sabitle
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.0%}",  # çubuğun üstüne yüzde değerini yaz
                 ha="center", fontsize=9)

axes[0].set_ylabel("Olasılık")
fig.suptitle("Aynı logitler, farklı sıcaklıklar: dağılım nasıl şekil değiştiriyor?")
plt.tight_layout()
plt.show()

def greedy_index(probs):
    # En yüksek olasılığa sahip elemanın indeksini döndürür -> her zaman aynı seçim, rastgelelik yok
    return int(np.argmax(probs))


def temperature_sample_index(probs, rng):
    # probs dizisini ağırlık olarak kullanarak rastgele bir indeks çeker (zar atmak gibi)
    return int(rng.choice(len(probs), p=probs))


def top_k_index(probs, k, rng):
    k = min(k, len(probs))                      # k, eleman sayısından büyük olamaz
    top_idx = np.argsort(probs)[-k:]            # en yüksek k olasılığın indeksleri (küçükten büyüğe sıralayıp son k'yı al)
    top_probs = probs[top_idx]
    top_probs = top_probs / top_probs.sum()      # havuzu yeniden normalize et (toplamları yine 1 olsun)
    return int(rng.choice(top_idx, p=top_probs))  # sadece bu k'lık havuzdan örnekle


def top_p_index(probs, p, rng):
    order = np.argsort(probs)[::-1]              # büyükten küçüğe sırala
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)          # kümülatif (biriken) olasılık toplamı
    cutoff = int(np.searchsorted(cumulative, p)) + 1  # kümülatif toplam p'yi geçtiği ilk noktayı bul
    pool_idx = order[:cutoff]                     # havuz: kümülatif olasılığı p'ye getiren en olası kelimeler
    pool_probs = probs[pool_idx]
    pool_probs = pool_probs / pool_probs.sum()     # havuzu yeniden normalize et
    return int(rng.choice(pool_idx, p=pool_probs))  # sadece bu (değişken boyutlu) havuzdan örnekle


# Aynı dağılım üzerinde dört stratejiyi karşılaştıralım
probs_demo = softmax_with_temperature(logits, T=1.0)  # T=1.0: ölçeklenmemiş "ham" dağılım
rng = np.random.default_rng(7)                          # bu hücreye özel, sabit bir rastgelelik kaynağı

print("Dağılım:", dict(zip(kelimeler, np.round(probs_demo, 3))))
print("Greedy        ->", kelimeler[greedy_index(probs_demo)])
print("Temperature   ->", kelimeler[temperature_sample_index(probs_demo, rng)])
print("Top-k (k=2)   ->", kelimeler[top_k_index(probs_demo, 2, rng)])
print("Top-p (p=0.8) ->", kelimeler[top_p_index(probs_demo, 0.8, rng)])

corpus = """
Sabah trafik çok yoğundu. Sabah trafik biraz sakindi. Sabah trafik aniden tıkandı.
Tramvay erkenden geldi ve yolcuları topladı. Tramvay öğleden sonra arızalandı ve yolda kaldı. Gençler kafede oturdu ve konuştu. Gençler kafede kahve içti ve güldü. Gençler kütüphanede sessizce çalıştı. Kediler sokakta koştu ve miyavladı. Kediler kaldırımda güneşlendi. Öğle olunca acıkma hissi baş gösterdi ve restoranlar doldu. Öğle olunca meydan kalabalıklaştı. Öğle olunca herkes bir an durakladı.
Mesai birden bitti ve ofisler boşaldı. Mesai kısa sürdü ve erken çıkıldı. Mesai uzun sürdü ve herkes yoruldu. Mesai birden yoğunlaştı ve herkes koşturdu.
Müşteriler hesapları ödedi ve kalktı. Müşteriler vitrinlere dikkatle baktı. Akşam olunca sokak lambaları yandı. Akşam olunca şehir ışıl ışıl belirdi. Akşam olunca gürültü azaldı.
Herkes odasına huzur içinde çekildi. Herkes yorgun bir şekilde uyudu. Bugün şehir çok hareketliydi ama verimli bir gündü.
"""


def tokenize(text):
    text = text.replace("İ", "i").replace("I", "ı")  # Türkçe büyük/küçük harf tuzağı: Python'ın varsayılan .lower() "İ"yi yanlış çevirir
    text = text.lower()                                # tüm metni küçük harfe çevir (aksi halde "Hava" ve "hava" farklı kelime sayılır)
    text = re.sub(r"[^\w\sçğıöşüâî]", " ", text, flags=re.UNICODE)  # noktalama işaretlerini boşlukla değiştir, Türkçe harfleri koru
    return text.split()                                  # boşluklardan böl -> kelime listesi döndür


tokens = tokenize(corpus)
print(f"Toplam token sayısı: {len(tokens)}")
print(f"Benzersiz kelime sayısı: {len(set(tokens))}")
print("İlk 15 token:", tokens[:15])

# Bigram sayım tablosu: bigram_counts["hava"] = {"çok": 2, "birden": 1, ...}
bigram_counts = {}
# zip(tokens[:-1], tokens[1:]) -> her kelimeyi kendinden sonraki kelimeyle eşler:
# [bugün, hava, çok, ...] üzerinde (bugün, hava), (hava, çok), ... çiftlerini gezer
for current_word, next_word in zip(tokens[:-1], tokens[1:]):
    bigram_counts.setdefault(current_word, {})                                   # bu kelime ilk kez görülüyorsa boş bir sözlük aç
    bigram_counts[current_word][next_word] = bigram_counts[current_word].get(next_word, 0) + 1  # sayacı bir artır

# "hava" kelimesinden sonra hangi kelimeler hangi sıklıkla geliyor?
print("'mesai' kelimesinden sonra gözlenen kelimeler:")
for kelime, sayi in bigram_counts["mesai"].items():
    print(f"  {kelime!r:<12} -> {sayi} kez")