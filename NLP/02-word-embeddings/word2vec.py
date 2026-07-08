"""word2vec.ipynb

"""

import re
from collections import Counter

paragraf="""Eski bir kütüphanenin tozlu rafları arasında dolaşırken, zamanın dışına çıkmış gibi hissetmek işten bile değildi. Dışarıda sağanak bir yağmur İstanbul'un sokaklarını yıkarken, içeride sadece sararmış sayfaların kokusu ve ahşap zeminin hafif gıcırtısı duyuluyordu. Köşedeki küçük masaya bırakılmış, kapağı aşınmış kalın bir defter dikkat çekiyordu. Kim bilir hangi unutulmuş hikayeyi, hangi yarım kalmış projeyi ya da sadece bir akşamüstü karalanmış rastgele düşünceleri barındırıyordu içinde. Sayfalardan birini rastgele çevirdiğimde, mürekkebin kokusu sanki geçmişten gelen bir fısıltı gibi odaya yayıldı."""

# 2. Metin Temizleme (Noktalama işaretlerini kaldırıp tüm kelimeleri küçük harfe çeviriyoruz)
# Böylece "kütüphanenin," ile "kütüphanenin" veya "Eski" ile "eski" aynı kelime olarak sayılır.
temiz_metin = re.sub(r'[^\w\s\']', '', paragraf.lower())
kelimeler = temiz_metin.split()

# 3. İstatistiklerin Hesaplanması
cümleler = [c.strip() for c in paragraf.split('.') if c.strip()]
benzersiz_kelimeler = set(kelimeler)
kelime_sayilari = Counter(kelimeler)

# 4. Sonuçların Ekrana Yazdırılması
print(f"Korpus boyutu: {len(cümleler)} cümle")
print(f"Toplam kelime sayısı: {len(kelimeler)}")
print(f"Benzersiz kelime sayısı: {len(benzersiz_kelimeler)}")

print("\nİlk 3 cümle:")
for i, cumle in enumerate(cümleler[:3]):
    print(f"  {i+1}. {cumle}.")

print(f"\nEn sık geçen 10 kelime: {kelime_sayilari.most_common(10)}")

# Cümleleri ayırıyoruz (Noktaya göre split)
korpus = [c.strip() for c in paragraf.split('.') if c.strip()]

# 2. Tokenize ve Kelime Dağarcığı Oluşturma (Senin fonksiyonun - Noktalama temizliği eklendi)
def tokenize(text):
    # Kelimelerin sonundaki nokta, virgül gibi işaretleri temizleyerek split yapıyoruz
    temiz_metin = re.sub(r'[^\w\s\']', '', text.lower())
    return temiz_metin.split()

tokenize_korpus = [tokenize(doc) for doc in korpus]

# Benzersiz kelimeleri sıralı olarak listeleme ve indeksleme
kelimeler = sorted(set(w for doc in tokenize_korpus for w in doc))
kelime_idx = {w: i for i, w in enumerate(kelimeler)}
idx_kelime = {i: w for w, i in kelime_idx.items()}
V = len(kelimeler)

print(f"Kelime dağarcığı boyutu (V): {V}")
print(f"Kelimeler: {kelimeler}\n")
print("-" * 50)

# 3. Komşu (Context) Kelimeleri Çıkarma (Senin fonksiyonun aynen korunarak)
def get_context_pairs(tokenize_korpus, window_size=2):
    pairs = []
    for doc in tokenize_korpus:
        for i, hedef in enumerate(doc):
            bas = max(0, i - window_size)
            son = min(len(doc), i + window_size + 1)
            for j in range(bas, son):
                if i != j:
                    pairs.append((hedef, doc[j]))
    return pairs

context_pairs = get_context_pairs(tokenize_korpus, window_size=2)
print(f"\nToplam (hedef, komşu) çifti: {len(context_pairs)}")
print("İlk 10 çift:")
for pair in context_pairs[:10]:
    print(f"  '{pair[0]}' -> '{pair[1]}'")