"""TF_İDF_haber.ipynb

"""

import math
import sys
import warnings
warnings.filterwarnings('ignore')

kucuk_dokumanlar=[
"""İstanbulda yaşarken şehir hayatının stresli ortamından uzaklaşmak için ailesiyle Bayburt Yedigözeler köyüne yerleşen Feryal Haşlak tarhun üretimiyle ekonomiye katkıda bulunuyor.
Bartınlı Feryal Haşlak İbrahim Haşlak evlendikten sonra İstanbul yerleşti.
Uzun yıllar bu şehirde gıda toptancılığı yapan Haşlak çifti 8 yıl önce köye dönmeye karar verdi.
İbrahim Haşlakın baba ocağı merkeze bağlı Yedigözeler köyüne yerleşen aile ata yadigarı arazileri değerlendirmek için harekete geçti.
Feryal Haşlak AA muhabirine 8 yıl önce yerleştiği köyde önce hobi amaçlı sebze yetiştirmeye başladığını söyledi.
Daha sonra atıl durumdaki yaklaşık 20 dönümlük arazilerini değerlendirmek istediğini belirten Haşlak köydeki akraba ve komşularından çiftçilik konusunda bilgi aldığını dile getirdi.
Haşlak, Bayburtta kadınlar tarafından yetiştirilen tarhunun üretimine yöneldiğini belirterek Ben sadece bu ürünün sesi olmaya çalıştım.
Üretimden işlemeye tarladan paketlemeye kadar her aşamada kadınlarımızla birlikte çalışıyoruz ve onların emeğini görünür kılıyoruz.
Güzel işler yaptık.
Son 4 senedir tarhunun ihracatını yapıyoruz.
Köylerimizdeki ve komşu köylerdeki üreticilerle birlikte çalışıyoruz.
Gençlerimizin köylerindeki boş arazileri değerlendirmesi için örnek olmaya çalışıyorum.
Köyünde üretim yapmak isteyen kadınlara da gidip başkasının yanında çalışmak yerine kendi üretiminizi yapın diyorum.
ifadesini kullandı."""
]

def tokenize(text):
    return text.lower().split()

tokenize_kucuk = [tokenize(d) for d in kucuk_dokumanlar]
print("Tokenize edilmiş dokümanlar:" ,(kucuk_dokumanlar))

def tf(term, tokenize_doc):
    count = tokenize_doc.count(term)
    return count / len(tokenize_doc) if len(tokenize_doc) > 0 else 0

def idf(term, tokenize_dokumanlar):
    doc_count = sum(1 for doc in tokenize_dokumanlar if term in doc)
    return math.log(len(tokenize_dokumanlar) / doc_count)

print("TF-IDF Değerleri:\n")
for doc in tokenize_kucuk:
  for term in sorted(set(doc)):
        tf_val = tf(term, doc)
        idf_val = idf(term, tokenize_kucuk)
        tfidf_val = tf_val * idf_val
        print(f"  '{term}' -> TF: {tf_val:.4f}, IDF: {idf_val:.4f}, TF-IDF: {tfidf_val:.4f}")
print()