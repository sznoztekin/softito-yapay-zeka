"""perfume_recommender.ipynb
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Görselleştirmelerin Colab'de düzgün görüntülenmesi için seaborn teması
sns.set_theme(style="whitegrid")

class PerfumeDataLoader:
    """
    Parfumo veri setini yükleyen, eksik/hatalı verileri temizleyen
    ve model için gerekli metin özelliklerini birleştiren sınıf.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """CSV dosyasını okur."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[HATA] '{self.file_path}' dosyası bulunamadı! Lütfen Colab'e yüklediğinizden emin olun.")

        try:
            self.df = pd.read_csv(self.file_path)
            print(f"[INFO] Veri başarıyla yüklendi. Toplam Satır: {self.df.shape[0]}, Sütun Sayısı: {self.df.shape[1]}")
            return self.df
        except Exception as e:
            print(f"[HATA] Veri okunurken bir problem oluştu: {e}")
            return None

    def clean_data(self):
        """Veri setindeki N/A değerleri ve eksik alanları temizler."""
        if self.df is None:
            raise ValueError("Veri henüz yüklenmedi. Önce 'load_data()' fonksiyonunu çağırın.")

        # Parfumo veri setindeki "N/A" string ifadelerini gerçek NaN değerlerine dönüştürür
        self.df = self.df.replace("N/A", np.nan)

        # Öneri için kritik olan metin sütunlarındaki eksik değerleri boş string ile doldurur
        text_cols = ['Main_Accords', 'Top_Notes', 'Middle_Notes', 'Base_Notes', 'Brand', 'Name', 'Concentration']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str).str.strip()

        # Sayısal alanları dönüştürür ve hatalı kayıtları güvenli hale getirir
        if 'Rating_Value' in self.df.columns:
            self.df['Rating_Value'] = pd.to_numeric(self.df['Rating_Value'], errors='coerce').fillna(0.0)
        if 'Rating_Count' in self.df.columns:
            self.df['Rating_Count'] = pd.to_numeric(self.df['Rating_Count'], errors='coerce').fillna(0).astype(int)
        if 'Release_Year' in self.df.columns:
            self.df['Release_Year'] = pd.to_numeric(self.df['Release_Year'], errors='coerce').fillna(0).astype(int)

        print("[INFO] Veri temizleme ve tip dönüşümleri başarıyla tamamlandı.")
        return self.df

    def sample_data(self, fraction=1.0, random_state=42):
        """Veri setini örnekler (RAM sorunlarını önlemek için)."""
        if self.df is None:
            raise ValueError("Veri seti mevcut değil. Önce 'load_data()' ve 'clean_data()' fonksiyonlarını çağırın.")

        if fraction < 1.0:
            initial_rows = self.df.shape[0]
            self.df = self.df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
            print(f"[INFO] Veri seti %{fraction*100:.0f} oranında örneklendi. {initial_rows} satırdan {self.df.shape[0]} satıra düşürüldü.")
        else:
            print("[INFO] Veri seti örneklenmedi (fraction=1.0).")
        return self.df

    def create_features_soup(self):
        """Parfümün koku profilini oluşturmak için tüm notaları ve akorları birleştirir."""
        if self.df is None:
            raise ValueError("Veri seti mevcut değil.")

        # Akorlar, markalar ve notalar bir araya getirilerek zengin bir içerik metni oluşturulur
        def combine_features(row):
            return f"{row['Main_Accords']} {row['Top_Notes']} {row['Middle_Notes']} {row['Base_Notes']} {row['Brand']}"

        self.df['soup'] = self.df.apply(combine_features, axis=1)
        self.df['soup'] = self.df['soup'].str.lower() # Büyük/küçük harf duyarlılığını kaldırma
        print("[INFO] Model girdisi için koku profili öznitelikleri ('soup') oluşturuldu.")
        return self.df

class PerfumeRecommenderModel:
    """
    Metin verilerini TF-IDF matrisine dönüştüren, kosinüs benzerliğini hesaplayan
    ve benzerlik skorlarına göre öneri üreten sınıf.
    """
    def __init__(self, cleaned_df):
        self.df = cleaned_df
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000) # Further reduced max_features
        self.tfidf_matrix = None
        self.cosine_sim = None

    def train(self):
        """Koku profillerini vektörleştirir ve benzerlik matrisini eğitir."""
        if 'soup' not in self.df.columns:
            raise ValueError("DataFrame içinde 'soup' sütunu bulunamadı. Önce özellikleri hazırlayın.")

        print("[INFO] TF-IDF vektörleştirme işlemi başlatılıyor...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['soup'])

        print("[INFO] Kosinüs Benzerliği matrisi hesaplanıyor...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("[INFO] Model başarıyla eğitildi ve benzerlikler hafızaya alındı.")

    def recommend(self, perfume_name, top_n=5):
        """Kullanıcının seçtiği parfüme en benzer top_n adet parfümü bulur."""
        if self.cosine_sim is None:
            raise ValueError("Model henüz eğitilmedi. Önce 'train()' metodunu çağırın.")

        # Arama kolaylığı için isimleri küçük harfe çevirelim
        self.df['Name_lower'] = self.df['Name'].str.lower()
        perfume_name_lower = perfume_name.lower()

        # Tam eşleşme arama
        matching_indices = self.df[self.df['Name_lower'] == perfume_name_lower].index

        # Tam eşleşme yoksa kısmi arama (içinde geçiyor mu kontrolü)
        if len(matching_indices) == 0:
            matching_indices = self.df[self.df['Name_lower'].str.contains(perfume_name_lower)].index
            if len(matching_indices) == 0:
                print(f"[UYARI] '{perfume_name}' isimli veya benzeri bir parfüm veri setinde bulunamadı.")
                return pd.DataFrame(), perfume_name

        # Bulunan ilk eşleşen parfümün indeksini baz alalım
        idx = matching_indices[0]
        target_perfume_real_name = self.df.iloc[idx]['Name']

        # Hedef parfümün tüm parfümlerle olan benzerlik skorlarını çekme
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Skorları en yüksekten en düşüğe sıralama
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Parfümün kendisini listeden çıkarma
        sim_scores = [score for score in sim_scores if score[0] != idx]

        # En benzer top_n adet parfümü seçme
        top_scores = sim_scores[:top_n]

        # Önerilen parfümlerin bilgilerini toplama
        recommendations = []
        for index, score in top_scores:
            rec_row = self.df.iloc[index].copy()
            rec_row['Similarity_Score'] = score
            recommendations.append(rec_row)

        rec_df = pd.DataFrame(recommendations)
        return rec_df, target_perfume_real_name

class PerfumeVisualizer:
    """
    Öneri sonuçlarını ekrana düzenli basan ve
    benzerlik skorlarını grafik olarak gösteren sınıf.
    """
    def __init__(self):
        pass

    def display_text_output(self, target_name, recommendations_df):
        """Sonuçları konsol ekranında listeler."""
        if recommendations_df.empty:
            print("Görüntülenecek herhangi bir öneri bulunamadı.")
            return

        print("\n" + "="*65)
        print(f"--- '{target_name.upper()}' İÇİN PARFÜM ÖNERİLERİ ---")
        print("="*65)

        for i, row in enumerate(recommendations_df.itertuples(), 1):
            year_str = str(int(row.Release_Year)) if row.Release_Year > 0 else "Bilinmiyor"
            rating_str = f"{row.Rating_Value}/10 ({int(row.Rating_Count)} Oy)" if row.Rating_Count > 0 else "Puan Yok"

            print(f"{i}. PARFÜM: {row.Name}")
            print(f"   Marka / Ev: {row.Brand} | Çıkış Yılı: {year_str}")
            print(f"   Konsantrasyon: {row.Concentration} | Puan: {rating_str}")
            print(f"   Baskın Akorlar: {row.Main_Accords}")
            print(f"   Koku Notaları: Üst: {row.Top_Notes} | Orta: {row.Middle_Notes} | Alt: {row.Base_Notes}")
            print(f"   Koku Benzerlik Oranı: %{row.Similarity_Score * 100:.2f}")
            print("-" * 65)

    def plot_similarity_scores(self, target_name, recommendations_df, save_filename="parfum_onerileri.png"):
        """Önerilen parfümlerin benzerlik yüzdelerini yatay bar grafik olarak gösterir ve kaydeder."""
        if recommendations_df.empty:
            print("[UYARI] Grafik çizmek için veri bulunamadı.")
            return

        # Grafikte en yüksek benzerlik en üstte görünsün diye veriyi sıralıyoruz
        sorted_df = recommendations_df.sort_values(by='Similarity_Score', ascending=True)

        plt.figure(figsize=(10, 5))
        colors = sns.color_palette("viridis", len(sorted_df))

        # Yatay bar çizimi
        bars = plt.barh(sorted_df['Name'], sorted_df['Similarity_Score'] * 100, color=colors, edgecolor='grey')

        # Barların üzerine yüzde değerlerini yazma
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"%{width:.1f}",
                     va='center', ha='left', fontsize=10, fontweight='bold')

        plt.title(f"'{target_name}' Parfümüne En Benzer Kokular ve Oranları", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Benzerlik Oranı (%)", fontsize=11)
        plt.ylabel("Önerilen Parfümler", fontsize=11)
        plt.xlim(0, 115)
        plt.tight_layout()

        # Grafik dosyaya kaydedilir ve Colab hücresinde gösterilir
        plt.savefig(save_filename, dpi=300)
        plt.show()
        print(f"[INFO] Analiz grafiği '{save_filename}' adıyla kaydedildi.")

# -----------------------------------------------------------------
# ÇALIŞTIRMA AYARLARI
# -----------------------------------------------------------------
# Veri setini sol menüden doğrudan yüklediyseniz aynen bırakın.
# Google Drive bağladıysanız "/content/drive/MyDrive/Parfumo_Perfumes.csv" şeklinde güncelleyin.
csv_file_name = "Parfumo_Perfumes.csv"

# Öneri almak istediğiniz örnek parfüm ismi (Örn: Aventus, Sauvage, Eros vb.)
secilen_parfum = "Aventus"

# RAM sorunlarını önlemek için veri setinin ne kadarının kullanılacağını belirleyin (0.1 = %10)
# Küçük veri setleri için 1.0 (tamamını kullan) olarak bırakılabilir.
sample_fraction = 0.1

# -----------------------------------------------------------------
# PIPELINE (AKIŞ) TETİKLENMESİ
# -----------------------------------------------------------------

# 1. Adım: Veri Yükleme ve Temizleme Sınıfı Çalışıyor
data_loader = PerfumeDataLoader(csv_file_name)
data_loader.load_data()
data_loader.clean_data()
cleaned_df = data_loader.sample_data(fraction=sample_fraction)
cleaned_df = data_loader.create_features_soup()

# 2. Adım: Model Sınıfı Çalışıyor ve Eğitiliyor
recommender_model = PerfumeRecommenderModel(cleaned_df)
recommender_model.train()

# Belirttiğimiz parfüm için en benzer 5 parfümü istiyoruz
recommendations_df, real_target_name = recommender_model.recommend(secilen_parfum, top_n=5)

# 3. Adım: Görselleştirme ve Çıktı Sınıfı Sonuçları Basıyor
visualizer = PerfumeVisualizer()
visualizer.display_text_output(real_target_name, recommendations_df)
visualizer.plot_similarity_scores(real_target_name, recommendations_df, save_filename="parfum_onerileri_raporu.png")