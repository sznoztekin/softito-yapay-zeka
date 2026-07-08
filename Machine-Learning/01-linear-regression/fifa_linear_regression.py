import pandas as pd
import numpy as np
import matplotlib
# MacOS için en stabil pencere yöneticisi
matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- AYARLAR ---
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')

# --- DOSYA YÜKLEME ---
file_path = 'fifa_wc_mens_match_dataset_1970_2022.csv'
if not os.path.exists(file_path):
    print(f"HATA: '{file_path}' dosyası bulunamadı.")
    exit()
df = pd.read_csv(file_path)

# --- VERİ HAZIRLIĞI ---
target = 'goals_for'
features = ['shots', 'shots_on_target', 'possession', 'passes_completed']
df = df[features + [target]].dropna()
print(f"Temizlikten sonra kalan veri sayısı: {df.shape[0]}")

# --- REGRESYON MODELLERİ ---
X_simple = df[['shots_on_target']].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
model_simple = LinearRegression().fit(X_train, y_train)

X_multi = df[features].values
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
model_multi = LinearRegression().fit(X_train_m, y_train_m)
y_pred_test_m = model_multi.predict(X_test_m)

# --- GÖRSELLEŞTİRME (INTERAKTİF PENCERELER) ---

# 1. Artık Analizi
plt.figure(figsize=(10, 4))
plt.scatter(y_pred_test_m, y_test_m - y_pred_test_m)
plt.axhline(0, color='red', linestyle='--')
plt.title('Artık Analizi (Hata Dağılımı)')
plt.show() # Grafik penceresi açılacak

# 2. Özellik Önem Sırası
plt.figure(figsize=(10, 6))
sns.barplot(x=model_multi.coef_, y=features)
plt.title('Özelliklerin Gol Üzerindeki Etkisi')
plt.show() # Grafik penceresi açılacak

# 3. Tahmin Başarısı
plt.figure(figsize=(10, 6))
plt.scatter(y_test_m, y_pred_test_m, alpha=0.5)
plt.plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], 'r--')
plt.title('Çoklu Regresyon: Gerçek vs Tahmin')
plt.show() # Grafik penceresi açılacak

print("Tüm grafikler pencerelerde görüntülendi. Pencereyi kapattığında bir sonrakine geçecektir.")