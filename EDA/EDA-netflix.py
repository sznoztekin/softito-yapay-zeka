# Temel veri işleme kütüphaneleri
import pandas as pd
import numpy as np
# Görselleştirme kütüphaneleri
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
# İstatistik kütüphaneleri
from scipy import stats
import warnings

# Ayarlar
warnings.filterwarnings('ignore') # Uyarı mesajlarını kapatır
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Grafik stili ayarları
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print('Tüm kütüphaneler başarıyla yüklendi')
print(f'Pandas version: {pd.__version__}')
print(f'Numpy version: {np.__version__}')
print(f'Seaborn version: {sns.__version__}')

# Veriyi okuma
df = pd.read_csv('/content/netflix_titles_2021.csv')

# İlk 5 satır ve rastgele örnekleme
print(df.head())
print(df.sample(5))

# Veri tipleri ve null analizi
dtype_df = pd.DataFrame({
    'sütun': df.columns,
    'veri tipi': df.dtypes.values,
    'null değeri': df.isnull().sum().values,
    'null oranı(%)': (df.isnull().sum().values / len(df) * 100).round(2),
    'unıque': df.nunique().values
})
print(dtype_df.to_string(index=False))

# Temel istatistikler
print(df.describe().T)
print(df.describe(include=['object', 'category']).T)

# Eksik veri analizi görselleştirmesi
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
eksik_olan = df.isnull().sum()
eksik_olan = eksik_olan[eksik_olan > 0].sort_values(ascending=True)

colors = ['#e74c3c' if x / len(df) * 100 > 30 else '#f39c12' if x / len(df) * 100 > 10 else '#3498db'
          for x in eksik_olan.values]

axes[0].barh(eksik_olan.index, eksik_olan.values, color=colors)
axes[0].set_title('Eksik Veri Sayısı (Sütun Bazında)')
for i, (val, col) in enumerate(zip(eksik_olan.values, eksik_olan.index)):
    axes[0].text(val + 1, i, f'{val} ({val/len(df)*100:.1f}%)', va='center')

sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=axes[1])
axes[1].set_title('Eksik Veri Haritası')
plt.tight_layout()
plt.show()

# Veri temizleme
df_temiz = df.copy()
df_temiz['director'].fillna('Belirsiz', inplace=True)
df_temiz['country'].fillna(df_temiz['country'].mode()[0], inplace=True)
df_temiz['cast'].fillna('Belirsiz', inplace=True)

# Tek değişkenli analiz (Sayısal)
sayisal_kolonlar = ['release_year']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
veri = df_temiz['release_year'].dropna()

axes[0].hist(veri, bins=30, color="steelblue", edgecolor='white', alpha=0.8)
axes[0].axvline(veri.mean(), color='tomato', linestyle='dashed', label=f'Ort: {veri.mean():.2f}')
axes[0].legend()

axes[1].boxplot(veri, patch_artist=True, boxprops=dict(facecolor='lightblue'))
axes[1].set_title('Release Year - Box Plot')

veri.plot.kde(ax=axes[2], color='darkorange', linewidth=2)
plt.show()

# Kategorik analiz
kategorik_kolonlar = ['type', 'rating', 'duration', 'country']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, kolon in enumerate(kategorik_kolonlar):
    degerler = df_temiz[kolon].value_counts().head(5) if kolon in ['country', 'duration', 'rating'] else df_temiz[kolon].value_counts()
    axes[i].bar(degerler.index.astype(str), degerler.values, color=sns.color_palette('husl', len(degerler)))
    axes[i].set_title(f'{kolon} - Frekans Dağılımı')
    plt.setp(axes[i].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()