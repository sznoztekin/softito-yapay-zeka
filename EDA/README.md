# Netflix İçerik Veri Analizi (EDA) Projesi

Bu proje, Netflix içerik verileri (2021) üzerinde Keşifçi Veri Analizi (EDA) yaparak, platformdaki film ve dizi trendlerini, içerik dağılımlarını ve veri setindeki eksiklikleri incelemektedir.

## İçerik
- **EDA-netflix.py**: Veri setini yükleyen, eksik verileri temizleyen, istatistiksel özetleri çıkaran ve çeşitli grafikler (Histogram, Boxplot, Heatmap vb.) ile veriyi görselleştiren Python kodudur.
- **netflix_titles_2021.csv**: Analiz edilen ham Netflix veri setidir.

## Özellikler
- Eksik verilerin (null) tespiti ve görselleştirilmesi.
- Kategorik ve sayısal değişkenlerin frekans analizi.
- Yayın yılları ve içerik türlerine göre trend analizi.

## Gereksinimler
Kodun çalışması için gerekli kütüphaneler:
```bash
pip install pandas numpy matplotlib seaborn scipy