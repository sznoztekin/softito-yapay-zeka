# House Prices Prediction - Random Forest Modeli 🏠

Bu proje, Kaggle "House Prices" veri seti kullanılarak ev fiyatlarını tahmin etmeyi amaçlayan bir makine öğrenmesi modelidir. Proje, veri ön işleme, imputer kullanımı ve Random Forest Regressor algoritmasını içerir.

## Proje İçeriği
- **train.py**: Veri setini yükler, eksik verileri tamamlar (imputation) ve `RandomForestRegressor` ile modeli eğiterek sonuçları raporlar.
- **train.csv**: Analiz edilen ham ev fiyatları veri setidir.
- **Dockerfile**: Projenin herhangi bir ortamda çalışması için gerekli olan konteyner yapılandırmasıdır.

## Gereksinimler
Kodun çalışması için gereken kütüphaneler `requirements.txt` dosyasında belirtilmiştir:
```bash
pip install -r requirements.txt