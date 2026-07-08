# Diyabet ML Analiz Paneli 🩺

Bu proje, Scikit-learn kütüphanesindeki `load_diabetes` veri seti üzerinde çeşitli regresyon modellerinin performansını karşılaştıran bir **Streamlit** dashboard'udur.

## 🚀 Proje Hakkında
Proje, farklı makine öğrenmesi algoritmalarını (Linear, Ridge, Random Forest, Gradient Boosting) aynı veri seti üzerinde test eder, sonuçları metrik (R2, RMSE, MAE) bazında tablolar ve görselleştirir.

## 🛠 Kullanılan Teknolojiler
* **Python 3.10**
* **Streamlit**: İnteraktif web arayüzü.
* **Scikit-learn**: Makine öğrenmesi modelleri ve metrikler.
* **Pandas & Numpy**: Veri işleme ve hesaplamalar.
* **Docker**: Konteyner tabanlı dağıtım.

## 📋 Kurulum ve Çalıştırma

### 1. Yerel Ortamda Çalıştırma
Öncelikle gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt