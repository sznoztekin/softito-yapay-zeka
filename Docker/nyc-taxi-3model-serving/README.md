# NYC Taxi Durasyon Tahmin API (Mikroservis Mimarisi) 🚕

Bu proje, NYC Taxi veri seti kullanılarak eğitilen farklı makine öğrenmesi modellerini bir API üzerinden sunan, **Docker** ve **Nginx** ile mikroservis mimarisine göre yapılandırılmış ölçeklenebilir bir uygulamadır.

## 🏗 Mimari Yapı
Proje, bir **API Gateway (Nginx)** arkasında çalışan üç bağımsız tahmin servisinden oluşur:
1. **Linear Regression Service**: Temel doğrusal model.
2. **XGBoost Service**: Yüksek performanslı gradyan artırma modeli.
3. **LightGBM Service**: Hızlı ve ölçeklenebilir ağaç tabanlı model.

## 🛠 Kullanılan Teknolojiler
* **FastAPI**: Hızlı ve modern web framework'ü.
* **Nginx**: İstekleri ilgili modele yönlendiren API Gateway.
* **Docker & Docker Compose**: Servislerin izolasyonu ve orkestrasyonu.
* **Scikit-learn, XGBoost, LightGBM**: Tahminleme modelleri.
* **Joblib**: Eğitilmiş modellerin seri hale getirilip saklanması.

## 📋 Kurulum ve Çalıştırma

### Gereksinimler
* Bilgisayarınızda **Docker** ve **Docker Compose** yüklü olmalıdır.

### Çalıştırma
Projenin ana dizininde terminali açın ve tüm servisleri ayağa kaldırmak için şu komutu çalıştırın:
```bash
docker-compose up --build