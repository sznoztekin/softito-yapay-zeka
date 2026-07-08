# Görüntü İşleme ve Makine Öğrenmesi Analizleri 🐼

Bu depo, görüntü işleme temelleri ile MNIST veri seti üzerinde çeşitli sınıflandırma ve boyut indirgeme yöntemlerini inceleyen kapsamlı bir çalışma setidir.

## 📁 Proje İçeriği
- **görüntü_işleme_panda.py**: OpenCV kullanarak görüntü üzerinde nesne tespiti, kontur analizi ve ağırlık merkezi belirleme (Panda görseli üzerinde) çalışmalarını içerir.
- **görüntü_işleme.py**: MNIST veri seti üzerinde PCA ve t-SNE kullanarak boyut indirgeme ve SVM/KNN/MLP modelleri ile el yazısı rakam sınıflandırma uygulamalarını içerir.
- **panda.jpeg**: Görüntü işleme denemelerinde kullanılan örnek görsel.

## 🛠 Kullanılan Teknolojiler
* **OpenCV (cv2)**: Görüntü filtreleme, kenar bulma ve kontur analizi.
* **Scikit-learn**: MNIST sınıflandırma modelleri (SVC, KNN, MLP) ve boyut indirgeme (PCA, t-SNE).
* **Matplotlib & Seaborn**: Veri görselleştirme ve karmaşıklık matrisleri.
* **Scipy**: Görüntü filtreleri (Gaussian, Sobel, Median).

## 📋 Öne Çıkan Çalışmalar
1. **Nesne Analizi**: Görüntü üzerinde konturların çıkarılması, dikdörtgen sınırlayıcı kutuların çizilmesi ve nesne merkezlerinin tespiti.
2. **Boyut İndirgeme**: MNIST gibi yüksek boyutlu (784 piksel) verilerin PCA ve t-SNE ile 2D/3D düzleme yansıtılarak görselleştirilmesi.
3. **Sınıflandırma**: El yazısı rakamların farklı algoritmalarla (SVM, KNN, MLP) tahmin edilmesi ve başarı metriklerinin (Confusion Matrix) raporlanması.

## 🚀 Kullanım
Projedeki analizleri çalıştırmak için gerekli kütüphaneleri yükleyin:
```bash
pip install numpy matplotlib opencv-python scikit-learn seaborn scipy pillow