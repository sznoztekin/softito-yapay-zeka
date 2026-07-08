# Yaya Tespiti ve Bölütleme (Pedestrian Detection & Segmentation) 🚶‍♂️

Bu proje, görüntü işleme ve derin öğrenme tekniklerini kullanarak yayaları tespit etmeyi ve maskelemeyi amaçlayan farklı mimarileri içeren bir araştırma deposudur.

## 🏗 Proje İçeriği
Bu repo, temel CNN'den gelişmiş R-CNN ve U-Net yapılarına kadar farklı yaklaşımları içerir:

- **cnn.py**: İnsan var/yok sınıflandırması yapan temel CNN mimarisi.
- **r-cnn.py**: Mask R-CNN kullanarak nesne tespiti ve örnek bölütleme (instance segmentation) eğitimi.
- **u-net.py**: Yaya bölgelerini piksel düzeyinde ayıran U-Net mimarisi.
- **test_rcnn.py**: Eğitilmiş bir Mask R-CNN modelini kullanarak yeni resimler üzerinde tahmin yapma betiği.

## 🛠 Kullanılan Teknolojiler
* **PyTorch & Torchvision**: Model mimarileri ve eğitim döngüsü.
* **PIL & NumPy**: Görüntü işleme ve maske manipülasyonları.
* **Mask R-CNN & U-Net**: Gelişmiş derin öğrenme mimarileri.

## 📋 Veri Seti Yapısı
Proje, **Penn-Fudan Pedestrian Database** yapısını temel alır:
* `PNGImages/`: Orijinal yaya resimleri.
* `PedMasks/`: Yayaların piksel düzeyindeki maske görüntüleri.

## 🚀 Çalıştırma

### Eğitim
İlgili modeli eğitmek için dosyayı çalıştırın:
```bash
python r-cnn.py   # Mask R-CNN eğitimi için
python u-net.py   # U-Net eğitimi için