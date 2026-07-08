# 🧠 Doğal Dil İşleme (NLP) ve Sıralı Veri Analizi Portfolyosu

Bu depo, metin madenciliği temellerinden modern derin öğrenme mimarilerine (RNN, LSTM, Transformer) kadar uzanan geniş kapsamlı NLP çalışma projelerini içermektedir.

## 📂 Proje İçeriği

### 1. Metin İşleme ve İstatistiksel Analiz
* **TF-IDF Analizi:** `tf_i̇df_haber.py` ile doküman benzerliklerini belirleme ve metin ağırlıklandırma.
* **Parfüm Öneri Sistemi:** `perfume_recommender.py` ile içerik tabanlı (content-based) benzerlik ve öneri motoru uygulaması.

### 2. Kelime Vektörleme (Word Embeddings)
* **Word2Vec & Embeddings:** `word2vec.py` ve `word2vec_glove_fasttext.py` ile kelimeleri anlamlı vektör uzaylarına taşıyan GloVe, FastText ve Word2Vec tekniklerinin kıyaslamalı analizi.

### 3. Sıralı Veri ve Dil Modelleri (RNN & LSTM)
* **Temel RNN:** `rnn.py` ve `simple_rnn.py` ile zaman serisi verileri ve metin dizileri üzerinde tekrarlayan sinir ağı uygulamaları.
* **LSTM (Long Short-Term Memory):** `lstm.py` ve `spotify_lstm_sınıflandırıcı.py` ile müzik öznitelikleri (acousticness, danceability vb.) kullanılarak yapılan şarkı sınıflandırma analizleri.

### 4. Gelişmiş Derin Öğrenme
* **Duygu Analizi:** `sentiment_sınıflandırması_example.py` ile müşteri yorumları üzerinde pozitif/negatif ayrımı.
* **Transformer Mimarisi:** `transformer.py` ile "Attention is All You Need" mekanizmasının (Positional Encoding, Multi-Head Attention) sıfırdan implementasyonu.



---

## 🛠 Kullanılan Teknolojiler
* **Kütüphaneler:** `PyTorch`, `TensorFlow/Keras`, `Scikit-learn`, `Gensim`, `Pandas`, `Numpy`.
* **Görselleştirme:** `Matplotlib`, `Seaborn`.

## 🚀 Kurulum ve Çalıştırma
Projeleri çalıştırmak için gerekli olan kütüphaneleri şu komutla yükleyebilirsiniz:

```bash
pip install torch tensorflow gensim scikit-learn pandas matplotlib seaborn