"""sentiment_sınıflandırması_example

"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("trendyol.csv")
df = df[['review_body', 'review_rating']].dropna()

df = df[df['review_rating'] != 3]
df['label'] = df['review_rating'].apply(lambda x: 1 if x > 3 else 0)

X = df['review_body'].values
y = df['label'].values

print(f"Toplam temizlenen veri sayısı: {len(df)}")

X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=16, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.1)

loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"\n--- Modelin Genel Başarısı ---")
print(f"Test Accuracy (Doğruluk Oranı): %{accuracy*100:.2f}")

def yeni_yorum_test_et(metin):
    seq = tokenizer.texts_to_sequences([metin])
    pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    tahmin = model.predict(pad)[0][0]
    durum = "Pozitif" if tahmin > 0.5 else "Negatif"
    print(f"Yorum: {metin} -> Tahmin: {durum} (Olasılık: {tahmin:.2f})")

yeni_yorum_test_et("Ürün beklediğimden çok daha kaliteli çıktı, teşekkürler.")
yeni_yorum_test_et("Paranızı çöpe atmayın, paketleme berbattı.")