import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Veri setini yükle ve DataFrame yap
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# --- GÖRSELLEŞTİRME 1: Veri Seti İlişkileri ---
sns.pairplot(df, hue='species', palette='viridis')
plt.title("İris Çiçeği Özelliklerinin Dağılımı")
plt.show()

# 2. Modeli Hazırla
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Random Forest Modelini Eğit
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Tahmin ve Başarı
y_pred = rf_model.predict(X_test)
print(f"Model Doğruluğu: %{accuracy_score(y_test, y_pred) * 100:.2f}")

# --- GÖRSELLEŞTİRME 2: Özellik Önemi (Feature Importance) ---
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(4).plot(kind='barh', color='skyblue')
plt.title("Hangi Özellik Tahmin İçin Daha Önemli?")
plt.show()

# --- GÖRSELLEŞTİRME 3: Karmaşıklık Matrisi (Confusion Matrix) ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Karmaşıklık Matrisi')
plt.show()