import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

# 1. Veri Yükleme ve Temizleme
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# 2. Model Kurulumu
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

# 3. Tahminler
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 4. GRAFİKLER VE SONUÇLAR
# --- Grafik 1: Confusion Matrix ---
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Hata Matrisi (Confusion Matrix)')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.show()

# --- Grafik 2: Özellik Önemi ---
importance = pd.DataFrame({'Özellik': X.columns, 'Katsayı': model.coef_[0]}).sort_values(by='Katsayı', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Katsayı', y='Özellik', data=importance.head(10), palette='viridis')
plt.title('Churn Etkileyen En Önemli 10 Faktör')
plt.show()

# --- Sonuç Raporu ---
print("\n=== PERFORMANS METRİKLERİ ===")
print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Skoru: {roc_auc_score(y_test, y_prob):.4f}")