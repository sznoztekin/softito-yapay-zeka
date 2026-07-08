import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Sayfa genişlik ve başlık ayarları
st.set_page_config(page_title="Çoklu ML Analiz Paneli", layout="wide")
st.title("🩺 Diyabet Veri Seti - Çoklu ML Model Analiz Seti")
st.write("Scikit-learn `load_diabetes` veri seti kullanılarak farklı regresyon modellerinin karşılaştırılması.")

# 1. Veri Setini Yükleme
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Kullanıcıya ham veriyi inceleme seçeneği sunma
if st.checkbox("Ham Veriyi Göster (İlk 5 Satır)"):
    df_display = X.copy()
    df_display['TARGET'] = y
    st.dataframe(df_display.head())

# Train-test split (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Çoklu ML Modellerinin Tanımlanması
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# 3. Eğitim ve Analiz Butonu
if st.button("Tüm Modelleri Eğit ve Karşılaştır"):
    results = []
    
    col1, col2 = st.columns(2)
    
    for name, model in models.items():
        # Modeli eğit
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Metrikleri hesapla
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Sonuçları listeye ekle
        results.append({
            "Model": name,
            "R2 Score": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })

    # Sonuçları DataFrame'e dönüştür ve R2 Score'a göre sırala
    df_results = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    
    # Sol Sütun: Metrik Tablosu (Hata veren columns parametresi 'subset' olarak güncellendi)
    with col1:
        st.subheader("📊 Model Performans Metrikleri")
        st.dataframe(df_results.style.highlight_max(axis=0, subset=['R2 Score'], color='lightgreen'))
        
    # Sağ Sütun: Grafik Görselleştirmesi
    with col2:
        st.subheader("📈 R2 Score Karşılaştırması")
        fig, ax = plt.subplots()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ax.bar(df_results["Model"], df_results["R2 Score"], color=colors[:len(df_results)])
        ax.set_ylabel("R2 Score (Daha yüksek daha iyi)")
        ax.set_ylim(0, 1.0)
        for i, v in enumerate(df_results["R2 Score"]):
            ax.text(i, v + 0.02, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)