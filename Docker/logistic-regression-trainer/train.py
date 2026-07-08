import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import os

def main():
    print("--- House Prices Random Forest Modeli Başlatılıyor ---")
    
    train_path = 'data/train.csv'
    if not os.path.exists(train_path):
        print(f"Hata: {train_path} bulunamadı! Veriyi 'data' klasörüne koyduğunuzdan emin olun.")
        return

    print("Veri yükleniyor...")
    df = pd.read_csv(train_path)
    
    X = df.drop(columns=['Id', 'SalePrice'])
    y = df['SalePrice']
    
    print("Veri ön işleme adımları uygulanıyor...")
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_numeric)
    
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    
    print("Random Forest Regressor eğitiliyor (n_estimators=100)...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Model değerlendiriliyor...")
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n=== MODEL SONUÇLARI ===")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R2) Skoru: {r2:.4f}")
    print("=========================\n")
    print("Eğitim başarıyla tamamlandı!")

if __name__ == '__main__':
    main()