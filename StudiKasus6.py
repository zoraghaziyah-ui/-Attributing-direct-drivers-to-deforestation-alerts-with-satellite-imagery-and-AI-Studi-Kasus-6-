import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=== 1. MEMUAT & MENYIAPKAN DATASET MONITORING HUTAN ===")
np.random.seed(42)
n_samples = 2000

data = {
    'Sentinel1_Backscatter': np.random.uniform(-25.0, -5.0, size=n_samples),
    'Sentinel2_NDVI': np.random.uniform(-0.2, 0.8, size=n_samples),        
    'Alert_Patch_Size_Ha': np.random.uniform(0.5, 50.0, size=n_samples),   
    'Distance_to_Road_KM': np.random.uniform(0.1, 30.0, size=n_samples),   
    'Slope_Degree': np.random.uniform(0, 45, size=n_samples)                
}

df = pd.DataFrame(data)

conditions = [
    (df['Alert_Patch_Size_Ha'] < 5) & (df['Distance_to_Road_KM'] < 5),
    (df['Alert_Patch_Size_Ha'] >= 5) & (df['Sentinel2_NDVI'] < 0.3),
    (df['Slope_Degree'] > 25) & (df['Sentinel1_Backscatter'] > -12),
    (df['Distance_to_Road_KM'] < 2) & (df['Alert_Patch_Size_Ha'] < 10),
    (df['Sentinel2_NDVI'] < 0.1)
]
choices = [0, 1, 2, 3, 4]

df['Driver_Class'] = np.select(conditions, choices, default=0)

print("5 Baris Pertama Data Fitur Satelit & Driver:")
print(df.head(), "\n")

print("=== 2. DATA PREPARATION & SPLITTING ===")
X = df.drop(columns=['Driver_Class'])
y = df['Driver_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Jumlah Data Training: {X_train.shape[0]} baris")
print(f"Jumlah Data Testing : {X_test.shape[0]} baris\n")

print("=== 3. PELATIHAN MODEL KLASIFIKASI DRIVER DEFORESTASI ===")
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

print("=== 4. EVALUASI PERFORMA MODEL ===")
y_pred = model.predict(X_test)

print("Akurasi Model:", accuracy_score(y_test, y_pred))
print("\nClassification Report (Macro F1 & Per-Class Metrics):")
print(classification_report(y_test, y_pred, target_names=['Small Agriculture', 'Large Agriculture', 'Mining', 'Selective Logging', 'Wildfire']))

df_result = X_test.copy()
df_result['Actual_Driver'] = y_test
df_result['Predicted_Driver'] = y_pred
df_result.to_csv("deforestation_driver_results.csv", index=False)
print("\nHasil klasifikasi driver berhasil disimpan ke 'deforestation_driver_results.csv'")