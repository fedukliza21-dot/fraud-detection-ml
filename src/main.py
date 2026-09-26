import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score

#пути

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "predictions.xlsx")

#загрузка данных

df = pd.read_csv(
    DATA_PATH,
    sep=",",
    engine="python",
    header=0,
    skipinitialspace=True
)

# защита от склеенных данных
if df.shape[1] == 1:
    df = df.iloc[:, 0].str.split(",", expand=True)

    df.columns = [
        "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
        "V10","V11","V12","V13","V14","V15","V16","V17","V18",
        "V19","V20","V21","V22","V23","V24","V25","V26","V27",
        "V28","Amount","Class"
    ]

# очистка класса
df["Class"] = df["Class"].astype(str).str.replace('"', '', regex=False).astype(int)

#признаки

X = df.drop(columns=["Class"])
y_true = df["Class"].values

#масштабирование

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#доля аномалий

contamination = y_true.mean()

#Isolation Forest

iso = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42
)

iso_pred = iso.fit_predict(X_scaled)
iso_pred = (iso_pred == -1).astype(int)

#LOF

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination
)

lof_pred = lof.fit_predict(X_scaled)
lof_pred = (lof_pred == -1).astype(int)

#DBSCAN

db = DBSCAN(
    eps=3,
    min_samples=10
)

db_labels = db.fit_predict(X_scaled)
db_pred = (db_labels == -1).astype(int)

#метрики

def report(name, y_true, y_pred):
    print("\n======================")
    print(name)
    print("======================")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

#вывод

report("Isolation Forest", y_true, iso_pred)
report("Local Outlier Factor", y_true, lof_pred)
report("DBSCAN", y_true, db_pred)

#сохранение

df["IF"] = iso_pred
df["LOF"] = lof_pred
df["DBSCAN"] = db_pred

df.to_excel(OUTPUT_PATH, index=False)

print("\nSaved to file:", OUTPUT_PATH)