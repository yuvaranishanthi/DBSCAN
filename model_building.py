import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Config
dataset_path = "bank-full.csv"
eps_value = 0.5
min_samples_value = 5
important_features = ["age", "job", "balance", "housing", "loan", "duration"]

# Load dataset
df = pd.read_csv(dataset_path, sep=';')

# Encode categorical features
encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Scale entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[important_features])

# Run DBSCAN
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
clusters = dbscan.fit_predict(X_scaled)

# Train classifier to mimic DBSCAN clustering
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, clusters)

# Save everything
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/dbscan_classifier.pkl")
joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(important_features, "model/important_features.pkl")

# Save dropdown options for HTML form
dropdown_options = {}
for feature in important_features:
    if feature in encoders:
        dropdown_options[feature] = sorted(list(encoders[feature].classes_))
joblib.dump(dropdown_options, "model/dropdown_options.pkl")

print("✅ Model building complete: DBSCAN + Classifier saved.")

