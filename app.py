# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load dataset
df = pd.read_csv(r"C:\Users\BHB\Desktop\Go my code Data science\Python\New folder\Expresso_churn_dataset.csv")

# 2. Basic cleaning (drop NA)
df = df.dropna()

# 3. Select features (example: all numeric except 'CHURN')
X = df.drop(columns=["CHURN"])
y = df["CHURN"]

# Keep only numeric features
X = X.select_dtypes(include=["number"])

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Save model, scaler, and features
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Training complete! Files saved: model.pkl, scaler.pkl, features.pkl")
