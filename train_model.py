# train_model.py

# Install necessary packages (run in terminal if not installed)
# pip install pandas scikit-learn streamlit ydata-profiling joblib

import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ======================
# Load Data
# ======================
df = pd.read_csv(r"C:\Users\BHB\Desktop\Go my code Data science\Python\New folder\Expresso_churn_dataset.csv")


# Display basic info
print(df.info())
print(df.head())

# ======================
# Profiling Report
# ======================
profile = ProfileReport(df, title="Expresso Churn Dataset Report", explorative=True)
profile.to_file("profiling_report.html")

# ======================
# Handle Missing Values
# ======================
df = df.drop_duplicates()
df = df.dropna()   # simple drop, you can impute if needed

# ======================
# Handle Outliers (basic z-score method for numeric features)
# ======================
import numpy as np
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df = df[(np.abs((df[col] - df[col].mean()) / df[col].std()) < 3)]

# ======================
# Encode Categorical Features
# ======================
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ======================
# Split Data
# ======================
X = df.drop("CHURN", axis=1)  # change target column name if needed
y = df["CHURN"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# Scale Features
# ======================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# Train Model
# ======================
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# ======================
# Evaluate
# ======================
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ======================
# Save Model & Scaler
# ======================
joblib.dump(clf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "features.pkl")
print("Model, scaler and features saved!")
