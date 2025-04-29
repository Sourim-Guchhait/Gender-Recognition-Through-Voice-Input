import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib
import os

# 1. Load dataset
df = pd.read_csv("data/voice.csv")

# 2. Balance dataset
df_male = df[df.label == "male"]
df_female = df[df.label == "female"]

df_female_upsampled = resample(df_female,
                               replace=True,
                               n_samples=len(df_male),
                               random_state=42)

df_balanced = pd.concat([df_male, df_female_upsampled])

# 3. Features (X) and Labels (y)
X = df_balanced.drop(["label", "filename"], axis=1)  # Drop filename and label
y = df_balanced["label"]

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Hyperparameter tuning for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'max_iter': [300, 500, 700],
    'solver': ['lbfgs']
}

grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train)

print("\n✅ Best Parameters:", grid.best_params_)

# 7. Evaluate
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/gender_classifier.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Model saved at: models/gender_classifier.pkl")
print("✅ Scaler saved at: models/scaler.pkl")
