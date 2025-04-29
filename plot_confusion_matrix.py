import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/voice.csv")

# Balance dataset
from sklearn.utils import resample
df_majority = df[df.label == "male"]
df_minority = df[df.label == "female"]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced.drop(["label", "filename"], axis=1)
y = df_balanced["label"]

# Load the scaler
scaler = joblib.load("models/scaler.pkl")
X_scaled = scaler.transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Load model
model = joblib.load("models/gender_classifier.pkl")

# scale model
scaler = joblib.load('models/scaler.pkl')

# Predict
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
