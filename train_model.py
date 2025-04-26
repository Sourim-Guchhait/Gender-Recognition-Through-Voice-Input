from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import pickle
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("data/voice.csv")



# 🔄 Balance the dataset
df_majority = df[df.label == "male"]
df_minority = df[df.label == "female"]

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])




# Separate features and labels
X = df.drop(columns=[col for col in ['label', 'filename'] if col in df.columns])

y = df['label']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest with better params
model = RandomForestClassifier(
    n_estimators=200,    # Increase number of trees
    max_depth=20,        # Limit depth to avoid overfitting
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy on Test Set: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🧩 Confusion Matrix:")
print(cm)

# Save model
import os
os.makedirs("models", exist_ok=True)
# Save model
with open("models/gender_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved to models/gender_classifier.pkl")
