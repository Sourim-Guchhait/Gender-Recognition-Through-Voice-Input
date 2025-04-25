from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("data/voice.csv")

# Separate features and labels
X = df.drop(['label', 'filename'], axis=1)
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
with open("models/gender_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved to models/gender_classifier.pkl")
