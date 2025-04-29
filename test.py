import pickle

with open("models/gender_classifier.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))
