import sklearn
import pandas as pd
import joblib
with open("models/lead_model_lr.pkl", "rb") as f:
    model = joblib.load(f)

X = pd.read_csv("data/processed/X_test.csv")
y = pd.read_csv("data/processed/y_test.csv")
print(model.predict(X.head(5)), y.head(5))
