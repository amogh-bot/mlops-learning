from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib,os

X,y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42)
model.fit(X, y)

joblib.dump(model, "models/model.pkl")
print("Model saved to model/model.pkl")