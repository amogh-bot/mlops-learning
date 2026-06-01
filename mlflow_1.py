# quick_test.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-classification")

# Run 1: baseline
with mlflow.start_run(run_name="rf-baseline"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))
    mlflow.sklearn.log_model(model, "model")
    print(f"Run 1 — Accuracy: {accuracy_score(y_test, preds):.4f}")

# Run 2: deeper trees
with mlflow.start_run(run_name="rf-deeper-trees"):
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))
    mlflow.sklearn.log_model(model, "model")
    print(f"Run 2 — Accuracy: {accuracy_score(y_test, preds):.4f}")

# Run 3: more trees, shallow
with mlflow.start_run(run_name="rf-wide-shallow"):
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("max_depth", 3)
    model = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))
    mlflow.sklearn.log_model(model, "model")
    print(f"Run 3 — Accuracy: {accuracy_score(y_test, preds):.4f}")