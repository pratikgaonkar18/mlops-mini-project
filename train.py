import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import mlflow
import mlflow.sklearn
import json
from pathlib import Path


# Set experiment name (groups related runs)
mlflow.set_experiment("iris-classification")

# 1. Load data
data = pd.read_csv("data/iris.csv")

X = data.drop("target", axis=1)
y = data["target"]

# 2. Split into train and test (fixed random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameters (we will track these)
n_estimators = 200
random_state = 42

with mlflow.start_run():
    # 3. Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # 4. Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(X_train, y_train)

    # 5. Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy}")

    import json
    from pathlib import Path

    metrics = {
        "accuracy": float(accuracy)
    } 

    Path("metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Metrics saved to metrics.json")
	
    THRESHOLD = 0.90

    if accuracy >= THRESHOLD:
         print(f"Accuracy {accuracy} >= {THRESHOLD}. Model passes quality gate. Registering model...")
    # Your existing MLflow registration/logging code should stay here
    # (e.g., mlflow.log_metric, mlflow.sklearn.log_model, or your current registration logic)
    else:
         raise RuntimeError(f"Accuracy {accuracy} < {THRESHOLD}. Model failed quality gate. Not registering.")

    # 6. Log metric
    mlflow.log_metric("accuracy", accuracy)

    # 7. Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)

    print("Model saved to models/model.pkl")

    # 8. Log model as artifact
    mlflow.log_artifact(model_path)

    # (Optional but nice) log model in MLflow format
    mlflow.sklearn.log_model(model, "model")

