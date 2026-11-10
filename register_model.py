
import mlflow
from mlflow import MlflowClient

# ===== 1. Se connecter à la base MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 2. Retrouver le meilleur run =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

if runs.empty:
    raise ValueError(
        "Aucun run trouvé dans l'expérience churn-prediction."
    )

best = runs.iloc[0]
run_id = best["run_id"]

print(
    "Meilleur run :",
    best["tags.mlflow.runName"],
    "| id :",
    run_id[:8],
    "..."
)

# ===== 3. Enregistrer le modèle dans le Registry =====
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print(
    "Modèle enregistré : churn-model | version",
    result.version
)

# ===== 4. Définir l'alias champion =====
client = MlflowClient()

client.set_registered_model_alias(
    "churn-model",
    "champion",
    result.version
)

print(
    "Alias 'champion' -> version",
    result.version
)
