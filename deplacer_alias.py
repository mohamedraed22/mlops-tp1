
import mlflow
from mlflow import MlflowClient

# ===== 1. Se connecter à la base MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# ===== 2. Retrouver le run essai-2 =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)

if runs.empty:
    raise ValueError(
        "Le run essai-2 est introuvable dans MLflow."
    )

run_id = runs.iloc[0]["run_id"]

# ===== 3. Enregistrer une nouvelle version =====
v = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print("Nouvelle version créée :", v.version)

# ===== 4. Retrouver l'alias champion actuel =====
actuel = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

ancienne_version = actuel.version

print("Champion actuel : version", ancienne_version)

# ===== 5. Déplacer temporairement l'alias =====
client.set_registered_model_alias(
    "churn-model",
    "champion",
    v.version
)

champion_apres_deplacement = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

print(
    "Après déplacement : champion -> version",
    champion_apres_deplacement.version
)

# ===== 6. Restaurer l'alias =====
client.set_registered_model_alias(
    "churn-model",
    "champion",
    ancienne_version
)

champion_final = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

print(
    "Après retour : champion -> version",
    champion_final.version
)
