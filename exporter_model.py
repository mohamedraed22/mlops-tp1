
# ===== 1. Importer les bibliothèques =====
import shutil

import mlflow
import mlflow.sklearn

# ===== 2. Se connecter au Registry =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Charger le modèle champion =====
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

# ===== 4. Supprimer l'ancien export pour pouvoir réécrire le dossier =====
if shutil.os.path.exists("model"):
    shutil.rmtree("model")

# ===== 5. Exporter le modèle dans le dossier model =====
mlflow.sklearn.save_model(
    model,
    "model",
    serialization_format="cloudpickle"
)

print("Champion exporté dans le dossier model/")
