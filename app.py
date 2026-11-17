
# ===== 1. Importer les bibliothèques =====
import os
from typing import Literal
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")

# ===== 3. Lire la configuration =====
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/churn-model@champion"
)

# ===== 4. Charger le modèle au démarrage =====
mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.sklearn.load_model(MODEL_URI)

print(f"Modèle chargé depuis : {MODEL_URI}")

# ===== 5. Décrire les données reçues =====
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float

# ===== 6. Route de vérification =====
@app.get("/health")
def health():
    return {"status": "ok"}

# ===== 7. Route de prédiction =====
@app.post("/predict")
def predict(client: Client):

    # Transformer les données en DataFrame
    ligne = pd.DataFrame([client.model_dump()])

    # Transformer abonnement en colonnes numériques
    ligne = pd.get_dummies(ligne)

    # Garder les colonnes attendues par le modèle
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Effectuer la prédiction
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": "Le client va partir"
        if resultat == 1
        else "Le client reste"
    }
