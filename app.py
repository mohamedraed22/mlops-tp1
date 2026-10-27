
# ===== 1. Importer les bibliothèques =====
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")


# ===== 3. Charger le modèle champion au démarrage =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model("models:/churn-model@champion")


# ===== 4. Décrire les données attendues =====
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 5. Vérifier que l'API fonctionne =====
@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 6. Prédire le churn d'un client =====
@app.post("/predict")
def predict(client: Client):

    # Transformer les données reçues en DataFrame
    ligne = pd.DataFrame([client.model_dump()])

    # Encoder la variable catégorielle
    ligne = pd.get_dummies(ligne)

    # Aligner les colonnes sur celles de l'entraînement
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Effectuer la prédiction
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": "Le client va partir" if resultat == 1 else "Le client reste"
    }
