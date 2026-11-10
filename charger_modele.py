
import mlflow
import mlflow.sklearn
import pandas as pd

# 1. Retrouver le meilleur run selon le F1-score
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

if runs.empty:
    raise ValueError(
        "Aucun run trouvé. Exécute d'abord train.py."
    )

run_id = runs.iloc[0]["run_id"]

# 2. Recharger le modèle enregistré dans MLflow
model = mlflow.sklearn.load_model(
    f"runs:/{run_id}/model"
)

print("Modèle rechargé :", type(model).__name__)
print("Nombre de variables attendues :", model.n_features_in_)

# 3. Créer les données d'un nouveau client
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])

# 4. Garder les variables attendues par le modèle,
# dans le bon ordre
client = client.reindex(
    columns=model.feature_names_in_,
    fill_value=False
)

# 5. Effectuer la prédiction
prediction = model.predict(client)[0]

if prediction == 1:
    print("Le client va-t-il partir ? -> OUI")
else:
    print("Le client va-t-il partir ? -> NON")
