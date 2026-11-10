
import mlflow

# 1. Récupérer les runs et les trier par F1 décroissant
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# 2. Vérifier qu'il existe des résultats
if runs.empty:
    print("Aucun run trouvé. Exécute d'abord train.py.")
else:
    # 3. Choisir les colonnes à afficher
    colonnes = [
        "tags.mlflow.runName",
        "params.n_estimators",
        "params.max_depth",
        "metrics.accuracy",
        "metrics.f1"
    ]

    print(runs[colonnes].to_string(index=False))

    # 4. Identifier le meilleur run
    meilleur = runs.iloc[0]

    print()
    print(
        "Meilleur run :",
        meilleur["tags.mlflow.runName"],
        "| id :",
        meilleur["run_id"][:8],
        "..."
    )
