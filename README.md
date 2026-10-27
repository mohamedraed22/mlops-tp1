# TP1 MLOps - Prédiction du churn client

Ce projet construit un premier modèle de prédiction du départ de clients (churn) avant d'introduire les outils MLOps. Il utilise un jeu de données synthétique, explore et prépare les données, entraîne des modèles Random Forest et compare trois essais.

## Jeu de données

`churn_data.csv` contient 500 clients fictifs et 6 colonnes :

- `age`, `salaire`, `abonnement`, `anciennete` et `consommation` sont les variables explicatives.
- `churn` est la cible binaire : elle vaut 1 si `consommation < 30` et `anciennete < 2`, sinon 0.

Les données sont générées par `generate_data.py` avec une graine NumPy fixe (42). Le taux de churn est de 3,8 % (19 clients sur 500).

## Étapes du notebook

`churn_model.ipynb` documente le travail :

1. Exploration : dimensions, valeurs manquantes, répartition de la cible.
2. Préparation : encodage de `abonnement` (Basique=0, Standard=1, Premium=2), puis découpage 80/20 avec `random_state=42`.
3. Entraînement et évaluation d'un `RandomForestClassifier` (accuracy et rapport de classification).
4. Sauvegarde du modèle avec joblib dans `model.pkl`.
5. Comparaison de trois essais sur le même jeu de test.

## Résultats

| Essai | Configuration | Accuracy |
|---|---|---:|
| 1 | 100 arbres, profondeur 5, 5 features | 0.99 |
| 2 | 50 arbres, profondeur 3, 5 features | 0.99 |
| 3 | 100 arbres, profondeur 5, 4 features (sans `salaire`) | 1.00 |

Le jeu de test contient seulement 5 clients qui partent : un écart de 0.01 d'accuracy correspond à un seul client. Chaque essai écrase `model.pkl` ; à la fin, ce fichier contient le modèle de l'Essai 3. Il est exclu de Git.

## Limites

- Les classes sont très déséquilibrées (3,8 % de churn) : prédire toujours « 0 » donnerait déjà 0.95 d'accuracy sur le test.
- Un seul découpage train/test et une seule métrique : pas de validation croisée.
- Les modèles des essais précédents sont écrasés, ce qui motive l'utilisation d'un outil de suivi d'expériences.

## Technologies

Python, NumPy, pandas, scikit-learn, joblib, Jupyter Notebook, ipykernel, Git et GitHub.

## Installation et exécution

Depuis PowerShell, dans le dossier du projet :

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn joblib jupyter ipykernel
python generate_data.py
```

Ouvrir ensuite `churn_model.ipynb` avec le kernel `.venv` et exécuter les cellules de haut en bas.