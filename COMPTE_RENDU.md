# Guide du travail réalisé – MLOps TP

## 1. Objectif du projet

Ce projet a permis de mettre en place une chaîne MLOps complète autour d’un modèle de prédiction du churn :

- génération des données,
- entraînement d’un modèle scikit-learn,
- suivi d’expériences avec MLflow,
- enregistrement du modèle dans le registre MLflow,
- préparation d’une API FastAPI pour la prédiction,
- conteneurisation avec Docker,
- validation locale du service,
- préparation du push Docker vers GitHub Container Registry (GHCR).

---

## 2. Structure du projet

Les fichiers principaux sont :

- `app.py` : API FastAPI de prédiction
- `docker-compose.yml` : orchestration MLflow + API
- `Dockerfile` : image Docker de l’API
- `exporter_model.py` : export du meilleur modèle vers le dossier `model/`
- `enregistrer_modele.py` : enregistrement du modèle dans MLflow
- `train_mlflow.py` : entraînement et suivi MLflow
- `requirements.txt` : dépendances Python
- `model/` : artefacts exportés du modèle

---

## 3. Étapes réalisées

### 3.1 Génération des données

Le script `generate_data.py` a permis de produire un dataset synthétique pour la prédiction du churn.

### 3.2 Entraînement du modèle

Le modèle a été entraîné avec scikit-learn sur les données générées.

### 3.3 Suivi MLflow

Le script `train_mlflow.py` configure le tracking MLflow et loggue les métriques ainsi que le modèle.

### 3.4 Enregistrement dans le registre MLflow

Le modèle a été enregistré dans le registre MLflow via les commandes d’MLflow, avec alias et versioning.

### 3.5 Export du modèle

Le script `exporter_model.py` a permis d’exporter le meilleur modèle dans le dossier `model/` pour l’utiliser par l’API.

### 3.6 API FastAPI

Le fichier `app.py` expose deux routes :

- `/health` : vérifie la santé de l’API
- `/predict` : reçoit un client et retourne une prédiction de churn

### 3.7 Validation locale

L’API a été lancée localement avec Uvicorn et les endpoints ont été testés avec succès.

### 3.8 Dockerisation

Le projet a été conteneurisé via :

- `Dockerfile`
- `docker-compose.yml`

La configuration couvre notamment :

- un service MLflow,
- un service API,
- le mappage des ports,
- le volume partagé pour MLflow.

### 3.9 Publication sur GHCR

L’image Docker a été préparée pour être poussée vers GitHub Container Registry (GHCR), avec les étapes suivantes :

1. génération d’un token GitHub avec permissions `repo` et `write:packages`
2. connexion Docker à GHCR
3. tag de l’image
4. push vers ghcr.io

---

## 4. Commandes utiles

### 4.1 Exécuter le projet en local

```powershell
cd C:\Users\DELL1\OneDrive\Desktop\MLOps\TP1
.\.venv\Scripts\python.exe exporter_model.py
$env:MODEL_URI = "model"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4.2 Tester l’API

```powershell
python test_api.py
```

### 4.3 Vérifier le compose Docker

```powershell
docker compose -f docker-compose.yml config
```

### 4.4 Lancer les services Docker

```powershell
docker compose -f docker-compose.yml up --build -d
```

### 4.5 Vérifier les logs

```powershell
docker compose -f docker-compose.yml logs -f
```

### 4.6 Push GHCR

```powershell
$COMPTE = "mohamedraed22"
$env:GHCR_TOKEN = "ghp_TON_JETON"

$env:GHCR_TOKEN | docker login ghcr.io -u $COMPTE --password-stdin
docker tag churn-api:1.0 ghcr.io/$COMPTE/churn-api:1.0
docker push ghcr.io/$COMPTE/churn-api:1.0
```

---

## 5. Problèmes rencontrés et solutions

### Problème 1 : MLflow refusait le modèle

Cause : le modèle scikit-learn importé n’était pas considéré comme fiable par MLflow.

Solution : utilisation d’un export compatible et validation du bon chemin d’export.

### Problème 2 : mauvais interpréteur Python

Cause : exécution du script avec le mauvais Python système au lieu du venv du projet.

Solution : utilisation explicite de :

```powershell
.\.venv\Scripts\python.exe
```

### Problème 3 : dossier `model` déjà existant

Cause : MLflow refusait d’écraser un dossier non vide.

Solution : suppression préalable du dossier avant ré-export.

### Problème 4 : Docker non visible dans le terminal

Cause : PATH Windows incomplet / Docker Desktop non visible dans la session PowerShell.

Solution : appel explicite du binaire Docker ou ouverture d’un terminal PowerShell frais avec Docker déjà démarré.

---

## 6. Résultat final

Le projet est prêt pour :

- l’entraînement et le suivi MLflow,
- la prédiction via API FastAPI,
- la conteneurisation avec Docker,
- la publication d’image sur GHCR,
- et la gestion Git/GitHub avec branche de développement.

---

## 7. Branches utilisées

- `TP5` : branche de travail du TP
- `develop` : branche de fusion de la version stable

---

## 8. Conclusion

Ce TP a permis de couvrir la chaîne MLOps complète, du modèle à l’API et au déploiement conteneurisé, en gardant un bon niveau de reproductibilité, de validation locale et de publication GitHub.
