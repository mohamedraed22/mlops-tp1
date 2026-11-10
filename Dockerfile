# 1. Image de base : Python 3.11
FROM python:3.12-slim
# 2. Dossier de travail
WORKDIR /app

# 3. Installer les dépendances
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le code et le modèle
COPY app.py ./
COPY model/ ./model/

# 5. Variables d'exécution
ENV MODEL_URI=model
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

# 6. Port exposé
EXPOSE 8000

# 7. Commande de démarrage
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]