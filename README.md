# Colon Cancer Precision Oncology

Ce projet implémente un pipeline complet de Data Science et de DevOps pour la classification du cancer du côlon à partir de données d'expression génétique. Il utilise une sélection rigoureuse de **6 gènes clés** pour prédire l'état d'un patient avec une haute précision.

---

## Architecture du Projet

Le projet est divisé en trois services principaux orchestrés par Docker :

1.  **Analyse (Jupyter)** : Exploration des données et validation des gènes.
2.  **Entraînement (ML)** : Pipeline automatisé qui entraîne le modèle final sur les gènes sélectionnés.
3.  **Déploiement (Flask)** : Interface web interactive pour les diagnostics en temps réel.

---

## Installation et Lancement

### 1. Prérequis
*   Docker et Docker Compose installés sur votre machine.

### 2. Lancer l'Analyse (Jupyter Notebook)
Pour explorer les données et voir les graphiques de sélection des gènes :
```bash
docker-compose up jupyter
```
Ouvrez ensuite le lien généré dans votre terminal (généralement `http://localhost:8888`). Le fichier principal est `logistic_regression_analysis.ipynb`.

### 3. Lancer le Pipeline de Production (Train + Web App)
Pour entraîner le modèle et lancer l'application de diagnostic :
```bash
docker-compose up --build train deploy
```
*   **L'entraînement** se fera automatiquement et sauvegardera le modèle dans le dossier `./models`.
*   **L'application Web** sera accessible à l'adresse : **[http://localhost:5000](http://localhost:5000)**.

---

## Gènes Sélectionnés

Le modèle utilise les 6 gènes suivants qui offrent une exactitude de **92.31%** sur le set de test :
*   **M63391**
*   **T62947**
*   **D14812**
*   **T51250**
*   **H66976**
*   **X55362**

---

## Structure des fichiers

*   `logistic_regression_analysis.ipynb` : Notebook d'analyse complète.
*   `train/` : Code source pour l'entraînement automatique du modèle.
*   `deploy/` : Code source du serveur Flask et de l'interface utilisateur.
*   `models/` : Contient les artefacts du modèle (`model.pkl`, `scaler.pkl`, `label_encoder.pkl`).
*   `docker-compose.yml` : Orchestration des conteneurs.
