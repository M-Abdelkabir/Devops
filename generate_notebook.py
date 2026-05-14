import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Analyse Complète : Sélection Globale et Évaluation Entraînement/Test\n",
    "Ce notebook combine deux étapes cruciales :\n",
    "1. **Sélection Globale** : Identifier les gènes les plus impactants sur l'ensemble des 62 patients.\n",
    "2. **Évaluation Rigoureuse** : Mesurer la performance réelle de ces gènes en séparant les données en Entraînement et Test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "sns.set_style('darkgrid')\n",
    "\n",
    "# Chargement\n",
    "df = pd.read_csv('./colon cancer dataset.csv')\n",
    "X_raw = df.drop('Class', axis=1)\n",
    "y_raw = LabelEncoder().fit_transform(df['Class'])\n",
    "X_scaled_full = StandardScaler().fit_transform(X_raw)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Sélection des Gènes sur la Totalité du Dataset (62 patients)\n",
    "Nous utilisons la Forward Selection pour identifier les gènes qui expliquent le mieux l'ensemble de la population actuelle."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "available_features = list(range(X_scaled_full.shape[1]))\n",
    "feature_names = X_raw.columns.tolist()\n",
    "selected_indices = []\n",
    "selected_names = []\n",
    "global_accuracies = []\n",
    "\n",
    "print(\"Calcul de la sélection globale...\")\n",
    "for step in range(10): # On regarde les 10 premiers\n",
    "    best_acc = 0.0\n",
    "    best_idx = None\n",
    "    \n",
    "    for idx in available_features:\n",
    "        temp_indices = selected_indices + [idx]\n",
    "        model = LogisticRegression(random_state=42)\n",
    "        model.fit(X_scaled_full[:, temp_indices], y_raw)\n",
    "        acc = accuracy_score(y_raw, model.predict(X_scaled_full[:, temp_indices]))\n",
    "        \n",
    "        if acc > best_acc:\n",
    "            best_acc = acc\n",
    "            best_idx = idx\n",
    "            \n",
    "    selected_indices.append(best_idx)\n",
    "    selected_names.append(feature_names[best_idx])\n",
    "    global_accuracies.append(best_acc)\n",
    "    available_features.remove(best_idx)\n",
    "    print(f\"Étape {step+1}: {feature_names[best_idx]} -> {best_acc*100:.2f}%\")\n",
    "\n",
    "# Affichage de la courbe globale\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(range(1, 11), [a*100 for a in global_accuracies], marker='o', color='purple')\n",
    "plt.xticks(range(1, 11), selected_names, rotation=45)\n",
    "plt.title('Evolution de l\\'Exactitude Globale (62 patients)')\n",
    "plt.ylabel('Accuracy (%)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Évaluation sur Entraînement vs Test (Top 6 Gènes)\n",
    "Nous prenons les 6 meilleurs gènes identifiés ci-dessus et testons leur capacité de généralisation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_genes = selected_names[:6]\n",
    "X_subset = df[final_genes]\n",
    "y = y_raw\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_s = scaler.fit_transform(X_train)\n",
    "X_test_s = scaler.transform(X_test)\n",
    "\n",
    "model_final = LogisticRegression(random_state=42)\n",
    "model_final.fit(X_train_s, y_train)\n",
    "\n",
    "print(f\"--- ÉVALUATION SUR LES 6 GÈNES CHOISIS ---\")\n",
    "print(f\"Exactitude Entraînement : {accuracy_score(y_train, model_final.predict(X_train_s))*100:.2f}%\")\n",
    "print(f\"Exactitude Test : {accuracy_score(y_test, model_final.predict(X_test_s))*100:.2f}%\")\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "sns.heatmap(confusion_matrix(y_train, model_final.predict(X_train_s)), annot=True, fmt='d', cmap='Blues', ax=axes[0])\n",
    "axes[0].set_title('Confusion (Entraînement)')\n",
    "sns.heatmap(confusion_matrix(y_test, model_final.predict(X_test_s)), annot=True, fmt='d', cmap='Greens', ax=axes[1])\n",
    "axes[1].set_title('Confusion (Test)')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('logistic_regression_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
