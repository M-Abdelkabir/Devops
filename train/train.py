import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Configuration
GENES = ['M63391', 'T62947', 'D14812', 'T51250', 'H66976', 'X55362']
MODEL_DIR = '/models'
DATA_PATH = './colon cancer dataset.csv'

def train():
    print("Chargement des données...")
    df = pd.read_csv(DATA_PATH)
    
    X = df[GENES]
    y = df['Class']
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Entraînement du modèle sur {len(GENES)} gènes...")
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # Sauvegarde des artefacts
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(model, os.path.join(MODEL_DIR, 'model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    
    print("Modèle et artefacts sauvegardés dans /models")

if __name__ == "__main__":
    train()
