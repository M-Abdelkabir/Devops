from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = '/models/model.pkl'
SCALER_PATH = '/models/scaler.pkl'
LE_PATH = '/models/label_encoder.pkl'

GENES = ['M63391', 'T62947', 'D14812', 'T51250', 'H66976', 'X55362']

model = None
scaler = None
le = None

def load_model():
    global model, scaler, le
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html', genes=GENES)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        if not load_model():
            return jsonify({'error': 'Modèle non trouvé. Veuillez lancer l\'entraînement.'}), 500
    
    try:
        data = [float(request.form[gene]) for gene in GENES]
        features = np.array(data).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0]
        
        result = le.inverse_transform(prediction)[0]
        confidence = float(np.max(probability))
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence*100:.2f}%",
            'probability': probability.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
