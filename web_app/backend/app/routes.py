from flask import Blueprint, request, jsonify
import pandas as pd
from .models.load_models import load_model_and_preprocessor  # 👈 Importar la función

bp = Blueprint('api', __name__)

@bp.route('/predict', methods=['POST'])
def predict():
    # 1. Obtener parámetros
    data = request.get_json()
    model_name = data.get("model", "xgboost")  # Por defecto usa XGBoost
    
    try:
        # 2. Cargar modelo y preprocesador
        model, preprocessor = load_model_and_preprocessor(model_name)
        
        # 3. Preprocesar datos
        df = pd.DataFrame([data])
        processed_data = preprocessor.transform(df)
        
        # 4. Predecir
        probability = model.predict_proba(processed_data)[0][1]
        
        return jsonify({
            "model": model_name,
            "risk": round(probability, 4),
            "message": "Success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "model": model_name
        }), 500