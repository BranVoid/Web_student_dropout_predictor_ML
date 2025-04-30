from flask import Blueprint, request, jsonify
import joblib
import pandas as pd

bp = Blueprint('api', __name__)

# ✅ Cargar el pipeline completo (preprocesador + modelo)
model = joblib.load('../ml_model/saved_models/xgboost.pkl')

@bp.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Solo se aceptan datos en formato JSON"}), 400
    
    try:
        data = request.get_json()
        df = pd.DataFrame([data])  # No aplicar dummies ni otro preprocesamiento

        # ✅ El pipeline hace todo (preprocesar + predecir)
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "risk": round(probability, 4),
            "message": "Success"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
