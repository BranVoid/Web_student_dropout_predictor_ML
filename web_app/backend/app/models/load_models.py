import joblib

def load_model_and_preprocessor(model_name):
    model = joblib.load(f"ml_model/saved_models/{model_name}.pkl")
    preprocessor = joblib.load(f"ml_model/saved_models/preprocessor_{model_name}.pkl")
    return model, preprocessor