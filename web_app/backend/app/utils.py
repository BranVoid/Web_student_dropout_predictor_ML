def preprocess_input(data, preprocessor):
    # Aplicar el mismo preprocesamiento que en el entrenamiento
    processed_data = preprocessor.transform(data)
    return processed_data