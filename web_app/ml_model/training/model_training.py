import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import joblib

# 1. Cargar datos
df = pd.read_csv("data/synthetic/synthetic_dropout_data.csv")
X = df.drop("dropout_thought", axis=1)
y = np.where(df["dropout_thought"] == "Yes", 1, 0)  # 1=Abandono, 0=No abandono

# 2. Definir preprocesador
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ]
)

# 3. Definir modelos a entrenar
models = {
    "Logistic Regression": {
        "model": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=1722)
    },
    "Random Forest": {
        "model": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=1722)
    },
    "XGBoost": {
        "model": XGBClassifier(
            scale_pos_weight=np.sum(y == 0) / np.sum(y == 1),  # Balanceo de clases
            eval_metric="logloss",
            random_state=1722
        )
    }
}

# 4. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1722
)

# 5. Entrenar y evaluar cada modelo
results = []
used_preprocessor = False  # Bandera para controlar el guardado

for model_name, model_config in models.items():
    print(f"\n🚀 Entrenando {model_name}...")
    
    # Crear pipeline (usa el preprocesador original)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),  # Usa el ColumnTransformer definido
        ("classifier", model_config["model"])
    ])
    
    # Entrenar
    pipeline.fit(X_train, y_train)

    # Guardar el preprocesador SOLO en la primera iteración
    if not used_preprocessor:
        joblib.dump(pipeline.named_steps["preprocessor"], "ml_model/saved_models/preprocessor.pkl")
        print("✅ Preprocesador guardado!")
        used_preprocessor = True
    
    # Predecir
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = {
        "Modelo": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 2),
        "Precision": round(precision_score(y_test, y_pred), 2),
        "Recall": round(recall_score(y_test, y_pred), 2),
        "F1": round(f1_score(y_test, y_pred), 2),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 2)
    }
    results.append(metrics)
    
    # Guardar modelo
    joblib.dump(pipeline, f"ml_model/saved_models/{model_name.replace(' ', '_').lower()}.pkl")
    print(f"{model_name} guardado!")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.savefig(f"ml_model/saved_models/confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.close()

# 6. Mostrar resultados comparativos
results_df = pd.DataFrame(results)
print("\n📋 Comparación de Modelos:")
print(results_df.to_markdown(index=False))