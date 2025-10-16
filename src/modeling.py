"""
modeling.py
===========
Funciones para entrenamiento y evaluación de modelos de clasificación.

Incluye:
- Preparación de datos (split, scaling, encoding)
- Entrenamiento de 4 modelos: Logistic Regression, KNN, Decision Tree, Random Forest
- Evaluación con múltiples métricas
- Visualizaciones de resultados
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import pickle

# ======================================================================================================================
# 1. Preparación de datos (3 funciones)
# ======================================================================================================================
def prepare_data(
        df: pd.DataFrame,
        target_col: str = 'impact_class',
        test_size: float = 0.25,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Prepara los datos para modelado: separa features, target y hace split.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con todas las columnas
    target_col : str
        Nombre de la columna objetivo
    test_size : float
        Proporción del test set (default: 0.25)
    random_state : int
        Semilla aleatoria para reproducibilidad

    Returns:
    --------
    X_train, X_test, y_train, y_test : DataFrames y Series
    numeric_features : list
        Lista de features numéricas
    categorical_features : list
        Lista de features categóricas
    """
    # Identificar columnas a excluir
    id_cols = ['naics_code', 'naics_title']

    # Separar features y target
    X = df.drop(columns=id_cols + [target_col])
    y = df[target_col]

    # Identificar tipos de features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("=" * 70)
    print("📊 PREPARACIÓN DE DATOS COMPLETADA")
    print("=" * 70)
    print(f"\n🔢 Dimensiones:")
    print(f"   Train set: {X_train.shape[0]} filas × {X_train.shape[1]} features")
    print(f"   Test set:  {X_test.shape[0]} filas × {X_test.shape[1]} features")
    print(f"\n📋 Features:")
    print(f"   Numéricas: {len(numeric_features)}")
    print(f"   Categóricas: {len(categorical_features)}")
    print(f"\n🎯 Distribución del target:")
    print(f"   Train:\n{y_train.value_counts().sort_index()}")
    print(f"   Test:\n{y_test.value_counts().sort_index()}")

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


def encode_categorical_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        categorical_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Codifica features categóricas usando LabelEncoder.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Datasets de entrenamiento y prueba
    categorical_features : list
        Lista de columnas categóricas a codificar

    Returns:
    --------
    X_train_encoded, X_test_encoded : pd.DataFrame
        Datasets con features codificadas
    encoders : dict
        Diccionario con los encoders por feature
    """
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    encoders = {}

    for col in categorical_features:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col])
        X_test_encoded[col] = le.transform(X_test[col])
        encoders[col] = le

    print(f"\n✅ Features categóricas codificadas: {categorical_features}")

    return X_train_encoded, X_test_encoded, encoders


def scale_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        numeric_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Escala features numéricas usando StandardScaler.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Datasets de entrenamiento y prueba
    numeric_features : list
        Lista de features numéricas a escalar

    Returns:
    --------
    X_train_scaled, X_test_scaled : pd.DataFrame
        Datasets con features escaladas
    scaler : StandardScaler
        Scaler ajustado
    """
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    print(f"\n✅ Features numéricas escaladas: {len(numeric_features)} features")

    return X_train_scaled, X_test_scaled, scaler

# ======================================================================================================================
# 2. Entrenamiento
# ======================================================================================================================
def train_logistic_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        max_iter: int = 1000,
        random_state: int = 42
) -> LogisticRegression:
    """
    Entrena modelo de Regresión Logística.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de entrenamiento (deben estar escaladas)
    y_train : pd.Series
        Target de entrenamiento
    max_iter : int
        Máximo número de iteraciones
    random_state : int
        Semilla aleatoria

    Returns:
    --------
    model : LogisticRegression
        Modelo entrenado
    """
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        multi_class='multinomial',
        solver='lbfgs'
    )

    model.fit(X_train, y_train)

    print(f"\n✅ Logistic Regression entrenado")

    return model


def train_knn(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_neighbors: int = 5
) -> KNeighborsClassifier:
    """
    Entrena modelo K-Nearest Neighbors.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de entrenamiento (deben estar escaladas)
    y_train : pd.Series
        Target de entrenamiento
    n_neighbors : int
        Número de vecinos

    Returns:
    --------
    model : KNeighborsClassifier
        Modelo entrenado
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    model.fit(X_train, y_train)

    print(f"\n✅ KNN entrenado (k={n_neighbors})")

    return model


def train_decision_tree(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        max_depth: int = 10,
        min_samples_split: int = 20,
        random_state: int = 42
) -> DecisionTreeClassifier:
    """
    Entrena modelo de Árbol de Decisión.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de entrenamiento (sin escalar)
    y_train : pd.Series
        Target de entrenamiento
    max_depth : int
        Profundidad máxima del árbol
    min_samples_split : int
        Mínimo de muestras para dividir un nodo
    random_state : int
        Semilla aleatoria

    Returns:
    --------
    model : DecisionTreeClassifier
        Modelo entrenado
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    print(f"\n✅ Decision Tree entrenado (max_depth={max_depth})")

    return model


def train_random_forest(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_split: int = 10,
        random_state: int = 42
) -> RandomForestClassifier:
    """
    Entrena modelo de Random Forest.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Features de entrenamiento (sin escalar)
    y_train : pd.Series
        Target de entrenamiento
    n_estimators : int
        Número de árboles
    max_depth : int
        Profundidad máxima de cada árbol
    min_samples_split : int
        Mínimo de muestras para dividir un nodo
    random_state : int
        Semilla aleatoria

    Returns:
    --------
    model : RandomForestClassifier
        Modelo entrenado
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print(f"\n✅ Random Forest entrenado (n_estimators={n_estimators})")

    return model

# ======================================================================================================================
# 3. Evaluación
# ======================================================================================================================

def evaluate_model(
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
) -> Dict:
    """
    Evalúa un modelo y retorna métricas.

    Parameters:
    -----------
    model : sklearn model
        Modelo entrenado
    X_test : pd.DataFrame
        Features de prueba
    y_test : pd.Series
        Target de prueba
    model_name : str
        Nombre del modelo

    Returns:
    --------
    metrics : dict
        Diccionario con todas las métricas
    """
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # AUC-ROC (multiclase)
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc_roc = None

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'y_pred': y_pred,
        'y_test': y_test,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    # Imprimir resultados
    print("\n" + "=" * 70)
    print(f"📊 RESULTADOS - {model_name}")
    print("=" * 70)
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    if auc_roc:
        print(f"   AUC-ROC:   {auc_roc:.4f}")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


def compare_models(metrics_list: List[Dict]) -> pd.DataFrame:
    """
    Compara métricas de múltiples modelos.

    Parameters:
    -----------
    metrics_list : list
        Lista de diccionarios con métricas de cada modelo

    Returns:
    --------
    comparison_df : pd.DataFrame
        DataFrame con comparación de métricas
    """
    comparison_data = []

    for metrics in metrics_list:
        comparison_data.append({
            'Model': metrics['model_name'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'AUC-ROC': metrics['auc_roc'] if metrics['auc_roc'] else np.nan
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("🏆 COMPARACIÓN DE MODELOS")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    return comparison_df


def plot_confusion_matrices(metrics_list: List[Dict], figsize: Tuple[int, int] = (16, 4)):
    """
    Grafica matrices de confusión de todos los modelos.

    Parameters:
    -----------
    metrics_list : list
        Lista de diccionarios con métricas de cada modelo
    figsize : tuple
        Tamaño de la figura
    """
    n_models = len(metrics_list)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for idx, metrics in enumerate(metrics_list):
        cm = metrics['confusion_matrix']

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[idx],
            cbar=False,
            square=True
        )

        axes[idx].set_title(f"{metrics['model_name']}\nAccuracy: {metrics['accuracy']:.3f}")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(comparison_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)):
    """
    Grafica comparación de métricas entre modelos.

    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame con comparación de métricas
    figsize : tuple
        Tamaño de la figura
    """
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    df_plot = comparison_df.set_index('Model')[metrics_to_plot]

    ax = df_plot.plot(kind='bar', figsize=figsize, rot=45)
    ax.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Modelo', fontsize=12)
    ax.legend(title='Métrica', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Añadir líneas de referencia
    ax.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, label='Target: 0.70')
    ax.axhline(y=0.65, color='orange', linestyle='--', alpha=0.5, label='Target F1: 0.65')

    plt.tight_layout()
    plt.show()


def get_feature_importance(
        model,
        feature_names: List[str],
        top_n: int = 15
) -> pd.DataFrame:
    """
    Obtiene importancia de features para modelos basados en árboles.

    Parameters:
    -----------
    model : sklearn model
        Modelo Random Forest o Decision Tree
    feature_names : list
        Lista con nombres de features
    top_n : int
        Número de top features a retornar

    Returns:
    --------
    importance_df : pd.DataFrame
        DataFrame con importancia de features
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Este modelo no tiene feature_importances_")
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    print("\n" + "=" * 70)
    print(f"🎯 TOP {top_n} FEATURES MÁS IMPORTANTES")
    print("=" * 70)
    print(importance_df.to_string(index=False))

    return importance_df


def plot_feature_importance(
        importance_df: pd.DataFrame,
        title: str = 'Feature Importance',
        figsize: Tuple[int, int] = (10, 6)
):
    """
    Grafica importancia de features.

    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame con columnas 'feature' e 'importance'
    title : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura
    """
    plt.figure(figsize=figsize)

    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


# ======================================================================================================================
# 4. Persistencia
# ======================================================================================================================

def save_model(model, filepath: str):
    """
    Guarda modelo entrenado usando pickle.

    Parameters:
    -----------
    model : sklearn model
        Modelo a guardar
    filepath : str
        Ruta donde guardar el modelo
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n✅ Modelo guardado en: {filepath}")


def load_model(filepath: str):
    """
    Carga modelo desde archivo pickle.

    Parameters:
    -----------
    filepath : str
        Ruta del modelo guardado

    Returns:
    --------
    model : sklearn model
        Modelo cargado
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    print(f"\n✅ Modelo cargado desde: {filepath}")

    return model