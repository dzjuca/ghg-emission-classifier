"""
=============================================================================
APLICACIÓN STREAMLIT - CLASIFICADOR DE IMPACTO AMBIENTAL GEI
=============================================================================
Sistema de predicción del nivel de impacto ambiental de sectores industriales
basado en factores de emisión de gases de efecto invernadero (GEI)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Clasificador GEI - Impacto Ambiental",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

@st.cache_resource
def load_model():
    """Carga el modelo entrenado y los encoders"""
    try:
        model_path = Path("modelos/best_model.pkl")
        encoders_path = Path("modelos/encoders.pkl")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)

        return model, encoders
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None


@st.cache_data
def load_dataset():
    """Carga el dataset completo para consultas"""
    try:
        data_path = Path("datos/dataset_for_modeling.csv")
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return None


@st.cache_data
def load_metrics():
    """Carga las métricas de comparación de modelos"""
    try:
        metrics_path = Path("modelos/model_comparison.csv")
        df = pd.read_csv(metrics_path)
        return df
    except Exception as e:
        st.warning("No se pudieron cargar las métricas del modelo")
        return None


@st.cache_data
def load_feature_importance():
    """Carga la importancia de features"""
    try:
        fi_path = Path("modelos/feature_importance.csv")
        df = pd.read_csv(fi_path)
        return df
    except Exception as e:
        st.warning("No se pudo cargar la importancia de features")
        return None


def get_class_label(class_num):
    """Convierte número de clase a etiqueta"""
    labels = {
        0: "🟢 Bajo",
        1: "🟡 Medio-Bajo",
        2: "🟠 Medio-Alto",
        3: "🔴 Alto"
    }
    return labels.get(class_num, "Desconocido")


def get_class_description(class_num):
    """Descripción detallada de cada clase"""
    descriptions = {
        0: "Sectores con emisiones bajas (< 0.108 kg CO₂e/$). Incluye principalmente servicios, seguros y actividades administrativas.",
        1: "Sectores con emisiones medio-bajas (0.108 - 0.173 kg CO₂e/$). Incluye manufactura ligera y servicios especializados.",
        2: "Sectores con emisiones medio-altas (0.173 - 0.329 kg CO₂e/$). Incluye agricultura, construcción y manufactura media.",
        3: "Sectores con emisiones altas (> 0.329 kg CO₂e/$). Incluye industria pesada, ganadería y manufactura intensiva."
    }
    return descriptions.get(class_num, "Sin descripción disponible")


# =============================================================================
# CARGA INICIAL DE DATOS
# =============================================================================

model, encoders = load_model()
df_data = load_dataset()
df_metrics = load_metrics()
df_importance = load_feature_importance()

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

# Título principal
st.title("🌍 Clasificador de Impacto Ambiental - Emisiones GEI")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("🧭 Navegación")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🎯 Predicción", "📊 Análisis del Modelo", "📈 Exploración de Datos", "ℹ️ Información"]
)

# =============================================================================
# PÁGINA 1: PREDICCIÓN
# =============================================================================

if page == "🎯 Predicción":
    st.header("🎯 Predicción de Impacto Ambiental")

    if model is None or df_data is None:
        st.error(
            "⚠️ No se pudo cargar el modelo o los datos. Verifica que los archivos estén en las carpetas correctas.")
        st.stop()

    st.markdown("""
    Esta herramienta predice el nivel de impacto ambiental de sectores industriales 
    basándose en sus factores de emisión de gases de efecto invernadero (GEI).
    """)

    # Dos columnas para entrada de datos
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 Selección de Sector")

        # Opción 1: Seleccionar sector existente
        use_existing = st.radio(
            "¿Cómo deseas ingresar los datos?",
            ["Seleccionar sector NAICS existente", "Ingresar valores manualmente"]
        )

        if use_existing == "Seleccionar sector NAICS existente":
            # Crear lista de opciones con código y nombre
            sector_options = df_data[['naics_code', 'naics_title']].copy()
            sector_options['display'] = sector_options['naics_code'].astype(str) + ' - ' + sector_options['naics_title']

            selected_display = st.selectbox(
                "Selecciona un sector NAICS:",
                options=sector_options['display'].values
            )

            # Obtener el sector seleccionado
            selected_naics = int(selected_display.split(' - ')[0])
            sector_data = df_data[df_data['naics_code'] == selected_naics].iloc[0]

            # Mostrar información del sector
            st.info(f"**Sector seleccionado:** {sector_data['naics_title']}")
            st.info(f"**Código NAICS:** {selected_naics}")

            # Preparar features para predicción
            feature_values = {
                'co2_emission': sector_data['co2_emission'],
                'ch4_emission': sector_data['ch4_emission'],
                'n2o_emission': sector_data['n2o_emission'],
                'hfcs_emission': sector_data['hfcs_emission'],
                'pfcs_emission': sector_data['pfcs_emission'],
                'sf6_emission': sector_data['sf6_emission'],
                'nf3_emission': sector_data['nf3_emission'],
                'hfc_pfc_unspecified': sector_data['hfc_pfc_unspecified'],
                'num_gases_emitted': sector_data['num_gases_emitted'],
                'gas_diversity': sector_data['gas_diversity'],
                'sef_mef_ratio': sector_data['sef_mef_ratio'],
                'naics_2dig': sector_data['naics_2dig'],
                'dominant_gas': sector_data['dominant_gas']
            }

    with col2:
        st.subheader("📋 Características del Sector")

        if use_existing == "Seleccionar sector NAICS existente":
            # Mostrar valores en formato de tabla
            features_df = pd.DataFrame({
                'Feature': list(feature_values.keys()),
                'Valor': [f"{v:.6f}" if isinstance(v, float) else str(v) for v in feature_values.values()]
            })
            st.dataframe(features_df, use_container_width=True, hide_index=True)

        else:
            # Entrada manual de valores
            st.markdown("**Ingresa los valores de emisiones:**")

            feature_values = {}

            # Features numéricas principales
            col_a, col_b = st.columns(2)

            with col_a:
                feature_values['co2_emission'] = st.number_input(
                    "CO₂ Emission", min_value=0.0, value=0.1, format="%.6f"
                )
                feature_values['ch4_emission'] = st.number_input(
                    "CH₄ Emission", min_value=0.0, value=0.01, format="%.6f"
                )
                feature_values['n2o_emission'] = st.number_input(
                    "N₂O Emission", min_value=0.0, value=0.001, format="%.6f"
                )
                feature_values['hfcs_emission'] = st.number_input(
                    "HFCs Emission", min_value=0.0, value=0.0001, format="%.6f"
                )
                feature_values['pfcs_emission'] = st.number_input(
                    "PFCs Emission", min_value=0.0, value=0.0001, format="%.6f"
                )
                feature_values['sf6_emission'] = st.number_input(
                    "SF₆ Emission", min_value=0.0, value=0.0001, format="%.6f"
                )

            with col_b:
                feature_values['nf3_emission'] = st.number_input(
                    "NF₃ Emission", min_value=0.0, value=0.0001, format="%.6f"
                )
                feature_values['hfc_pfc_unspecified'] = st.number_input(
                    "HFC/PFC Unspecified", min_value=0.0, value=0.0, format="%.6f"
                )
                feature_values['num_gases_emitted'] = st.number_input(
                    "Num Gases Emitted", min_value=0, value=5, step=1
                )
                feature_values['gas_diversity'] = st.number_input(
                    "Gas Diversity", min_value=0.0, max_value=1.0, value=0.5, format="%.6f"
                )
                feature_values['sef_mef_ratio'] = st.number_input(
                    "SEF/MEF Ratio", min_value=0.0, value=1.0, format="%.6f"
                )

                # Features categóricas
                feature_values['naics_2dig'] = st.selectbox(
                    "NAICS 2-digit code",
                    options=sorted(df_data['naics_2dig'].unique())
                )

                dominant_gas_options = sorted(df_data['dominant_gas'].unique())
                feature_values['dominant_gas'] = st.selectbox(
                    "Dominant Gas",
                    options=dominant_gas_options
                )

    # Botón de predicción
    st.markdown("---")

    if st.button("🚀 Realizar Predicción", type="primary", use_container_width=True):
        with st.spinner("Realizando predicción..."):
            try:
                # Preparar datos para predicción
                X_pred = pd.DataFrame([feature_values])

                # Codificar solo dominant_gas (naics_2dig ya es numérico)
                X_pred['dominant_gas'] = encoders['dominant_gas'].transform(X_pred['dominant_gas'])

                # Orden EXACTO según el modelo entrenado
                feature_order = [
                    'naics_2dig', 'sef_mef_ratio', 'co2_emission', 'ch4_emission',
                    'n2o_emission', 'hfcs_emission', 'pfcs_emission', 'sf6_emission',
                    'nf3_emission', 'hfc_pfc_unspecified', 'num_gases_emitted',
                    'gas_diversity', 'dominant_gas'
                ]
                X_pred = X_pred[feature_order]

                # Hacer predicción
                prediction = model.predict(X_pred)[0]
                probabilities = model.predict_proba(X_pred)[0]

                # Mostrar resultados
                st.markdown("---")
                st.subheader("📊 Resultados de la Predicción")

                # Resultado principal
                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

                with result_col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px;'>
                        <h1 style='font-size: 3em; margin: 0;'>{get_class_label(prediction)}</h1>
                        <p style='font-size: 1.2em; color: #666;'>Nivel de Impacto Ambiental</p>
                        <p style='font-size: 0.9em; color: #888; margin-top: 20px;'>{get_class_description(prediction)}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probabilidades por clase
                st.markdown("### 📈 Probabilidades por Clase")

                prob_df = pd.DataFrame({
                    'Clase': [get_class_label(i) for i in range(4)],
                    'Probabilidad': probabilities * 100
                })

                fig = px.bar(
                    prob_df,
                    x='Clase',
                    y='Probabilidad',
                    color='Probabilidad',
                    color_continuous_scale='RdYlGn_r',
                    text=prob_df['Probabilidad'].apply(lambda x: f"{x:.1f}%")
                )

                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Probabilidad (%)",
                    yaxis_range=[0, 100]
                )

                fig.update_traces(textposition='outside')

                st.plotly_chart(fig, use_container_width=True)

                # Métricas adicionales
                st.markdown("### 🔍 Análisis Detallado")

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                with metric_col1:
                    st.metric(
                        "Confianza de Predicción",
                        f"{probabilities[prediction] * 100:.1f}%",
                        delta=None
                    )

                with metric_col2:
                    st.metric(
                        "Segunda Opción",
                        get_class_label(np.argsort(probabilities)[-2]),
                        delta=f"{probabilities[np.argsort(probabilities)[-2]] * 100:.1f}%"
                    )

                with metric_col3:
                    emission_level = (feature_values['co2_emission'] +
                                      feature_values['ch4_emission'] +
                                      feature_values['n2o_emission'])
                    st.metric(
                        "Emisión Principal",
                        f"{emission_level:.4f}",
                        delta="kg CO₂e/$"
                    )

                with metric_col4:
                    st.metric(
                        "Gas Dominante",
                        feature_values['dominant_gas'],
                        delta=None
                    )

            except Exception as e:
                st.error(f"❌ Error durante la predicción: {e}")

# =============================================================================
# PÁGINA 2: ANÁLISIS DEL MODELO
# =============================================================================

elif page == "📊 Análisis del Modelo":
    st.header("📊 Análisis del Modelo de Clasificación")

    st.markdown("""
    Esta sección presenta el desempeño del modelo Random Forest utilizado para 
    clasificar el impacto ambiental de los sectores industriales.
    """)

    # Métricas del modelo
    if df_metrics is not None:
        st.subheader("🎯 Comparación de Modelos")

        # Mostrar tabla de métricas
        st.dataframe(
            df_metrics.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
            use_container_width=True,
            hide_index=True
        )

        # Gráfico comparativo
        fig = go.Figure()

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=df_metrics['Model'],
                y=df_metrics[metric],
                text=df_metrics[metric].apply(lambda x: f"{x:.2%}"),
                textposition='outside'
            ))

        fig.update_layout(
            title="Comparación de Métricas entre Modelos",
            xaxis_title="Modelo",
            yaxis_title="Score",
            barmode='group',
            height=500,
            yaxis_range=[0, 1.1]
        )

        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    if df_importance is not None:
        st.markdown("---")
        st.subheader("🔑 Importancia de Features")

        # Detectar nombres de columnas
        importance_col = 'importance' if 'importance' in df_importance.columns else 'Importance'
        feature_col = 'feature' if 'feature' in df_importance.columns else 'Feature'

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                df_importance.sort_values(importance_col, ascending=True).tail(10),
                x=importance_col,
                y=feature_col,
                orientation='h',
                color=importance_col,
                color_continuous_scale='Viridis',
                text=df_importance.sort_values(importance_col, ascending=True).tail(10)[importance_col].apply(
                    lambda x: f"{x:.2%}")
            )

            fig.update_layout(
                title="Top 10 Features Más Importantes",
                xaxis_title="Importancia",
                yaxis_title="",
                height=500,
                showlegend=False
            )

            fig.update_traces(textposition='outside')

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Top 5 Features:**")
            for idx, row in df_importance.head(5).iterrows():
                st.metric(
                    row[feature_col],
                    f"{row[importance_col]:.2%}",
                    delta=None
                )

    # Información del modelo
    st.markdown("---")
    st.subheader("ℹ️ Información del Modelo")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.info("""
        **Modelo Utilizado:**

        Random Forest Classifier

        - Número de árboles: 100
        - Estrategia: Bagging
        - Criterio: Gini
        """)

    with info_col2:
        st.info("""
        **Dataset:**

        - Total de sectores: 1,016
        - Training set: 762 (75%)
        - Test set: 254 (25%)
        - Features: 13
        """)

    with info_col3:
        st.info("""
        **Desempeño:**

        - Accuracy: 98.82%
        - F1-Score: 98.81%
        - Errores: 3/254
        - Balance: Excelente
        """)

# =============================================================================
# PÁGINA 3: EXPLORACIÓN DE DATOS
# =============================================================================

elif page == "📈 Exploración de Datos":
    st.header("📈 Exploración de Datos")

    if df_data is None:
        st.error("⚠️ No se pudieron cargar los datos.")
        st.stop()

    st.markdown("Explora el dataset utilizado para entrenar el modelo de clasificación.")

    # Filtros
    st.subheader("🔍 Filtros")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_class = st.multiselect(
            "Filtrar por clase de impacto:",
            options=[0, 1, 2, 3],
            default=[0, 1, 2, 3],
            format_func=get_class_label
        )

    with filter_col2:
        selected_naics = st.multiselect(
            "Filtrar por sector NAICS (2 dígitos):",
            options=sorted(df_data['naics_2dig'].unique()),
            default=sorted(df_data['naics_2dig'].unique())
        )

    # Aplicar filtros
    df_filtered = df_data[
        (df_data['impact_class'].isin(selected_class)) &
        (df_data['naics_2dig'].isin(selected_naics))
        ]

    # Mostrar dataset filtrado
    st.markdown(f"**Sectores mostrados:** {len(df_filtered)} de {len(df_data)}")

    st.dataframe(
        df_filtered[['naics_code', 'naics_title', 'impact_class',
                     'co2_emission', 'ch4_emission', 'n2o_emission',
                     'gas_diversity', 'dominant_gas']].head(50),
        use_container_width=True,
        hide_index=True
    )

    # Distribuciones
    st.markdown("---")
    st.subheader("📊 Distribuciones")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Distribución por clase
        class_dist = df_filtered['impact_class'].value_counts().sort_index()

        fig = px.pie(
            values=class_dist.values,
            names=[get_class_label(i) for i in class_dist.index],
            title="Distribución por Clase de Impacto",
            color_discrete_sequence=px.colors.sequential.RdBu_r
        )

        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        # Distribución de emisiones CO2
        fig = px.histogram(
            df_filtered,
            x='co2_emission',
            nbins=50,
            title="Distribución de Emisiones de CO₂",
            labels={'co2_emission': 'Emisión CO₂ (kg CO₂e/$)'},
            color_discrete_sequence=['#636EFA']
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    # Análisis de correlación
    st.markdown("---")
    st.subheader("🔗 Análisis de Correlación")

    numeric_cols = ['co2_emission', 'ch4_emission', 'n2o_emission',
                    'hfcs_emission', 'pfcs_emission', 'gas_diversity']

    corr_matrix = df_filtered[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title="Matriz de Correlación de Features Principales"
    )

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PÁGINA 4: INFORMACIÓN
# =============================================================================

elif page == "ℹ️ Información":
    st.header("ℹ️ Información del Proyecto")

    st.markdown("""
    ## 🌍 Clasificador de Impacto Ambiental - Emisiones GEI

    Esta aplicación utiliza Machine Learning para clasificar sectores industriales 
    según su nivel de impacto ambiental basándose en factores de emisión de gases 
    de efecto invernadero (GEI).

    ### 📋 Objetivo del Proyecto

    Desarrollar un sistema automatizado que permita:
    - Identificar rápidamente sectores de alto impacto ambiental
    - Facilitar la toma de decisiones en políticas de mitigación
    - Evaluar el impacto potencial de inversiones y cadenas de suministro
    - Proporcionar categorizaciones objetivas y consistentes

    ### 🎯 Clases de Impacto

    Los sectores se clasifican en 4 categorías:

    - **🟢 Bajo** (0.029 - 0.108 kg CO₂e/$): Servicios, seguros, actividades administrativas
    - **🟡 Medio-Bajo** (0.108 - 0.173 kg CO₂e/$): Manufactura ligera, servicios especializados
    - **🟠 Medio-Alto** (0.173 - 0.329 kg CO₂e/$): Agricultura, construcción, manufactura media
    - **🔴 Alto** (> 0.329 kg CO₂e/$): Industria pesada, ganadería, manufactura intensiva

    ### 🔬 Metodología

    **Datos utilizados:**
    - Factores de emisión GEI de la EPA v1.3.0
    - 1,016 sectores clasificados según NAICS 2017
    - 18 tipos de gases de efecto invernadero

    **Modelos evaluados:**
    1. Logistic Regression (89.30% accuracy)
    2. K-Nearest Neighbors (88.31% accuracy)
    3. Decision Tree (98.02% accuracy)
    4. **Random Forest (98.82% accuracy)** ⭐ Mejor modelo

    **Features más importantes:**
    1. Emisiones de CO₂ (32.25%)
    2. Emisiones de CH₄ (17.91%)
    3. Ratio SEF/MEF (12.09%)
    4. Emisiones de N₂O (10.71%)

    ### 📊 Desempeño del Modelo

    El modelo Random Forest alcanzó resultados excepcionales:
    - **Accuracy:** 98.82%
    - **F1-Score:** 98.81%
    - **Errores:** Solo 3 de 254 predicciones en el test set
    - **Balance:** Excelente distribución entre clases

    ### 🛠️ Tecnologías Utilizadas

    - **Python 3.8+**
    - **Scikit-learn** - Machine Learning
    - **Streamlit** - Interfaz web
    - **Pandas** - Manipulación de datos
    - **Plotly** - Visualizaciones interactivas

    ### 📚 Fuentes de Datos

    - **EPA Supply Chain GHG Emission Factors v1.3.0**
    - Datos de emisiones GEI de 2022
    - Potenciales de calentamiento global IPCC AR5
    - Factores expresados en dólares de 2022

    ### 👨‍💻 Desarrollo

    Proyecto desarrollado como parte de un análisis de Machine Learning 
    para clasificación de impacto ambiental en sectores industriales.

    ---

    ### 📧 Contacto

    Para más información sobre este proyecto, consulta la documentación 
    o contacta al equipo de desarrollo.
    """)

    # Información técnica adicional
    with st.expander("🔧 Información Técnica Adicional"):
        st.markdown("""
        **Estructura del Dataset:**

        - `naics_code`: Código NAICS 2017 (6 dígitos)
        - `naics_title`: Nombre del sector industrial
        - `co2_emission`: Emisiones de dióxido de carbono
        - `ch4_emission`: Emisiones de metano
        - `n2o_emission`: Emisiones de óxido nitroso
        - `hfcs_emission`: Emisiones de HFCs
        - `pfcs_emission`: Emisiones de PFCs
        - `sf6_emission`: Emisiones de SF₆
        - `nf3_emission`: Emisiones de NF₃
        - `hfc_pfc_unspecified`: HFC/PFC no especificados
        - `num_gases_emitted`: Número de gases emitidos
        - `gas_diversity`: Diversidad de gases emitidos
        - `sef_mef_ratio`: Ratio de factores con/sin márgenes
        - `naics_2dig`: Código NAICS a 2 dígitos (sector general)
        - `dominant_gas`: Gas con mayor emisión
        - `impact_class`: Clase de impacto (0-3)

        **Preprocesamiento:**

        1. Encoding de features categóricas (Label Encoding)
        2. Sin escalado para Random Forest (no lo requiere)
        3. Split estratificado 75/25 para preservar distribución de clases
        4. Validación cruzada para evaluación robusta
        """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🌍 Clasificador de Impacto Ambiental GEI | Powered by Machine Learning</p>
    <p>Datos: EPA Supply Chain GHG Emission Factors v1.3.0</p>
</div>
""", unsafe_allow_html=True)