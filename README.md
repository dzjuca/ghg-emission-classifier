# 🌍 Clasificador de Impacto Ambiental GEI - Aplicación Streamlit

## 📋 Descripción

Aplicación web interactiva desarrollada con Streamlit para predecir el nivel de impacto ambiental de sectores industriales basándose en factores de emisión de gases de efecto invernadero (GEI).

## 🚀 Instalación y Ejecución

### 1. Requisitos Previos

```bash
# Python 3.8 o superior
python --version

# pip actualizado
pip install --upgrade pip
```

### 2. Instalar Dependencias

```bash
# En el directorio raíz del proyecto
pip install -r requirements.txt
```

### 3. Verificar Estructura de Archivos

Asegúrate de tener la siguiente estructura:

```
proyecto_ghg/
├── streamlit_app.py          # Aplicación principal
├── requirements.txt           # Dependencias
├── datos/
│   └── dataset_for_modeling.csv
└── modelos/
    ├── best_model.pkl        # Modelo Random Forest entrenado
    ├── encoders.pkl          # Encoders para features categóricas
    ├── model_comparison.csv  # Comparación de modelos
    └── feature_importance.csv # Importancia de features
```

### 4. Ejecutar la Aplicación

```bash
# Desde el directorio raíz del proyecto
streamlit run streamlit_app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 🎯 Funcionalidades

### 1. 🎯 Predicción

**Dos modos de entrada:**

- **Selección de Sector NAICS**: Elige un sector existente del dropdown
- **Entrada Manual**: Ingresa valores personalizados para todos los features

**Resultados mostrados:**
- Clase de impacto predicha (Bajo, Medio-Bajo, Medio-Alto, Alto)
- Probabilidades para cada clase
- Confianza de la predicción
- Métricas adicionales del sector

### 2. 📊 Análisis del Modelo

- Comparación de 4 modelos evaluados
- Métricas de desempeño (Accuracy, Precision, Recall, F1-Score)
- Gráficos comparativos
- Top 10 features más importantes
- Información técnica del modelo

### 3. 📈 Exploración de Datos

- Visualización del dataset completo
- Filtros por clase de impacto y sector NAICS
- Distribuciones y estadísticas
- Matriz de correlación entre features
- Análisis interactivo

### 4. ℹ️ Información

- Objetivo del proyecto
- Metodología utilizada
- Descripción de las clases
- Tecnologías implementadas
- Fuentes de datos

## 📊 Clases de Impacto

| Clase | Rango (kg CO₂e/$) | Descripción |
|-------|-------------------|-------------|
| 🟢 **Bajo** | 0.029 - 0.108 | Servicios, seguros, actividades administrativas |
| 🟡 **Medio-Bajo** | 0.108 - 0.173 | Manufactura ligera, servicios especializados |
| 🟠 **Medio-Alto** | 0.173 - 0.329 | Agricultura, construcción, manufactura media |
| 🔴 **Alto** | > 0.329 | Industria pesada, ganadería, manufactura intensiva |

## 🔍 Features del Modelo

### Features Numéricas (10):
1. `co2_emission` - Emisiones de CO₂
2. `ch4_emission` - Emisiones de CH₄ (metano)
3. `n2o_emission` - Emisiones de N₂O (óxido nitroso)
4. `total_emission` - Emisiones totales
5. `emission_diversity` - Diversidad de gases
6. `sef_mef_ratio` - Ratio de factores con/sin márgenes
7. `high_impact_gas_count` - Número de gases de alto impacto
8. `total_gwp` - Potencial de calentamiento global total
9. `avg_gwp` - GWP promedio
10. `max_gwp` - GWP máximo

### Features Categóricas (2):
1. `naics_2dig` - Código NAICS a 2 dígitos (sector general)
2. `dominant_gas` - Gas con mayor emisión

## 📈 Desempeño del Modelo

**Random Forest (Modelo Seleccionado):**
- **Accuracy**: 98.82%
- **F1-Score**: 98.81%
- **Precision**: 98.83%
- **Recall**: 98.82%
- **Errores**: 3 de 254 predicciones en test set

## 🛠️ Tecnologías

- **Python 3.8+**
- **Streamlit 1.28+** - Framework web
- **Scikit-learn 1.3+** - Machine Learning
- **Pandas 2.0+** - Manipulación de datos
- **Plotly 5.17+** - Visualizaciones interactivas
- **NumPy 1.24+** - Computación numérica

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError"
```bash
# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

### Error: "FileNotFoundError: best_model.pkl"
```bash
# Verificar que ejecutaste el notebook 03_Modeling.ipynb
# El modelo debe estar en: modelos/best_model.pkl
```

### Error: "streamlit: command not found"
```bash
# Asegúrate de que streamlit está instalado
pip install streamlit
```

### La aplicación no carga
```bash
# Verificar puerto en uso
streamlit run streamlit_app.py --server.port 8502

# O especificar otro puerto
streamlit run streamlit_app.py --server.port 8080
```

## 📝 Uso de Ejemplo

### Ejemplo 1: Predicción con Sector Existente

1. Navega a "🎯 Predicción"
2. Selecciona "Seleccionar sector NAICS existente"
3. Elige un sector del dropdown (ej: "331110 - Iron and Steel Mills")
4. Click en "🚀 Realizar Predicción"
5. Revisa los resultados y probabilidades

### Ejemplo 2: Predicción Manual

1. Navega a "🎯 Predicción"
2. Selecciona "Ingresar valores manualmente"
3. Ingresa valores para cada feature:
   - CO₂ Emission: 0.5
   - CH₄ Emission: 0.02
   - N₂O Emission: 0.003
   - Total Emission: 0.6
   - etc.
4. Click en "🚀 Realizar Predicción"

### Ejemplo 3: Explorar Datos

1. Navega a "📈 Exploración de Datos"
2. Usa los filtros para seleccionar clases específicas
3. Explora las distribuciones y correlaciones
4. Identifica patrones en el dataset

## 🔄 Actualización del Modelo

Si re-entrenas el modelo con nuevos datos:

1. Ejecuta el notebook `03_Modeling.ipynb`
2. Verifica que se generaron los archivos en `modelos/`
3. Reinicia la aplicación Streamlit
4. Los cambios se reflejarán automáticamente

## 📧 Soporte

Para reportar problemas o sugerencias:
1. Verifica que tienes todos los archivos necesarios
2. Revisa la sección de Troubleshooting
3. Consulta la documentación del proyecto

## 📄 Licencia

Proyecto educativo - Análisis de Machine Learning para clasificación de impacto ambiental.

---

**Desarrollado con** 🌍 **para un futuro más sostenible**