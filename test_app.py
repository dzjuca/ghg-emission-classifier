"""
=============================================================================
TEST_APP.PY - SCRIPT DE PRUEBA PARA LA APLICACIÓN STREAMLIT
=============================================================================
Verifica que todos los componentes necesarios estén presentes y funcionales
antes de ejecutar la aplicación Streamlit.
=============================================================================
"""

import sys
from pathlib import Path
import pickle
import pandas as pd


def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_file_exists(filepath, description):
    """Verifica si un archivo existe"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size / 1024  # KB
        print(f"✅ {description}")
        print(f"   Ubicación: {filepath}")
        print(f"   Tamaño: {size:.2f} KB")
        return True
    else:
        print(f"❌ {description}")
        print(f"   Ubicación esperada: {filepath}")
        print(f"   FALTA - La aplicación no funcionará sin este archivo")
        return False


def check_module(module_name):
    """Verifica si un módulo de Python está instalado"""
    try:
        __import__(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError:
        print(f"❌ {module_name} - NO INSTALADO")
        return False


def test_model_loading():
    """Prueba cargar el modelo y verificar su estructura"""
    try:
        with open('modelos/best_model.pkl', 'rb') as f:
            model = pickle.load(f)

        print(f"✅ Modelo cargado correctamente")
        print(f"   Tipo: {type(model).__name__}")

        if hasattr(model, 'n_estimators'):
            print(f"   N° de estimadores: {model.n_estimators}")
        if hasattr(model, 'n_features_in_'):
            print(f"   N° de features: {model.n_features_in_}")

        return True
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return False


def test_encoders_loading():
    """Prueba cargar los encoders"""
    try:
        with open('modelos/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)

        print(f"✅ Encoders cargados correctamente")
        print(f"   Encoders disponibles: {list(encoders.keys())}")

        return True
    except Exception as e:
        print(f"❌ Error al cargar encoders: {e}")
        return False


def test_dataset_loading():
    """Prueba cargar el dataset"""
    try:
        df = pd.read_csv('datos/dataset_for_modeling.csv')

        print(f"✅ Dataset cargado correctamente")
        print(f"   Filas: {len(df)}")
        print(f"   Columnas: {len(df.columns)}")
        print(f"   Columnas: {list(df.columns)}")

        # Verificar columnas requeridas
        required_cols = ['naics_code', 'naics_title', 'impact_class',
                         'co2_emission', 'ch4_emission', 'n2o_emission',
                         'naics_2dig', 'dominant_gas']

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"⚠️  Columnas faltantes: {missing_cols}")
            return False

        return True
    except Exception as e:
        print(f"❌ Error al cargar dataset: {e}")
        return False


def main():
    """Función principal de prueba"""
    print_header("🧪 TEST DE LA APLICACIÓN STREAMLIT GEI")

    all_passed = True

    # 1. Verificar módulos de Python
    print_header("📦 Verificando Módulos de Python")

    modules = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'plotly',
        'pickle'
    ]

    for module in modules:
        if not check_module(module):
            all_passed = False

    # 2. Verificar archivos del modelo
    print_header("🤖 Verificando Archivos del Modelo")

    files_to_check = [
        ('modelos/best_model.pkl', 'Modelo Random Forest'),
        ('modelos/encoders.pkl', 'Encoders para features categóricas'),
        ('modelos/model_comparison.csv', 'Comparación de modelos'),
        ('modelos/feature_importance.csv', 'Importancia de features')
    ]

    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_passed = False

    # 3. Verificar datos
    print_header("📊 Verificando Datos")

    if not check_file_exists('datos/dataset_for_modeling.csv', 'Dataset para modelado'):
        all_passed = False

    # 4. Verificar archivo de la aplicación
    print_header("🌐 Verificando Aplicación Streamlit")

    if not check_file_exists('streamlit_app.py', 'Archivo principal de Streamlit'):
        all_passed = False

    # 5. Pruebas de carga
    print_header("🔄 Pruebas de Carga")

    print("\n1. Probando carga del modelo...")
    if not test_model_loading():
        all_passed = False

    print("\n2. Probando carga de encoders...")
    if not test_encoders_loading():
        all_passed = False

    print("\n3. Probando carga del dataset...")
    if not test_dataset_loading():
        all_passed = False

    # 6. Resumen final
    print_header("📋 RESUMEN DE PRUEBAS")

    if all_passed:
        print("✅ ¡TODAS LAS PRUEBAS PASARON!")
        print("\n🚀 La aplicación está lista para ejecutarse.")
        print("\nPara iniciar la aplicación, ejecuta:")
        print("   streamlit run streamlit_app.py")
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON")
        print("\n⚠️  La aplicación puede no funcionar correctamente.")
        print("\nAcciones recomendadas:")
        print("   1. Instala los módulos faltantes: pip install -r requirements.txt")
        print("   2. Verifica que ejecutaste el notebook 03_Modeling.ipynb")
        print("   3. Asegúrate de que todos los archivos estén en las carpetas correctas")

    print("\n" + "=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)