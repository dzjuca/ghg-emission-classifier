"""
data_loader.py
Funciones para cargar y validar los datasets de emisiones GHG
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_datasets(path_dataset1='data/raw/dataset1.csv',
                  path_dataset2='data/raw/dataset2.csv'):
    """
    Carga los dos datasets de emisiones GHG

    Parameters:
    -----------
    path_dataset1 : str
        Ruta al dataset1 (factores agregados)
    path_dataset2 : str
        Ruta al dataset2 (factores desagregados por gas)

    Returns:
    --------
    df1, df2 : tuple of DataFrames
        Los dos datasets cargados
    """
    try:
        df1 = pd.read_csv(path_dataset1)
        df2 = pd.read_csv(path_dataset2)

        print("✅ Datasets cargados exitosamente")
        print(f"   - Dataset 1: {df1.shape[0]} filas, {df1.shape[1]} columnas")
        print(f"   - Dataset 2: {df2.shape[0]} filas, {df2.shape[1]} columnas")

        return df1, df2

    except FileNotFoundError as e:
        print(f"❌ Error: No se encontró el archivo {e.filename}")
        return None, None
    except Exception as e:
        print(f"❌ Error al cargar datasets: {str(e)}")
        return None, None


def get_basic_info(df, dataset_name="Dataset"):
    """
    Muestra información básica del dataset

    Parameters:
    -----------
    df : DataFrame
        Dataset a analizar
    dataset_name : str
        Nombre para mostrar en el reporte
    """
    print("=" * 70)
    print(f"📊 INFORMACIÓN BÁSICA - {dataset_name}")
    print("=" * 70)

    print(f"\n🔢 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

    print(f"\n📋 Columnas:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col} ({df[col].dtype})")

    print(f"\n🔍 Valores nulos:")
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("   ✅ No hay valores nulos")
    else:
        print(nulls[nulls > 0])

    print(f"\n📈 Valores únicos por columna:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"   - {col}: {unique_count}")

    print(f"\n💾 Uso de memoria: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print("\n" + "=" * 70)


def get_descriptive_stats(df, numeric_cols=None):
    """
    Obtiene estadísticas descriptivas de columnas numéricas

    Parameters:
    -----------
    df : DataFrame
        Dataset a analizar
    numeric_cols : list, optional
        Lista de columnas numéricas específicas a analizar

    Returns:
    --------
    DataFrame con estadísticas descriptivas
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = df[numeric_cols].describe().T

    # Añadir estadísticas adicionales
    stats['median'] = df[numeric_cols].median()
    stats['skew'] = df[numeric_cols].skew()
    stats['kurtosis'] = df[numeric_cols].kurtosis()
    stats['missing'] = df[numeric_cols].isnull().sum()
    stats['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df)) * 100

    # Reordenar columnas
    cols_order = ['count', 'missing', 'missing_pct', 'mean', 'median',
                  'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis']
    stats = stats[[col for col in cols_order if col in stats.columns]]

    return stats


def validate_data_quality(df):
    """
    Valida la calidad de los datos

    Parameters:
    -----------
    df : DataFrame
        Dataset a validar

    Returns:
    --------
    dict con resultados de validación
    """
    results = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
    }

    print("\n🔎 VALIDACIÓN DE CALIDAD DE DATOS")
    print("=" * 70)
    print(f"Total de filas: {results['total_rows']:,}")
    print(f"Total de columnas: {results['total_cols']}")
    print(f"Valores faltantes: {results['missing_values']:,}")
    print(f"Filas duplicadas: {results['duplicate_rows']:,}")
    print(f"Columnas numéricas: {results['numeric_cols']}")
    print(f"Columnas categóricas: {results['categorical_cols']}")

    # Verificar integridad de columnas esperadas
    expected_cols = [
        '2017 NAICS Code',
        '2017 NAICS Title',
        'GHG',
        'Unit',
        'Supply Chain Emission Factors without Margins',
        'Margins of Supply Chain Emission Factors',
        'Supply Chain Emission Factors with Margins',
        'Reference USEEIO Code'
    ]

    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        print(f"\n⚠️  Columnas faltantes: {missing_cols}")
    else:
        print("\n✅ Todas las columnas esperadas están presentes")

    print("=" * 70)

    return results


def preview_data(df, n_rows=5):
    """
    Muestra una vista previa del dataset

    Parameters:
    -----------
    df : DataFrame
        Dataset a mostrar
    n_rows : int
        Número de filas a mostrar

    Returns:
    --------
    None (imprime la vista previa)
    """
    print(f"\n👀 PRIMERAS {n_rows} FILAS:")
    print("=" * 70)
    display_df = df.head(n_rows)
    print(display_df.to_string())
    print("=" * 70)


def get_column_summary(df, col_name):
    """
    Obtiene resumen detallado de una columna específica

    Parameters:
    -----------
    df : DataFrame
        Dataset
    col_name : str
        Nombre de la columna

    Returns:
    --------
    dict con información de la columna
    """
    if col_name not in df.columns:
        print(f"❌ La columna '{col_name}' no existe en el dataset")
        return None

    col_data = df[col_name]

    summary = {
        'name': col_name,
        'dtype': col_data.dtype,
        'count': col_data.count(),
        'missing': col_data.isnull().sum(),
        'missing_pct': (col_data.isnull().sum() / len(df)) * 100,
        'unique': col_data.nunique(),
    }

    if pd.api.types.is_numeric_dtype(col_data):
        summary.update({
            'mean': col_data.mean(),
            'median': col_data.median(),
            'std': col_data.std(),
            'min': col_data.min(),
            'max': col_data.max(),
            'q25': col_data.quantile(0.25),
            'q75': col_data.quantile(0.75),
        })
    else:
        top_values = col_data.value_counts().head(10)
        summary['top_values'] = top_values.to_dict()

    print(f"\n📊 RESUMEN DE COLUMNA: {col_name}")
    print("=" * 70)
    for key, value in summary.items():
        if key != 'top_values':
            print(f"{key}: {value}")

    if 'top_values' in summary:
        print("\nTop 10 valores más frecuentes:")
        for val, count in summary['top_values'].items():
            print(f"  {val}: {count}")

    print("=" * 70)

    return summary