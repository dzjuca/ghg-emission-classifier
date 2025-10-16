"""
eda_functions.py
Funciones para análisis exploratorio de datos (EDA)
"""

import pandas as pd
import numpy as np
from scipy import stats


def analyze_emission_factors(df):
    """
    Análisis específico de los factores de emisión

    Parameters:
    -----------
    df : DataFrame
        Dataset con factores de emisión

    Returns:
    --------
    dict con análisis de factores de emisión
    """
    print("\n🌍 ANÁLISIS DE FACTORES DE EMISIÓN")
    print("=" * 70)

    # Columnas de emisión
    sef_col = 'Supply Chain Emission Factors without Margins'
    mef_col = 'Margins of Supply Chain Emission Factors'
    total_col = 'Supply Chain Emission Factors with Margins'

    results = {}

    # Estadísticas básicas
    print("\n📊 Estadísticas de Factores de Emisión:")
    print(f"\nSEF (sin márgenes):")
    print(f"  Media: {df[sef_col].mean():.4f}")
    print(f"  Mediana: {df[sef_col].median():.4f}")
    print(f"  Desv. Est.: {df[sef_col].std():.4f}")
    print(f"  Min: {df[sef_col].min():.4f}")
    print(f"  Max: {df[sef_col].max():.4f}")

    print(f"\nMEF (márgenes):")
    print(f"  Media: {df[mef_col].mean():.4f}")
    print(f"  Mediana: {df[mef_col].median():.4f}")
    print(f"  Desv. Est.: {df[mef_col].std():.4f}")

    print(f"\nTotal (con márgenes):")
    print(f"  Media: {df[total_col].mean():.4f}")
    print(f"  Mediana: {df[total_col].median():.4f}")
    print(f"  Desv. Est.: {df[total_col].std():.4f}")

    # Proporción MEF/SEF
    df_temp = df.copy()
    df_temp['mef_sef_ratio'] = df_temp[mef_col] / df_temp[sef_col]

    print(f"\n📈 Ratio MEF/SEF:")
    print(f"  Media: {df_temp['mef_sef_ratio'].mean():.4f}")
    print(f"  Mediana: {df_temp['mef_sef_ratio'].median():.4f}")

    # Sectores con mayores emisiones
    print(f"\n🔝 TOP 10 SECTORES CON MAYORES EMISIONES:")
    top_sectors = df.nlargest(10, total_col)[['2017 NAICS Title', total_col]]
    for i, (idx, row) in enumerate(top_sectors.iterrows(), 1):
        print(f"  {i}. {row['2017 NAICS Title']}: {row[total_col]:.4f}")

    results['top_sectors'] = top_sectors

    # Sectores con menores emisiones
    print(f"\n🔽 TOP 10 SECTORES CON MENORES EMISIONES:")
    bottom_sectors = df.nsmallest(10, total_col)[['2017 NAICS Title', total_col]]
    for i, (idx, row) in enumerate(bottom_sectors.iterrows(), 1):
        print(f"  {i}. {row['2017 NAICS Title']}: {row[total_col]:.4f}")

    results['bottom_sectors'] = bottom_sectors

    print("=" * 70)

    return results


def analyze_naics_distribution(df):
    """
    Analiza la distribución de códigos NAICS

    Parameters:
    -----------
    df : DataFrame
        Dataset con códigos NAICS

    Returns:
    --------
    dict con análisis de distribución NAICS
    """
    print("\n🏭 ANÁLISIS DE DISTRIBUCIÓN NAICS")
    print("=" * 70)

    df_temp = df.copy()

    # Extraer categorías NAICS
    df_temp['naics_2dig'] = df_temp['2017 NAICS Code'] // 10000
    df_temp['naics_3dig'] = df_temp['2017 NAICS Code'] // 1000
    df_temp['naics_4dig'] = df_temp['2017 NAICS Code'] // 100

    print(f"\n📊 Distribución por nivel de agregación:")
    print(f"  Códigos únicos de 2 dígitos: {df_temp['naics_2dig'].nunique()}")
    print(f"  Códigos únicos de 3 dígitos: {df_temp['naics_3dig'].nunique()}")
    print(f"  Códigos únicos de 4 dígitos: {df_temp['naics_4dig'].nunique()}")
    print(f"  Códigos únicos de 6 dígitos: {df_temp['2017 NAICS Code'].nunique()}")

    # Frecuencia por sectores de 2 dígitos
    print(f"\n📈 Top sectores NAICS (2 dígitos):")
    naics_2dig_counts = df_temp['naics_2dig'].value_counts().head(10)
    for naics, count in naics_2dig_counts.items():
        print(f"  Sector {naics}: {count} subsectores")

    results = {
        'naics_2dig_counts': naics_2dig_counts,
        'unique_2dig': df_temp['naics_2dig'].nunique(),
        'unique_3dig': df_temp['naics_3dig'].nunique(),
        'unique_4dig': df_temp['naics_4dig'].nunique(),
        'unique_6dig': df_temp['2017 NAICS Code'].nunique()
    }

    print("=" * 70)

    return results


def analyze_ghg_types(df):
    """
    Analiza los tipos de GHG en el dataset

    Parameters:
    -----------
    df : DataFrame
        Dataset con columna 'GHG'

    Returns:
    --------
    dict con análisis de tipos de GHG
    """
    print("\n☁️ ANÁLISIS DE TIPOS DE GHG")
    print("=" * 70)

    ghg_counts = df['GHG'].value_counts()

    print(f"\n📊 Distribución de tipos de GHG:")
    print(f"  Total de tipos únicos: {df['GHG'].nunique()}")
    print(f"\n  Frecuencias:")
    for ghg, count in ghg_counts.items():
        pct = (count / len(df)) * 100
        print(f"    {ghg}: {count} ({pct:.2f}%)")

    results = {
        'ghg_counts': ghg_counts,
        'unique_ghg': df['GHG'].nunique()
    }

    print("=" * 70)

    return results


def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detecta outliers en una columna específica

    Parameters:
    -----------
    df : DataFrame
        Dataset
    column : str
        Nombre de la columna
    method : str
        Método de detección: 'iqr' o 'zscore'
    threshold : float
        Umbral para IQR (1.5) o Z-score (3)

    Returns:
    --------
    DataFrame con outliers y estadísticas
    """
    if column not in df.columns:
        print(f"❌ La columna '{column}' no existe")
        return None

    data = df[column].dropna()

    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        print(f"\n🔍 DETECCIÓN DE OUTLIERS (IQR) - {column}")
        print("=" * 70)
        print(f"Q1: {Q1:.4f}")
        print(f"Q3: {Q3:.4f}")
        print(f"IQR: {IQR:.4f}")
        print(f"Límite inferior: {lower_bound:.4f}")
        print(f"Límite superior: {upper_bound:.4f}")
        print(f"Outliers detectados: {len(outliers)} ({len(outliers) / len(df) * 100:.2f}%)")

    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        outliers = df[np.abs(stats.zscore(df[column].fillna(0))) > threshold]

        print(f"\n🔍 DETECCIÓN DE OUTLIERS (Z-Score) - {column}")
        print("=" * 70)
        print(f"Threshold: {threshold}")
        print(f"Outliers detectados: {len(outliers)} ({len(outliers) / len(df) * 100:.2f}%)")

    if len(outliers) > 0:
        print(f"\nTop 5 outliers:")
        top_outliers = outliers.nlargest(5, column)[['2017 NAICS Title', column]]
        for i, (idx, row) in enumerate(top_outliers.iterrows(), 1):
            print(f"  {i}. {row['2017 NAICS Title']}: {row[column]:.4f}")

    print("=" * 70)

    return outliers


def correlation_analysis(df, columns=None):
    """
    Análisis de correlación entre variables numéricas

    Parameters:
    -----------
    df : DataFrame
        Dataset
    columns : list, optional
        Lista de columnas a analizar

    Returns:
    --------
    DataFrame con matriz de correlación
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = df[columns].corr()

    print(f"\n🔗 ANÁLISIS DE CORRELACIÓN")
    print("=" * 70)
    print("\nMatriz de correlación:")
    print(corr_matrix.round(3))

    # Encontrar correlaciones fuertes
    print("\n📊 Correlaciones fuertes (|r| > 0.7):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_val:.3f}")

    print("=" * 70)

    return corr_matrix


def generate_eda_report(df, dataset_name="Dataset"):
    """
    Genera un reporte completo de EDA

    Parameters:
    -----------
    df : DataFrame
        Dataset a analizar
    dataset_name : str
        Nombre del dataset

    Returns:
    --------
    dict con todos los resultados del EDA
    """
    print("\n" + "=" * 70)
    print(f"📋 REPORTE EDA COMPLETO - {dataset_name}")
    print("=" * 70)

    results = {
        'dataset_name': dataset_name,
        'shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }

    # Análisis específico si es dataset de emisiones
    if 'Supply Chain Emission Factors with Margins' in df.columns:
        results['emission_analysis'] = analyze_emission_factors(df)
        results['naics_analysis'] = analyze_naics_distribution(df)

        # Outliers en factores de emisión
        results['outliers'] = detect_outliers(
            df,
            'Supply Chain Emission Factors with Margins',
            method='iqr'
        )

    # Análisis de GHG si es dataset2
    if 'GHG' in df.columns and df['GHG'].nunique() > 1:
        results['ghg_analysis'] = analyze_ghg_types(df)

    print(f"\n✅ Reporte EDA completado")
    print("=" * 70)

    return results


def rename_columns_dataset1(df):
    """
    Renombra columnas de Dataset 1 a snake_case.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset 1 original

    Returns:
    --------
    pd.DataFrame
        Dataset con columnas renombradas
    """
    column_mapping = {
        '2017 NAICS Code': 'naics_code',
        '2017 NAICS Title': 'naics_title',
        'GHG': 'ghg_type',
        'Unit': 'unit',
        'Supply Chain Emission Factors without Margins': 'sef_without_margins',
        'Margins of Supply Chain Emission Factors': 'mef_margins',
        'Supply Chain Emission Factors with Margins': 'sef_with_margins',
        'Reference USEEIO Code': 'useeio_code'
    }

    df_renamed = df.rename(columns=column_mapping)

    print("✅ Columnas renombradas - Dataset 1:")
    print(f"   Columnas nuevas: {list(df_renamed.columns)}")

    return df_renamed


def rename_columns_dataset2(df):
    """
    Renombra columnas de Dataset 2 a snake_case.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset 2 original

    Returns:
    --------
    pd.DataFrame
        Dataset con columnas renombradas
    """
    column_mapping = {
        '2017 NAICS Code': 'naics_code',
        '2017 NAICS Title': 'naics_title',
        'GHG': 'ghg_type',
        'Unit': 'unit',
        'Supply Chain Emission Factors without Margins': 'sef_without_margins',
        'Margins of Supply Chain Emission Factors': 'mef_margins',
        'Supply Chain Emission Factors with Margins': 'sef_with_margins',
        'Reference USEEIO Code': 'useeio_code'
    }

    df_renamed = df.rename(columns=column_mapping)

    print("✅ Columnas renombradas - Dataset 2:")
    print(f"   Columnas nuevas: {list(df_renamed.columns)}")

    return df_renamed


def convert_to_co2e(df, gwp_dict=None):
    """
    Convierte emisiones de kg gas/USD a kg CO2e/USD usando GWP.
    Solo convierte filas con unit = 'kg/2022 USD, purchaser price'.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset 2 con columnas renombradas
    gwp_dict : dict or None
        Diccionario con GWP por gas. Si None, usa valores por defecto.

    Returns:
    --------
    pd.DataFrame
        Dataset con todas las emisiones en kg CO2e/USD
    """
    import pandas as pd

    # GWP IPCC AR5 (100 años)
    if gwp_dict is None:
        gwp_dict = {
            'Carbon dioxide': 1,
            'Methane': 28,
            'Nitrous oxide': 265,
            'Carbon tetrafluoride': 6630,
            'Hexafluoroethane': 11100,
            'Sulfur hexafluoride': 23500,
            'Nitrogen trifluoride': 16100,
            'HFC-23': 12400,
            'HFC-32': 677,
            'HFC-125': 3170,
            'HFC-134a': 1300,
            'HFC-143a': 4800,
            'HFC-236fa': 8060,
            'Perfluoropropane': 8900,
            'Perfluorobutane': 9200,
            'Perfluorocyclobutane': 9540,
            'Perfluorohexane': 7910,
            'HFCs and PFCs, unspecified': 5950  # Promedio calculado
        }

    df_converted = df.copy()

    # Identificar filas a convertir
    mask_kg = df_converted['unit'] == 'kg/2022 USD, purchaser price'

    print(f"📊 Filas a convertir: {mask_kg.sum()} de {len(df_converted)}")

    # Aplicar conversión
    for gas, gwp in gwp_dict.items():
        mask_gas = (df_converted['ghg_type'] == gas) & mask_kg

        if mask_gas.sum() > 0:
            # Convertir factores
            df_converted.loc[mask_gas, 'sef_without_margins'] *= gwp
            df_converted.loc[mask_gas, 'mef_margins'] *= gwp
            df_converted.loc[mask_gas, 'sef_with_margins'] *= gwp

            # Actualizar unidad
            df_converted.loc[mask_gas, 'unit'] = 'kg CO2e/2022 USD, purchaser price'

            print(f"   ✅ {gas}: {mask_gas.sum()} filas convertidas (GWP={gwp})")

    # Verificación
    units_final = df_converted['unit'].value_counts()
    print(f"\n📋 Unidades finales:")
    for unit, count in units_final.items():
        print(f"   {unit}: {count} filas")

    return df_converted


def validate_conversion(df1, df2):
    """
    Valida que la suma de gases en Dataset 2 ≈ valores en Dataset 1.

    Parameters:
    -----------
    df1 : pd.DataFrame
        Dataset 1 (agregado, con columnas renombradas)
    df2 : pd.DataFrame
        Dataset 2 convertido a CO2e (con columnas renombradas)

    Returns:
    --------
    pd.DataFrame
        Tabla comparativa con diferencias
    """
    import pandas as pd

    # Excluir "HFCs and PFCs, unspecified" que ya está en CO2e
    df2_sin_hfc = df2[df2['ghg_type'] != 'HFCs and PFCs, unspecified']

    # Sumar por NAICS
    df2_summed = df2_sin_hfc.groupby('naics_code').agg({
        'sef_without_margins': 'sum',
        'sef_with_margins': 'sum'
    }).reset_index()

    # Merge con Dataset 1
    comparison = df1[['naics_code', 'naics_title', 'sef_without_margins', 'sef_with_margins']].merge(
        df2_summed,
        on='naics_code',
        suffixes=('_ds1', '_ds2')
    )

    # Calcular diferencias
    comparison['diff_sef_abs'] = comparison['sef_without_margins_ds1'] - comparison['sef_without_margins_ds2']
    comparison['diff_sef_pct'] = (comparison['diff_sef_abs'] / comparison['sef_without_margins_ds1']) * 100

    comparison['diff_total_abs'] = comparison['sef_with_margins_ds1'] - comparison['sef_with_margins_ds2']
    comparison['diff_total_pct'] = (comparison['diff_total_abs'] / comparison['sef_with_margins_ds1']) * 100

    # Estadísticas
    print("=" * 70)
    print("📊 VALIDACIÓN CRUZADA: Dataset 1 vs Dataset 2 (suma de gases)")
    print("=" * 70)
    print(f"\nDiferencia promedio (SEF without margins): {comparison['diff_sef_pct'].mean():.2f}%")
    print(f"Diferencia promedio (SEF with margins): {comparison['diff_total_pct'].mean():.2f}%")
    print(f"\nDiferencia máxima: {comparison['diff_total_pct'].abs().max():.2f}%")
    print(f"Sectores con >10% diferencia: {(comparison['diff_total_pct'].abs() > 10).sum()}")

    # Top 10 mayores diferencias
    print("\n🔍 Top 10 sectores con mayor diferencia:")
    top_diff = comparison.nlargest(10, 'diff_total_pct', keep='all')[
        ['naics_code', 'naics_title', 'sef_with_margins_ds1', 'sef_with_margins_ds2', 'diff_total_pct']
    ]
    print(top_diff.to_string(index=False))

    return comparison


def save_clean_datasets(df1, df2, output_dir='datos'):
    """
    Guarda datasets limpios y procesados.

    Parameters:
    -----------
    df1 : pd.DataFrame
        Dataset 1 limpio
    df2 : pd.DataFrame
        Dataset 2 convertido a CO2e
    output_dir : str
        Directorio de salida
    """
    import os

    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Rutas
    path1 = os.path.join(output_dir, 'df1_toFeatures.csv')
    path2 = os.path.join(output_dir, 'df2_toFeatures.csv')

    # Guardar
    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    print("=" * 70)
    print("💾 DATASETS GUARDADOS")
    print("=" * 70)
    print(f"✅ Dataset 1: {path1}")
    print(f"   Dimensiones: {df1.shape}")
    print(f"   Columnas: {list(df1.columns)}")
    print(f"\n✅ Dataset 2: {path2}")
    print(f"   Dimensiones: {df2.shape}")
    print(f"   Columnas: {list(df2.columns)}")
    print("=" * 70)