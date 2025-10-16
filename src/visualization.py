"""
visualization.py
Funciones para visualización de datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style():
    """
    Configura el estilo general de los gráficos
    """
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10


def plot_distribution(df, column, title=None, xlabel=None, bins=50, color='steelblue'):
    """
    Grafica la distribución de una variable

    Parameters:
    -----------
    df : DataFrame
        Dataset
    column : str
        Columna a graficar
    title : str, optional
        Título del gráfico
    xlabel : str, optional
        Etiqueta del eje X
    bins : int
        Número de bins para el histograma
    color : str
        Color del gráfico

    Returns:
    --------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Histograma
    ax.hist(df[column].dropna(), bins=bins, color=color, alpha=0.7, edgecolor='black')

    # Líneas de referencia
    mean_val = df[column].mean()
    median_val = df[column].median()

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.4f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.4f}')

    ax.set_title(title or f'Distribución de {column}', fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel or column, fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_boxplot(df, column, title=None, ylabel=None, color='lightblue'):
    """
    Grafica un boxplot de una variable

    Parameters:
    -----------
    df : DataFrame
        Dataset
    column : str
        Columna a graficar
    title : str, optional
        Título del gráfico
    ylabel : str, optional
        Etiqueta del eje Y
    color : str
        Color del boxplot

    Returns:
    --------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot(df[column].dropna(), vert=True, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    for patch in bp['boxes']:
        patch.set_facecolor(color)

    ax.set_title(title or f'Boxplot de {column}', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel or column, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir estadísticas
    stats_text = f"Media: {df[column].mean():.4f}\nMediana: {df[column].median():.4f}\nDesv. Est.: {df[column].std():.4f}"
    ax.text(1.15, df[column].median(), stats_text,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig, ax


def plot_top_sectors(df, column, n=15, title=None, figsize=(12, 8)):
    """
    Grafica los top N sectores según un factor de emisión

    Parameters:
    -----------
    df : DataFrame
        Dataset
    column : str
        Columna de factores de emisión
    n : int
        Número de sectores a mostrar
    title : str, optional
        Título del gráfico

    Returns:
    --------
    fig, ax : matplotlib objects
    """
    top_data = df.nlargest(n, column)[['2017 NAICS Title', column]].sort_values(column)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_data)))
    bars = ax.barh(range(len(top_data)), top_data[column], color=colors, edgecolor='black')

    ax.set_yticks(range(len(top_data)))
    ax.set_yticklabels(top_data['2017 NAICS Title'], fontsize=9)
    ax.set_xlabel('Factor de Emisión (kg CO₂e/$)', fontsize=11)
    ax.set_title(title or f'Top {n} Sectores con Mayores Emisiones',
                 fontsize=14, fontweight='bold')

    # Añadir valores en las barras
    for i, (idx, row) in enumerate(top_data.iterrows()):
        ax.text(row[column], i, f' {row[column]:.3f}',
                va='center', fontsize=8, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    return fig, ax


def plot_naics_distribution(df, level='2dig', top_n=15):
    """
    Grafica la distribución de sectores NAICS

    Parameters:
    -----------
    df : DataFrame
        Dataset
    level : str
        Nivel de agregación: '2dig', '3dig', '4dig'
    top_n : int
        Número de categorías a mostrar

    Returns:
    --------
    fig, ax : matplotlib objects
    """
    df_temp = df.copy()

    if level == '2dig':
        df_temp['naics_cat'] = df_temp['2017 NAICS Code'] // 10000
        title = 'Distribución de Sectores NAICS (2 dígitos)'
    elif level == '3dig':
        df_temp['naics_cat'] = df_temp['2017 NAICS Code'] // 1000
        title = 'Distribución de Sectores NAICS (3 dígitos)'
    else:
        df_temp['naics_cat'] = df_temp['2017 NAICS Code'] // 100
        title = 'Distribución de Sectores NAICS (4 dígitos)'

    naics_counts = df_temp['naics_cat'].value_counts().head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(naics_counts)))
    bars = ax.barh(range(len(naics_counts)), naics_counts.values, color=colors, edgecolor='black')

    ax.set_yticks(range(len(naics_counts)))
    ax.set_yticklabels([f'Sector {int(code)}' for code in naics_counts.index])
    ax.set_xlabel('Número de Subsectores', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')

    for i, val in enumerate(naics_counts.values):
        ax.text(val, i, f' {val}', va='center', fontsize=9, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    return fig, ax


def plot_correlation_heatmap(df, columns=None, annot=True, cmap='coolwarm', figsize=(10, 8)):
    """
    Grafica un heatmap de correlación

    Parameters:
    -----------
    df : DataFrame
        Dataset
    columns : list, optional
        Columnas a incluir
    annot : bool
        Si mostrar valores en el heatmap
    cmap : str
        Mapa de colores
    figsize : tuple
        Tamaño de la figura

    Returns:
    --------
    fig, ax : matplotlib objects
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = df[columns].corr()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.2f', ax=ax)

    ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, ax


def plot_scatter_with_regression(df, x_col, y_col, title=None, xlabel=None, ylabel=None):
    """
    Scatter plot con línea de regresión

    Parameters:
    -----------
    df : DataFrame
        Dataset
    x_col : str
        Columna para eje X
    y_col : str
        Columna para eje Y
    title : str, optional
        Título del gráfico

    Returns:
    --------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    ax.scatter(df[x_col], df[y_col], alpha=0.5, s=50, color='steelblue', edgecolor='black')

    # Línea de regresión
    z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
    p = np.poly1d(z)
    ax.plot(df[x_col].sort_values(), p(df[x_col].sort_values()),
            "r--", linewidth=2, label=f'y = {z[0]:.4f}x + {z[1]:.4f}')

    # Calcular R²
    correlation = df[[x_col, y_col]].corr().iloc[0, 1]
    r_squared = correlation ** 2

    ax.set_title(title or f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel or x_col, fontsize=11)
    ax.set_ylabel(ylabel or y_col, fontsize=11)
    ax.legend(title=f'R² = {r_squared:.4f}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_emission_comparison(df):
    """
    Compara los diferentes componentes de emisión (SEF, MEF, Total)

    Parameters:
    -----------
    df : DataFrame
        Dataset con factores de emisión

    Returns:
    --------
    fig, axes : matplotlib objects
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sef_col = 'Supply Chain Emission Factors without Margins'
    mef_col = 'Margins of Supply Chain Emission Factors'
    total_col = 'Supply Chain Emission Factors with Margins'

    # 1. Histograma SEF
    axes[0, 0].hist(df[sef_col], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df[sef_col].mean(), color='red', linestyle='--', linewidth=2, label='Media')
    axes[0, 0].set_title('Distribución de SEF (sin márgenes)', fontweight='bold')
    axes[0, 0].set_xlabel('kg CO₂e/')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Histograma MEF
    axes[0, 1].hist(df[mef_col], bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df[mef_col].mean(), color='red', linestyle='--', linewidth=2, label='Media')
    axes[0, 1].set_title('Distribución de MEF (márgenes)', fontweight='bold')
    axes[0, 1].set_xlabel('kg CO₂e/')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Histograma Total
    axes[1, 0].hist(df[total_col], bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df[total_col].mean(), color='red', linestyle='--', linewidth=2, label='Media')
    axes[1, 0].set_title('Distribución de Factor Total (con márgenes)', fontweight='bold')
    axes[1, 0].set_xlabel('kg CO₂e/')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Scatter SEF vs MEF
    axes[1, 1].scatter(df[sef_col], df[mef_col], alpha=0.5, s=30, color='purple', edgecolor='black')
    axes[1, 1].set_title('SEF vs MEF', fontweight='bold')
    axes[1, 1].set_xlabel('SEF (kg CO₂e/$)')
    axes[1, 1].set_ylabel('MEF (kg CO₂e/$)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_ghg_composition(df2):
    """
    Visualiza la composición de gases GHG en dataset2

    Parameters:
    -----------
    df2 : DataFrame
        Dataset2 con gases desagregados

    Returns:
    --------
    fig, axes : matplotlib objects
    """
    ghg_counts = df2['GHG'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(ghg_counts)))
    axes[0].barh(range(len(ghg_counts)), ghg_counts.values, color=colors, edgecolor='black')
    axes[0].set_yticks(range(len(ghg_counts)))
    axes[0].set_yticklabels(ghg_counts.index, fontsize=9)
    axes[0].set_xlabel('Número de Registros', fontsize=11)
    axes[0].set_title('Frecuencia de Tipos de GHG', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    for i, val in enumerate(ghg_counts.values):
        axes[0].text(val, i, f' {val}', va='center', fontsize=9)

    # 2. Pie chart
    axes[1].pie(ghg_counts.values, labels=ghg_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 9})
    axes[1].set_title('Proporción de Tipos de GHG', fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig, axes


def plot_outliers_analysis(df, column, method='iqr', threshold=1.5):
    """
    Visualiza el análisis de outliers

    Parameters:
    -----------
    df : DataFrame
        Dataset
    column : str
        Columna a analizar
    method : str
        'iqr' o 'zscore'
    threshold : float
        Umbral para detección

    Returns:
    --------
    fig, axes : matplotlib objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    data = df[column].dropna()

    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    else:
        from scipy import stats
        z_scores = np.abs(stats.zscore(data))
        outliers_mask = np.abs(stats.zscore(df[column].fillna(0))) > threshold

    # 1. Boxplot
    bp = axes[0].boxplot(data, vert=True, patch_artist=True, widths=0.5)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[0].set_title(f'Boxplot - {column}', fontweight='bold')
    axes[0].set_ylabel('Valor')
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Histograma con outliers resaltados
    axes[1].hist(df[~outliers_mask][column], bins=50, color='steelblue',
                 alpha=0.7, edgecolor='black', label='Normal')
    axes[1].hist(df[outliers_mask][column], bins=20, color='red',
                 alpha=0.7, edgecolor='black', label='Outliers')
    axes[1].set_title(f'Distribución con Outliers - {column}', fontweight='bold')
    axes[1].set_xlabel('Valor')
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """
    Guarda una figura en archivo

    Parameters:
    -----------
    fig : matplotlib figure
        Figura a guardar
    filename : str
        Nombre del archivo (incluir ruta y extensión)
    dpi : int
        Resolución
    bbox_inches : str
        Ajuste de bordes
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"✅ Figura guardada: {filename}")


def create_eda_visualizations(df1, df2, save_path='outputs/'):
    """
    Crea todas las visualizaciones del EDA y las guarda

    Parameters:
    -----------
    df1 : DataFrame
        Dataset 1 (factores agregados)
    df2 : DataFrame
        Dataset 2 (factores desagregados)
    save_path : str
        Ruta donde guardar las figuras

    Returns:
    --------
    dict con todas las figuras generadas
    """
    import os
    os.makedirs(save_path, exist_ok=True)

    print("\n📊 Generando visualizaciones EDA...")
    print("=" * 70)

    figures = {}

    # 1. Distribución del factor total
    print("1. Distribución del factor total de emisiones...")
    fig1, _ = plot_distribution(
        df1,
        'Supply Chain Emission Factors with Margins',
        title='Distribución del Factor Total de Emisiones',
        xlabel='kg CO₂e/$ (2022 USD)'
    )
    save_figure(fig1, f'{save_path}01_distribucion_factor_total.png')
    figures['distribucion_total'] = fig1

    # 2. Boxplot del factor total
    print("2. Boxplot del factor total...")
    fig2, _ = plot_boxplot(
        df1,
        'Supply Chain Emission Factors with Margins',
        title='Análisis de Outliers - Factor Total de Emisiones',
        ylabel='kg CO₂e/$ (2022 USD)'
    )
    save_figure(fig2, f'{save_path}02_boxplot_factor_total.png')
    figures['boxplot_total'] = fig2

    # 3. Top sectores con mayores emisiones
    print("3. Top 15 sectores con mayores emisiones...")
    fig3, _ = plot_top_sectors(
        df1,
        'Supply Chain Emission Factors with Margins',
        n=15,
        title='Top 15 Sectores con Mayores Emisiones GHG'
    )
    save_figure(fig3, f'{save_path}03_top15_sectores.png')
    figures['top_sectores'] = fig3

    # 4. Distribución NAICS
    print("4. Distribución de sectores NAICS...")
    fig4, _ = plot_naics_distribution(df1, level='2dig', top_n=15)
    save_figure(fig4, f'{save_path}04_distribucion_naics.png')
    figures['naics_dist'] = fig4

    # 5. Comparación de componentes de emisión
    print("5. Comparación SEF vs MEF vs Total...")
    fig5, _ = plot_emission_comparison(df1)
    save_figure(fig5, f'{save_path}05_comparacion_emisiones.png')
    figures['comparacion_emisiones'] = fig5

    # 6. Composición de gases GHG
    print("6. Composición de tipos de GHG...")
    fig6, _ = plot_ghg_composition(df2)
    save_figure(fig6, f'{save_path}06_composicion_ghg.png')
    figures['ghg_composition'] = fig6

    # 7. Análisis de outliers
    print("7. Análisis detallado de outliers...")
    fig7, _ = plot_outliers_analysis(
        df1,
        'Supply Chain Emission Factors with Margins',
        method='iqr'
    )
    save_figure(fig7, f'{save_path}07_analisis_outliers.png')
    figures['outliers_analysis'] = fig7

    # 8. Correlación entre variables
    print("8. Matriz de correlación...")
    emission_cols = [
        'Supply Chain Emission Factors without Margins',
        'Margins of Supply Chain Emission Factors',
        'Supply Chain Emission Factors with Margins'
    ]
    fig8, _ = plot_correlation_heatmap(df1, columns=emission_cols, figsize=(8, 6))
    save_figure(fig8, f'{save_path}08_matriz_correlacion.png')
    figures['correlacion'] = fig8

    print("\n✅ Todas las visualizaciones generadas y guardadas")
    print(f"📁 Ubicación: {save_path}")
    print("=" * 70)

    return figures


def plot_bubble_chart_top_sectors(df, n_top=30,
                                  sef_col='Supply Chain Emission Factors without Margins',
                                  total_col='Supply Chain Emission Factors with Margins',
                                  title_col='2017 NAICS Title',
                                  output_path=None):
    """
    Genera gráfico de burbujas para los top N sectores por emisiones (EDA).

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset 1 con factores de emisión
    n_top : int
        Número de sectores top a mostrar (default: 30)
    sef_col : str
        Columna con factor sin márgenes
    total_col : str
        Columna con factor total (para ranking y tamaño)
    title_col : str
        Columna con nombres de sectores
    output_path : str or None
        Ruta para guardar la imagen

    Returns:
    --------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Seleccionar top N sectores
    df_plot = df.nlargest(n_top, total_col).copy()

    # Valores para ejes
    x = df_plot[sef_col]
    y = df_plot[total_col]

    # Tamaño de burbujas proporcional al factor total
    sizes = (y / y.max()) * 1000

    # Color degradado según magnitud
    colors = y

    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 10))

    # Scatter plot con colormap
    scatter = ax.scatter(x, y, s=sizes, c=colors, cmap='YlOrRd',
                         alpha=0.6, edgecolors='black', linewidth=1.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Factor Total (kg CO₂e/$)', fontsize=10)

    # Etiquetar top 10 sectores
    for idx, row in df_plot.head(10).iterrows():
        sector_name = row[title_col]
        # Acortar nombres muy largos
        if len(sector_name) > 35:
            sector_name = sector_name[:32] + '...'

        ax.annotate(
            sector_name,
            xy=(row[sef_col], row[total_col]),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            ha='left',
            va='bottom'
        )

    # Etiquetas y título
    ax.set_xlabel('Factor sin Márgenes (kg CO₂e/2022 USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Factor Total con Márgenes (kg CO₂e/2022 USD)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n_top} Sectores Industriales por Emisiones de GEI',
                 fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Añadir línea de referencia y=x (para ver diferencia de márgenes)
    max_val = max(x.max(), y.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='Sin márgenes = Con márgenes')
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()

    # Guardar si se especifica
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {output_path}")

    return fig


def plot_bubble_chart_sectors(df, n_top=30,
                              total_col='Supply Chain Emission Factors with Margins',
                              sef_col='Supply Chain Emission Factors without Margins',
                              mef_col='Margins of Supply Chain Emission Factors',
                              title_col='2017 NAICS Title',
                              output_path=None):
    """
    Gráfico de burbujas: Eje X=Ranking, Eje Y=Emisiones, Tamaño=Magnitud.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset 1
    n_top : int
        Top sectores a mostrar
    total_col : str
        Factor total (eje Y)
    sef_col : str
        Factor sin márgenes (para comparar)
    mef_col : str
        Márgenes (para tamaño burbuja alternativo)
    title_col : str
        Nombres de sectores
    output_path : str or None
        Guardar imagen

    Returns:
    --------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Top N
    df_plot = df.nlargest(n_top, total_col).reset_index(drop=True)

    # Ejes
    x = np.arange(1, len(df_plot) + 1)  # Ranking 1, 2, 3...
    y = df_plot[total_col]

    # Tamaño: proporcional a la magnitud
    sizes = (y / y.max()) * 2000

    # Color: degradado por magnitud
    colors = y

    # Figura
    fig, ax = plt.subplots(figsize=(16, 10))

    # Burbujas
    scatter = ax.scatter(x, y, s=sizes, c=colors, cmap='Reds',
                         alpha=0.7, edgecolors='black', linewidth=2)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Emisiones (kg CO₂e/$)', fontsize=11, fontweight='bold')

    # Anotar top 10
    for i in range(min(10, len(df_plot))):
        name = df_plot.loc[i, title_col]
        if len(name) > 30:
            name = name[:27] + '...'
        ax.annotate(name,
                    xy=(i + 1, df_plot.loc[i, total_col]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=9,
                    ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Etiquetas
    ax.set_xlabel('Ranking de Sectores', fontsize=13, fontweight='bold')
    ax.set_ylabel('Factor Total de Emisiones (kg CO₂e/2022 USD)', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {n_top} Sectores por Emisiones de GEI - Vista de Burbujas',
                 fontsize=15, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Límites
    ax.set_xlim(0, len(df_plot) + 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {output_path}")

    return fig


def plot_bubble_chart_all_sectors(df,
                                  outlier_threshold=1.5,
                                  sef_col='Supply Chain Emission Factors without Margins',
                                  total_col='Supply Chain Emission Factors with Margins',
                                  title_col='2017 NAICS Title',
                                  output_path=None):
    """
    Gráfico de burbujas con TODOS los sectores, etiquetando solo outliers.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset 1 completo (1,016 sectores)
    outlier_threshold : float
        Umbral para etiquetar (default: 1.5 kg CO2e/$)
    sef_col : str
        Factor sin márgenes (eje X)
    total_col : str
        Factor total (eje Y y tamaño)
    title_col : str
        Nombres de sectores
    output_path : str or None
        Guardar imagen

    Returns:
    --------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Usar todos los datos
    df_plot = df.copy()

    # Identificar outliers
    outliers = df_plot[df_plot[total_col] > outlier_threshold]

    # Valores para ejes
    x = df_plot[sef_col]
    y = df_plot[total_col]

    # Tamaño de burbujas
    sizes = (y / y.max()) * 800

    # Colores según si es outlier o no
    colors = ['#e74c3c' if val > outlier_threshold else '#3498db'
              for val in y]

    # Figura
    fig, ax = plt.subplots(figsize=(14, 10))

    # Scatter plot
    scatter = ax.scatter(x, y, s=sizes, c=colors,
                         alpha=0.6, edgecolors='black', linewidth=0.8)

    # Etiquetar SOLO outliers
    for idx, row in outliers.iterrows():
        sector_name = row[title_col]
        if len(sector_name) > 35:
            sector_name = sector_name[:32] + '...'

        ax.annotate(
            sector_name,
            xy=(row[sef_col], row[total_col]),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='red'),
            ha='left',
            va='bottom',
            arrowprops=dict(arrowstyle='->', color='red', lw=0.8)
        )

    # Línea de umbral
    ax.axhline(y=outlier_threshold, color='red', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Umbral outliers ({outlier_threshold} kg CO₂e/$)')

    # Etiquetas
    ax.set_xlabel('Factor sin Márgenes (kg CO₂e/2022 USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Factor Total con Márgenes (kg CO₂e/2022 USD)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Distribución Completa de Emisiones por Sector Industrial\n({len(df_plot)} sectores, {len(outliers)} outliers etiquetados)',
        fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label=f'Outliers (> {outlier_threshold})'),
        Patch(facecolor='#3498db', label=f'Normal (≤ {outlier_threshold})')
    ]
    ax.legend(handles=legend_elements, title='Tipo de Sector', loc='upper left', fontsize=10)

    # Estadísticas en el gráfico
    textstr = f'Total sectores: {len(df_plot)}\nOutliers: {len(outliers)} ({len(outliers) / len(df_plot) * 100:.1f}%)'
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {output_path}")

    return fig


def plot_sunburst_emissions_xxxz(df,
                            total_col='Supply Chain Emission Factors with Margins',
                            naics_col='2017 NAICS Code',
                            title_col='2017 NAICS Title',
                            output_path=None):
    """
    Gráfico Sunburst jerárquico de emisiones por sectores NAICS (4 niveles).

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset 1 completo
    total_col : str
        Columna con factor total de emisiones
    naics_col : str
        Columna con código NAICS
    title_col : str
        Columna con nombres de sectores
    output_path : str or None
        Guardar como HTML

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    import plotly.express as px
    import pandas as pd

    # Preparar datos con jerarquía
    df_sun = df.copy()

    # Extraer niveles NAICS
    df_sun['NAICS_2'] = df_sun[naics_col].astype(str).str[:2]
    df_sun['NAICS_3'] = df_sun[naics_col].astype(str).str[:3]
    df_sun['NAICS_4'] = df_sun[naics_col].astype(str).str[:4]
    df_sun['NAICS_6'] = df_sun[naics_col].astype(str)

    # Nombres descriptivos para cada nivel
    naics_2_names = {
        '11': 'Agricultura',
        '21': 'Minería',
        '22': 'Utilities',
        '23': 'Construcción',
        '31': 'Manufactura 1',
        '32': 'Manufactura 2',
        '33': 'Manufactura 3',
        '42': 'Comercio Mayorista',
        '44': 'Comercio Minorista 1',
        '45': 'Comercio Minorista 2',
        '48': 'Transporte 1',
        '49': 'Transporte 2',
        '51': 'Información',
        '52': 'Finanzas',
        '53': 'Inmobiliario',
        '54': 'Servicios Profesionales',
        '55': 'Management',
        '56': 'Admin/Soporte',
        '61': 'Educación',
        '62': 'Salud',
        '71': 'Entretenimiento',
        '72': 'Alojamiento/Comida',
        '81': 'Otros Servicios'
    }

    df_sun['Label_2'] = df_sun['NAICS_2'].map(naics_2_names).fillna(df_sun['NAICS_2'])
    df_sun['Label_3'] = 'Subsector ' + df_sun['NAICS_3']
    df_sun['Label_4'] = 'Grupo ' + df_sun['NAICS_4']
    df_sun['Label_6'] = df_sun[title_col].str[:40]  # Limitar longitud

    # Crear estructura jerárquica
    hierarchy_data = []

    for idx, row in df_sun.iterrows():
        hierarchy_data.append({
            'labels': row['Label_6'],
            'parents': row['Label_4'],
            'values': row[total_col],
            'ids': row['NAICS_6']
        })

        # Nivel 4
        if row['Label_4'] not in [h['ids'] for h in hierarchy_data]:
            hierarchy_data.append({
                'labels': row['Label_4'],
                'parents': row['Label_3'],
                'values': 0,  # Se calculará automáticamente
                'ids': row['NAICS_4']
            })

        # Nivel 3
        if row['Label_3'] not in [h['ids'] for h in hierarchy_data]:
            hierarchy_data.append({
                'labels': row['Label_3'],
                'parents': row['Label_2'],
                'values': 0,
                'ids': row['NAICS_3']
            })

        # Nivel 2
        if row['Label_2'] not in [h['ids'] for h in hierarchy_data]:
            hierarchy_data.append({
                'labels': row['Label_2'],
                'parents': '',  # Raíz
                'values': 0,
                'ids': row['NAICS_2']
            })

    # Convertir a DataFrame
    df_hierarchy = pd.DataFrame(hierarchy_data)

    # Crear sunburst
    fig = px.sunburst(
        df_hierarchy,
        names='labels',
        parents='parents',
        values='values',
        ids='ids',
        title='Distribución Jerárquica de Emisiones GEI por Sector Industrial (NAICS)',
        color='values',
        color_continuous_scale='YlOrRd',
        hover_data=['values']
    )

    # Configuración
    fig.update_traces(
        textinfo='label+percent parent',
        hovertemplate='<b>%{label}</b><br>Emisiones: %{value:.3f} kg CO₂e/$<br>%{percentParent}<extra></extra>'
    )

    fig.update_layout(
        width=1200,
        height=1000,
        font=dict(size=10),
        coloraxis_colorbar=dict(title='Emisiones<br>(kg CO₂e/$)')
    )

    # Guardar HTML interactivo
    if output_path:
        fig.write_html(output_path)
        print(f"✅ Sunburst guardado: {output_path}")

    return fig


def plot_sunburst_emissions(df,
                            total_col='Supply Chain Emission Factors with Margins',
                            naics_col='2017 NAICS Code',
                            title_col='2017 NAICS Title',
                            output_path=None):
    """
    Gráfico Sunburst jerárquico de emisiones por sectores NAICS (4 niveles).
    """
    import plotly.graph_objects as go
    import pandas as pd

    # Preparar datos
    df_sun = df.copy()
    df_sun[naics_col] = df_sun[naics_col].astype(str).str.zfill(6)

    # Extraer niveles
    df_sun['NAICS_2'] = df_sun[naics_col].str[:2]
    df_sun['NAICS_3'] = df_sun[naics_col].str[:3]
    df_sun['NAICS_4'] = df_sun[naics_col].str[:4]
    df_sun['NAICS_6'] = df_sun[naics_col]

    # Diccionario de nombres nivel 2
    naics_2_names = {
        '11': 'Agricultura', '21': 'Minería', '22': 'Utilities',
        '23': 'Construcción', '31': 'Manufactura-1', '32': 'Manufactura-2',
        '33': 'Manufactura-3', '42': 'Comercio Mayor', '44': 'Comercio Menor-1',
        '45': 'Comercio Menor-2', '48': 'Transporte-1', '49': 'Transporte-2',
        '51': 'Información', '52': 'Finanzas', '53': 'Inmobiliario',
        '54': 'Profesional', '55': 'Management', '56': 'Admin/Soporte',
        '61': 'Educación', '62': 'Salud', '71': 'Entretenimiento',
        '72': 'Alojamiento', '81': 'Otros Servicios'
    }

    # Listas para construcción
    labels = []
    parents = []
    values = []
    ids = []

    # Nivel 2 (raíz)
    for n2 in df_sun['NAICS_2'].unique():
        labels.append(naics_2_names.get(n2, f'Sector-{n2}'))
        parents.append('')
        values.append(df_sun[df_sun['NAICS_2'] == n2][total_col].sum())
        ids.append(n2)

    # Nivel 3
    for n3 in df_sun['NAICS_3'].unique():
        n2 = n3[:2]
        labels.append(f'Sub-{n3}')
        parents.append(n2)
        values.append(df_sun[df_sun['NAICS_3'] == n3][total_col].sum())
        ids.append(n3)

    # Nivel 4
    for n4 in df_sun['NAICS_4'].unique():
        n3 = n4[:3]
        labels.append(f'Grp-{n4}')
        parents.append(n3)
        values.append(df_sun[df_sun['NAICS_4'] == n4][total_col].sum())
        ids.append(n4)

    # Nivel 6 (sectores individuales)
    for idx, row in df_sun.iterrows():
        sector_name = row[title_col][:30]
        labels.append(sector_name)
        parents.append(row['NAICS_4'])
        values.append(row[total_col])
        ids.append(row['NAICS_6'])

    # Crear figura
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        ids=ids,
        branchvalues='total',
        marker=dict(
            colorscale='YlOrRd',
            cmid=df[total_col].median()
        ),
        hovertemplate='<b>%{label}</b><br>Emisiones: %{value:.3f} kg CO₂e/$<extra></extra>'
    ))

    fig.update_layout(
        title='Distribución Jerárquica de Emisiones GEI por Sector NAICS',
        width=1200,
        height=1000,
        font=dict(size=10)
    )

    if output_path:
        fig.write_html(output_path)
        print(f"✅ Sunburst guardado: {output_path}")

    return fig


def plot_sunburst_top_sectors(df,
                              n_top=150,
                              total_col='Supply Chain Emission Factors with Margins',
                              naics_col='2017 NAICS Code',
                              title_col='2017 NAICS Title',
                              output_path=None):
    """
    Sunburst con 4 niveles: Niveles 2-3-4 completos, Nivel 6 top N sectores.
    """
    import plotly.graph_objects as go
    import pandas as pd

    # Preparar datos
    df_sun = df.copy()
    df_sun[naics_col] = df_sun[naics_col].astype(str).str.zfill(6)

    # Extraer niveles
    df_sun['NAICS_2'] = df_sun[naics_col].str[:2]
    df_sun['NAICS_3'] = df_sun[naics_col].str[:3]
    df_sun['NAICS_4'] = df_sun[naics_col].str[:4]
    df_sun['NAICS_6'] = df_sun[naics_col]

    # FILTRAR: Solo top N sectores para nivel 6
    df_top = df_sun.nlargest(n_top, total_col)

    # Diccionario nombres nivel 2
    naics_2_names = {
        '11': 'Agricultura', '21': 'Minería', '22': 'Utilities',
        '23': 'Construcción', '31': 'Manufactura-1', '32': 'Manufactura-2',
        '33': 'Manufactura-3', '42': 'Comercio Mayor', '44': 'Comercio Menor-1',
        '45': 'Comercio Menor-2', '48': 'Transporte-1', '49': 'Transporte-2',
        '51': 'Información', '52': 'Finanzas', '53': 'Inmobiliario',
        '54': 'Profesional', '55': 'Management', '56': 'Admin/Soporte',
        '61': 'Educación', '62': 'Salud', '71': 'Entretenimiento',
        '72': 'Alojamiento', '81': 'Otros Servicios'
    }

    # Construcción de listas
    labels = []
    parents = []
    values = []
    ids = []

    # Nivel 2 (todos)
    for n2 in df_sun['NAICS_2'].unique():
        labels.append(naics_2_names.get(n2, f'Sector-{n2}'))
        parents.append('')
        values.append(df_sun[df_sun['NAICS_2'] == n2][total_col].sum())
        ids.append(n2)

    # Nivel 3 (todos)
    for n3 in df_sun['NAICS_3'].unique():
        n2 = n3[:2]
        labels.append(f'Sub-{n3}')
        parents.append(n2)
        values.append(df_sun[df_sun['NAICS_3'] == n3][total_col].sum())
        ids.append(n3)

    # Nivel 4 (todos)
    for n4 in df_sun['NAICS_4'].unique():
        n3 = n4[:3]
        labels.append(f'Grp-{n4}')
        parents.append(n3)
        values.append(df_sun[df_sun['NAICS_4'] == n4][total_col].sum())
        ids.append(n4)

    # Nivel 6 (SOLO TOP N)
    for idx, row in df_top.iterrows():
        sector_name = row[title_col][:35]
        labels.append(sector_name)
        parents.append(row['NAICS_4'])
        values.append(row[total_col])
        ids.append(row['NAICS_6'])

    # Crear figura
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        ids=ids,
        branchvalues='total',
        marker=dict(
            colorscale='YlOrRd',
            cmid=df[total_col].median()
        ),
        hovertemplate='<b>%{label}</b><br>Emisiones: %{value:.3f} kg CO₂e/$<extra></extra>'
    ))

    fig.update_layout(
        title=f'Distribución Jerárquica de Emisiones GEI - Top {n_top} Sectores',
        width=1200,
        height=1000,
        font=dict(size=10)
    )

    if output_path:
        fig.write_html(output_path)
        print(f"✅ Sunburst (Top {n_top}) guardado: {output_path}")

    return fig


def plot_sunburst_threshold(df,
                            threshold=0.5,
                            total_col='Supply Chain Emission Factors with Margins',
                            naics_col='2017 NAICS Code',
                            title_col='2017 NAICS Title',
                            output_path=None):
    """
    Sunburst con 4 niveles: Niveles 2-3-4 completos, Nivel 6 solo sectores > umbral.
    """
    import plotly.graph_objects as go
    import pandas as pd

    # Preparar datos
    df_sun = df.copy()
    df_sun[naics_col] = df_sun[naics_col].astype(str).str.zfill(6)

    # Extraer niveles
    df_sun['NAICS_2'] = df_sun[naics_col].str[:2]
    df_sun['NAICS_3'] = df_sun[naics_col].str[:3]
    df_sun['NAICS_4'] = df_sun[naics_col].str[:4]
    df_sun['NAICS_6'] = df_sun[naics_col]

    # FILTRAR: Solo sectores > umbral para nivel 6
    df_filtered = df_sun[df_sun[total_col] > threshold]

    # Diccionario nombres nivel 2
    naics_2_names = {
        '11': 'Agricultura', '21': 'Minería', '22': 'Utilities',
        '23': 'Construcción', '31': 'Manufactura-1', '32': 'Manufactura-2',
        '33': 'Manufactura-3', '42': 'Comercio Mayor', '44': 'Comercio Menor-1',
        '45': 'Comercio Menor-2', '48': 'Transporte-1', '49': 'Transporte-2',
        '51': 'Información', '52': 'Finanzas', '53': 'Inmobiliario',
        '54': 'Profesional', '55': 'Management', '56': 'Admin/Soporte',
        '61': 'Educación', '62': 'Salud', '71': 'Entretenimiento',
        '72': 'Alojamiento', '81': 'Otros Servicios'
    }

    # Construcción de listas
    labels = []
    parents = []
    values = []
    ids = []

    # Nivel 2 (todos)
    for n2 in df_sun['NAICS_2'].unique():
        labels.append(naics_2_names.get(n2, f'Sector-{n2}'))
        parents.append('')
        values.append(df_sun[df_sun['NAICS_2'] == n2][total_col].sum())
        ids.append(n2)

    # Nivel 3 (todos)
    for n3 in df_sun['NAICS_3'].unique():
        n2 = n3[:2]
        labels.append(f'Sub-{n3}')
        parents.append(n2)
        values.append(df_sun[df_sun['NAICS_3'] == n3][total_col].sum())
        ids.append(n3)

    # Nivel 4 (todos)
    for n4 in df_sun['NAICS_4'].unique():
        n3 = n4[:3]
        labels.append(f'Grp-{n4}')
        parents.append(n3)
        values.append(df_sun[df_sun['NAICS_4'] == n4][total_col].sum())
        ids.append(n4)

    # Nivel 6 (SOLO > THRESHOLD)
    for idx, row in df_filtered.iterrows():
        sector_name = row[title_col][:35]
        labels.append(sector_name)
        parents.append(row['NAICS_4'])
        values.append(row[total_col])
        ids.append(row['NAICS_6'])

    # Crear figura
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        ids=ids,
        branchvalues='total',
        marker=dict(
            colorscale='YlOrRd',
            cmid=df[total_col].median()
        ),
        hovertemplate='<b>%{label}</b><br>Emisiones: %{value:.3f} kg CO₂e/$<extra></extra>'
    ))

    fig.update_layout(
        title=f'Distribución Jerárquica de Emisiones GEI - Sectores > {threshold} kg CO₂e/$',
        width=1200,
        height=1000,
        font=dict(size=10)
    )

    if output_path:
        fig.write_html(output_path)
        print(f"✅ Sunburst (Threshold {threshold}) guardado: {output_path}")

    return fig


def plot_sunburst_high_impact(df,
                              total_col='Supply Chain Emission Factors with Margins',
                              naics_col='2017 NAICS Code',
                              title_col='2017 NAICS Title',
                              output_path=None):
    """
    Sunburst enfocado en sectores de ALTO IMPACTO únicamente.
    Solo incluye sectores NAICS-2 con mayores emisiones.
    """
    import plotly.graph_objects as go
    import pandas as pd

    # Preparar datos
    df_sun = df.copy()
    df_sun[naics_col] = df_sun[naics_col].astype(str).str.zfill(6)

    # Extraer niveles
    df_sun['NAICS_2'] = df_sun[naics_col].str[:2]
    df_sun['NAICS_3'] = df_sun[naics_col].str[:3]
    df_sun['NAICS_4'] = df_sun[naics_col].str[:4]
    df_sun['NAICS_6'] = df_sun[naics_col]

    # FILTRAR: Solo sectores de alto impacto en nivel 2
    sectores_alto_impacto = ['11', '21', '31', '32', '33', '48', '56']
    df_filtered = df_sun[df_sun['NAICS_2'].isin(sectores_alto_impacto)]

    # Diccionario nombres
    naics_2_names = {
        '11': 'Agricultura',
        '21': 'Minería',
        '31': 'Manufactura-1',
        '32': 'Manufactura-2',
        '33': 'Manufactura-3',
        '48': 'Transporte-1',
        '56': 'Admin/Soporte'
    }

    # Construcción de listas
    labels = []
    parents = []
    values = []
    ids = []

    # Nivel 2 (solo alto impacto)
    for n2 in sectores_alto_impacto:
        if n2 in df_filtered['NAICS_2'].values:
            labels.append(naics_2_names[n2])
            parents.append('')
            values.append(df_filtered[df_filtered['NAICS_2'] == n2][total_col].sum())
            ids.append(n2)

    # Nivel 3 (solo de sectores filtrados)
    for n3 in df_filtered['NAICS_3'].unique():
        n2 = n3[:2]
        labels.append(f'Sub-{n3}')
        parents.append(n2)
        values.append(df_filtered[df_filtered['NAICS_3'] == n3][total_col].sum())
        ids.append(n3)

    # Nivel 4 (solo de sectores filtrados)
    for n4 in df_filtered['NAICS_4'].unique():
        n3 = n4[:3]
        labels.append(f'Grp-{n4}')
        parents.append(n3)
        values.append(df_filtered[df_filtered['NAICS_4'] == n4][total_col].sum())
        ids.append(n4)

    # Nivel 6 (todos los sectores individuales de grupos filtrados)
    for idx, row in df_filtered.iterrows():
        sector_name = row[title_col][:40]
        labels.append(sector_name)
        parents.append(row['NAICS_4'])
        values.append(row[total_col])
        ids.append(row['NAICS_6'])

    # Crear figura
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        ids=ids,
        branchvalues='total',
        marker=dict(
            colorscale='YlOrRd',
            cmid=df_filtered[total_col].median()
        ),
        hovertemplate='<b>%{label}</b><br>Emisiones: %{value:.3f} kg CO₂e/$<extra></extra>'
    ))

    fig.update_layout(
        title='Distribución Jerárquica - Sectores de Alto Impacto Ambiental',
        width=1200,
        height=1000,
        font=dict(size=11)
    )

    if output_path:
        fig.write_html(output_path)
        print(f"✅ Sunburst (Alto Impacto) guardado: {output_path}")

    # Stats
    n_sectores = len(df_filtered)
    pct = (n_sectores / len(df)) * 100
    print(f"📊 Sectores incluidos: {n_sectores} de {len(df)} ({pct:.1f}%)")
    print(f"🎯 Sectores NAICS-2: {', '.join([naics_2_names[s] for s in sectores_alto_impacto])}")

    return fig