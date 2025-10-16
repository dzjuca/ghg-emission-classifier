import pandas as pd
import numpy as np
from scipy.stats import entropy


def extract_gas_emissions(df2):
    """
    Extrae emisiones por gas desde Dataset 2 (ya en CO2e).
    Retorna DataFrame con naics_code como índice.
    """
    gas_mapping = {
        'co2_emission': ['Carbon dioxide'],
        'ch4_emission': ['Methane'],
        'n2o_emission': ['Nitrous oxide'],
        'hfcs_emission': ['HFC-23', 'HFC-32', 'HFC-125', 'HFC-134a', 'HFC-143a', 'HFC-236fa'],
        'pfcs_emission': ['Carbon tetrafluoride', 'Hexafluoroethane', 'Perfluoropropane',
                          'Perfluorobutane', 'Perfluorocyclobutane', 'Perfluorohexane'],
        'sf6_emission': ['Sulfur hexafluoride'],
        'nf3_emission': ['Nitrogen trifluoride'],
        'hfc_pfc_unspecified': ['HFCs and PFCs, unspecified']
    }

    result = []

    for naics in df2['naics_code'].unique():
        sector_data = df2[df2['naics_code'] == naics]
        row = {'naics_code': naics}

        for feature_name, gases in gas_mapping.items():
            total = sector_data[sector_data['ghg_type'].isin(gases)]['sef_with_margins'].sum()
            row[feature_name] = total

        result.append(row)

    return pd.DataFrame(result)


def calculate_gas_stats(gas_df):
    """
    Calcula features agregados: num_gases, dominant_gas, gas_diversity.
    """
    emission_cols = [col for col in gas_df.columns if col.endswith('_emission')]

    stats = []
    for idx, row in gas_df.iterrows():
        emissions = row[emission_cols].values

        # Número de gases significativos
        num_gases = (emissions > 0.001).sum()

        # Gas dominante
        dominant_idx = emissions.argmax()
        dominant_gas = emission_cols[dominant_idx].replace('_emission', '')

        # Diversidad (Shannon entropy)
        emissions_norm = emissions / emissions.sum() if emissions.sum() > 0 else emissions
        gas_diversity = entropy(emissions_norm + 1e-10)

        stats.append({
            'naics_code': row['naics_code'],
            'num_gases_emitted': num_gases,
            'dominant_gas': dominant_gas,
            'gas_diversity': gas_diversity
        })

    return pd.DataFrame(stats)


def calculate_proportions(gas_df):
    """
    Calcula proporciones de CO2, CH4, N2O.
    """
    props = []

    for idx, row in gas_df.iterrows():
        total = row[['co2_emission', 'ch4_emission', 'n2o_emission', 'hfcs_emission',
                     'pfcs_emission', 'sf6_emission', 'nf3_emission', 'hfc_pfc_unspecified']].sum()

        if total > 0:
            co2_prop = row['co2_emission'] / total
            ch4_prop = row['ch4_emission'] / total
            n2o_prop = row['n2o_emission'] / total
        else:
            co2_prop = ch4_prop = n2o_prop = 0

        props.append({
            'naics_code': row['naics_code'],
            'co2_proportion': co2_prop,
            'ch4_proportion': ch4_prop,
            'n2o_proportion': n2o_prop
        })

    return pd.DataFrame(props)


def create_target(df1):
    """
    Crea variable objetivo en 4 clases usando cuartiles.
    """
    df1['impact_class'] = pd.qcut(df1['sef_with_margins'], q=4, labels=[0, 1, 2, 3])
    return df1


def add_dataset1_features(df1):
    """
    Agrega features derivados desde Dataset 1.
    """
    df1 = df1.copy()
    df1['sef_mef_ratio'] = df1['sef_without_margins'] / (df1['mef_margins'] + 1e-10)
    df1['naics_2dig'] = (df1['naics_code'] // 10000).astype(str)
    return df1