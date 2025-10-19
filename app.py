import hashlib
import io
import json
import os
from datetime import datetime
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula a distância do círculo máximo entre dois pontos na Terra.

    Utiliza a fórmula de Haversine para calcular a distância em quilômetros
    entre duas coordenadas geográficas (latitude e longitude).

    Args:
        lat1 (float): Latitude do primeiro ponto.
        lon1 (float): Longitude do primeiro ponto.
        lat2 (float): Latitude do segundo ponto.
        lon2 (float): Longitude do segundo ponto.

    Returns:
        float: A distância entre os dois pontos em quilômetros.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Raio da Terra em quilômetros
    return c * r


@st.cache_data(ttl=3600)  # Cache de 1 hora
def load_and_prepare_onix_data(file_content, filename):
    """Carrega e prepara os dados do arquivo ONIX a partir do conteúdo do upload.

    Lê um arquivo Excel (em bytes), padroniza os nomes dos motoristas, converte
    colunas de data, e adiciona identificadores para a fonte de dados.

    Args:
        file_content (bytes): O conteúdo binário do arquivo Excel carregado.
        filename (str): O nome do arquivo original (usado para cache e referência).

    Returns:
        pd.DataFrame: Um DataFrame do pandas com os dados do ONIX processados.
    """
    df = pd.read_excel(io.BytesIO(file_content))

    # Padroniza nomes de motoristas para correspondência
    df['MOTORISTA_NORMALIZED'] = df['MOTORISTA'].str.strip().str.upper()

    # Converte data para datetime se não estiver
    df['DATA'] = pd.to_datetime(df['DATA'])

    # Cria colunas separadas de data e hora
    df['DATA_SEPARADA'] = df['DATA'].dt.strftime('%d/%m/%Y')
    df['HORA'] = df['DATA'].dt.strftime('%H:%M:%S')

    # Renomeia colunas para consistência
    df = df.rename(columns={'LATITU': 'LATITUDE', 'LONGIT': 'LONGITUDE', 'VELOCI': 'VELOCIDADE'})

    # Adiciona identificador de origem
    df['ORIGEM'] = 'ONIX'

    return df


@st.cache_data(ttl=3600)  # Cache de 1 hora
def load_and_prepare_jornada_data(file_content, filename):
    """Carrega e prepara os dados do Relatório de Jornada a partir do conteúdo do upload.

    Lê um arquivo Excel (em bytes), padroniza nomes, converte datas e colunas
    de coordenadas, e adiciona um identificador de fonte de dados.

    Args:
        file_content (bytes): O conteúdo binário do arquivo Excel carregado.
        filename (str): O nome do arquivo original (usado para cache e referência).

    Returns:
        pd.DataFrame: Um DataFrame do pandas com os dados de Jornada processados.
    """
    df = pd.read_excel(io.BytesIO(file_content))

    # Padroniza nomes de motoristas para correspondência
    df['CONDUTOR_NORMALIZED'] = df['Condutor'].str.strip().str.upper()

    # Converte colunas de data para datetime
    df['Data Cadastro'] = pd.to_datetime(df['Data Cadastro'], format='%d/%m/%Y %H:%M:%S')
    df['Data Geração'] = pd.to_datetime(df['Data Geração'], format='%d/%m/%Y %H:%M:%S')

    # Cria colunas separadas de data e hora para ambas as datas
    df['DATA_SEPARADA_CADASTRO'] = df['Data Cadastro'].dt.strftime('%d/%m/%Y')
    df['HORA_CADASTRO'] = df['Data Cadastro'].dt.strftime('%H:%M:%S')
    df['DATA_SEPARADA_GERACAO'] = df['Data Geração'].dt.strftime('%d/%m/%Y')
    df['HORA_GERACAO'] = df['Data Geração'].dt.strftime('%H:%M:%S')

    # Converte latitude e longitude para numérico (trata separador decimal de vírgula)
    df['Latitude'] = df['Latitude'].astype(str).str.replace(',', '.').astype(float)
    df['Longitude'] = df['Longitude'].astype(str).str.replace(',', '.').astype(float)

    # Adiciona identificador de origem
    df['ORIGEM'] = 'ATS'

    return df


@st.cache_data(ttl=3600)
def calculate_working_hours(onix_df, jornada_df):
    """Calcula as horas de trabalho para cada motorista a partir de ambas as fontes.

    Itera sobre motoristas comuns em ambos os DataFrames para encontrar eventos de
    início e fim de jornada, calculando o total de horas trabalhadas em cada sistema
    e a diferença entre eles.

    Args:
        onix_df (pd.DataFrame): DataFrame com os dados processados do ONIX.
        jornada_df (pd.DataFrame): DataFrame com os dados processados do Jornada.

    Returns:
        pd.DataFrame: Um DataFrame de resumo contendo as horas de trabalho calculadas
                      e contagens de eventos para cada motorista.
    """
    # Define indicadores de início/fim de trabalho
    onix_work_starts = ['INICIO DE VIAGEM', 'INICIO DE VIAGEM CARREGADO', 'REINICIO DE VIAGEM']
    onix_work_ends = ['FIM DE VIAGEM', 'FIM DE JORNADA']
    jornada_work_starts = ['Inicio de jornada']
    jornada_work_ends = ['Fim de jornada']

    working_hours_summary = []

    # Encontra motoristas comuns
    onix_drivers = set(onix_df['MOTORISTA_NORMALIZED'])
    jornada_drivers = set(jornada_df['CONDUTOR_NORMALIZED'])
    common_drivers = onix_drivers.intersection(jornada_drivers)

    for driver in common_drivers:
        driver_summary = {
            'DRIVER': driver,
            'ONIX_WORK_STARTS': 0,
            'ONIX_WORK_ENDS': 0,
            'JORNADA_WORK_STARTS': 0,
            'JORNADA_WORK_ENDS': 0,
            'ONIX_TOTAL_HOURS': 0,
            'JORNADA_TOTAL_HOURS': 0,
            'HOURS_DIFFERENCE': 0,
        }

        # Cálculo de horas de trabalho do ONIX
        onix_driver_data = onix_df[onix_df['MOTORISTA_NORMALIZED'] == driver].copy()
        onix_starts = onix_driver_data[onix_driver_data['MACRO'].isin(onix_work_starts)].sort_values('DATA')
        onix_ends = onix_driver_data[onix_driver_data['MACRO'].isin(onix_work_ends)].sort_values('DATA')

        driver_summary['ONIX_WORK_STARTS'] = len(onix_starts)
        driver_summary['ONIX_WORK_ENDS'] = len(onix_ends)

        # Calcula o total de horas de trabalho do ONIX
        onix_total_hours = 0
        for _, start_row in onix_starts.iterrows():
            start_time = start_row['DATA']
            # Encontra o próximo fim de trabalho após este início
            next_ends = onix_ends[onix_ends['DATA'] > start_time]
            if len(next_ends) > 0:
                end_time = next_ends.iloc[0]['DATA']
                hours_diff = (end_time - start_time).total_seconds() / 3600
                onix_total_hours += hours_diff

        driver_summary['ONIX_TOTAL_HOURS'] = round(onix_total_hours, 2)

        # Cálculo de horas de trabalho do Jornada
        jornada_driver_data = jornada_df[jornada_df['CONDUTOR_NORMALIZED'] == driver].copy()
        jornada_starts = jornada_driver_data[jornada_driver_data['Status Gerado'].isin(jornada_work_starts)].sort_values('Data Cadastro')
        jornada_ends = jornada_driver_data[jornada_driver_data['Status Gerado'].isin(jornada_work_ends)].sort_values('Data Cadastro')

        driver_summary['JORNADA_WORK_STARTS'] = len(jornada_starts)
        driver_summary['JORNADA_WORK_ENDS'] = len(jornada_ends)

        # Calcula o total de horas de trabalho do Jornada
        jornada_total_hours = 0
        for _, start_row in jornada_starts.iterrows():
            start_time = start_row['Data Cadastro']
            # Encontra o próximo fim de trabalho após este início
            next_ends = jornada_ends[jornada_ends['Data Cadastro'] > start_time]
            if len(next_ends) > 0:
                end_time = next_ends.iloc[0]['Data Cadastro']
                hours_diff = (end_time - start_time).total_seconds() / 3600
                jornada_total_hours += hours_diff

        driver_summary['JORNADA_TOTAL_HOURS'] = round(jornada_total_hours, 2)

        # Calcula a diferença
        driver_summary['HOURS_DIFFERENCE'] = round(driver_summary['ONIX_TOTAL_HOURS'] - driver_summary['JORNADA_TOTAL_HOURS'], 2)

        working_hours_summary.append(driver_summary)

    return pd.DataFrame(working_hours_summary)


@st.cache_data(ttl=3600)
def merge_data(onix_df, jornada_df):
    """Mescla os dados do ONIX e do Jornada com base nos nomes dos motoristas e datas.

    Para cada registro do ONIX, esta função procura o registro correspondente mais
    próximo no tempo no DataFrame do Jornada (dentro do mesmo dia) e os combina
    em um único registro.

    Args:
        onix_df (pd.DataFrame): DataFrame com os dados processados do ONIX.
        jornada_df (pd.DataFrame): DataFrame com os dados processados do Jornada.

    Returns:
        pd.DataFrame: Um DataFrame mesclado contendo dados de ambas as fontes.
    """
    # Encontra motoristas comuns
    onix_drivers = set(onix_df['MOTORISTA_NORMALIZED'])
    jornada_drivers = set(jornada_df['CONDUTOR_NORMALIZED'])
    common_drivers = onix_drivers.intersection(jornada_drivers)

    merged_records = []

    for driver in common_drivers:
        # Obtém dados para este motorista de ambas as fontes
        onix_driver_data = onix_df[onix_df['MOTORISTA_NORMALIZED'] == driver].copy()
        jornada_driver_data = jornada_df[jornada_df['CONDUTOR_NORMALIZED'] == driver].copy()

        # Para cada registro do ONIX, encontra o registro mais próximo do Jornada por tempo
        for _, onix_row in onix_driver_data.iterrows():
            onix_time = onix_row['DATA']

            # Encontra o registro mais próximo do Jornada por tempo (no mesmo dia)
            jornada_same_day = jornada_driver_data[jornada_driver_data['Data Cadastro'].dt.date == onix_time.date()]

            if len(jornada_same_day) > 0:
                # Encontra o registro mais próximo pela diferença de tempo
                time_diffs = abs((jornada_same_day['Data Cadastro'] - onix_time).dt.total_seconds())
                closest_idx = time_diffs.idxmin()
                closest_jornada = jornada_same_day.loc[closest_idx]

                # Cria o registro mesclado
                merged_record = {
                    # Dados do ONIX
                    'MOTORISTA': onix_row['MOTORISTA'],
                    'DATA_SEPARADA': onix_row['DATA_SEPARADA'],
                    'HORA': onix_row['HORA'],
                    'CIDADE': onix_row['CIDADE'],
                    'LATITUDE': onix_row['LATITUDE'],
                    'LONGITUDE': onix_row['LONGITUDE'],
                    'MACRO': onix_row['MACRO'],
                    'ORIGEM': onix_row['ORIGEM'],
                    # Dados do Jornada
                    'CONDUTOR': closest_jornada['Condutor'],
                    'DATA_SEPARADA.1': closest_jornada['DATA_SEPARADA_CADASTRO'],
                    'HORA.1': closest_jornada['HORA_CADASTRO'],
                    'STATUS GERADO': closest_jornada['Status Gerado'],
                    'LOCALIZAÇÃO': closest_jornada['Localização'],
                    'LATITUDE.1': closest_jornada['Latitude'],
                    'LONGITUDE.1': closest_jornada['Longitude'],
                    'ORIGEM.1': closest_jornada['ORIGEM'],
                }
            else:
                # Nenhum registro correspondente do Jornada para este dia, usa apenas dados do ONIX
                merged_record = {
                    # Dados do ONIX
                    'MOTORISTA': onix_row['MOTORISTA'],
                    'DATA_SEPARADA': onix_row['DATA_SEPARADA'],
                    'HORA': onix_row['HORA'],
                    'CIDADE': onix_row['CIDADE'],
                    'LATITUDE': onix_row['LATITUDE'],
                    'LONGITUDE': onix_row['LONGITUDE'],
                    'MACRO': onix_row['MACRO'],
                    'ORIGEM': onix_row['ORIGEM'],
                    # Dados vazios do Jornada
                    'CONDUTOR': None,
                    'DATA_SEPARADA.1': None,
                    'HORA.1': None,
                    'STATUS GERADO': None,
                    'LOCALIZAÇÃO': None,
                    'LATITUDE.1': None,
                    'LONGITUDE.1': None,
                    'ORIGEM.1': None,
                }

            merged_records.append(merged_record)

    # Converte para DataFrame
    merged_df = pd.DataFrame(merged_records)

    return merged_df


@st.cache_data(ttl=3600)
def calculate_coordinate_analysis(consolidado_df):
    """Calcula as diferenças de coordenadas e estatísticas para cada motorista.

    Filtra registros que possuem ambos os conjuntos de coordenadas (ONIX e Jornada)
    e calcula a distância Haversine, bem como as diferenças absolutas de latitude
    e longitude. Agrupa os resultados por motorista.

    Args:
        consolidado_df (pd.DataFrame): O DataFrame mesclado contendo ambos os conjuntos de coordenadas.

    Returns:
        pd.DataFrame: Um DataFrame de resumo com estatísticas de coordenadas por motorista.
    """
    # Filtra registros com ambos os conjuntos de coordenadas
    coord_data = consolidado_df[
        consolidado_df['LATITUDE'].notna()
        & consolidado_df['LONGITUDE'].notna()
        & consolidado_df['LATITUDE.1'].notna()
        & consolidado_df['LONGITUDE.1'].notna()
    ].copy()

    # Calcula estatísticas de coordenadas por motorista
    driver_coord_stats = []

    for driver in coord_data['MOTORISTA'].unique():
        driver_data = coord_data[coord_data['MOTORISTA'] == driver]

        distances = []
        lat_diffs = []
        lon_diffs = []

        for _, row in driver_data.iterrows():
            # Calcula a distância
            dist = haversine_distance(row['LATITUDE'], row['LONGITUDE'], row['LATITUDE.1'], row['LONGITUDE.1'])
            distances.append(dist)

            # Calcula as diferenças de coordenadas
            lat_diff = abs(row['LATITUDE'] - row['LATITUDE.1'])
            lon_diff = abs(row['LONGITUDE'] - row['LONGITUDE.1'])
            lat_diffs.append(lat_diff)
            lon_diffs.append(lon_diff)

        driver_coord_stats.append(
            {
                'DRIVER': driver,
                'RECORDS_WITH_COORDS': len(driver_data),
                'AVG_DISTANCE_KM': round(np.mean(distances), 3),
                'MAX_DISTANCE_KM': round(np.max(distances), 3),
                'MIN_DISTANCE_KM': round(np.min(distances), 3),
                'STD_DISTANCE_KM': round(np.std(distances), 3),
                'AVG_LAT_DIFF': round(np.mean(lat_diffs), 6),
                'AVG_LON_DIFF': round(np.mean(lon_diffs), 6),
                'MAX_LAT_DIFF': round(np.max(lat_diffs), 6),
                'MAX_LON_DIFF': round(np.max(lon_diffs), 6),
            }
        )

    return pd.DataFrame(driver_coord_stats)


def save_analysis(analysis_name, onix_df, jornada_df, working_hours_df, coord_df, merged_df):
    """Salva os dados da análise no estado da sessão do Streamlit.

    Armazena todos os DataFrames processados e metadados associados a uma análise
    específica no `st.session_state` para acesso posterior.

    Args:
        analysis_name (str): O nome da análise a ser salva.
        onix_df (pd.DataFrame): DataFrame de dados brutos do ONIX.
        jornada_df (pd.DataFrame): DataFrame de dados brutos do Jornada.
        working_hours_df (pd.DataFrame): DataFrame com o resumo das horas de trabalho.
        coord_df (pd.DataFrame): DataFrame com a análise de coordenadas.
        merged_df (pd.DataFrame): DataFrame com os dados mesclados.

    Returns:
        dict: Um dicionário contendo os metadados da análise salva.
    """
    analysis_data = {
        'name': analysis_name,
        'created_at': datetime.now().isoformat(),
        'onix_records': len(onix_df),
        'jornada_records': len(jornada_df),
        'working_hours_drivers': len(working_hours_df),
        'coordinate_drivers': len(coord_df),
        'merged_records': len(merged_df),
    }

    # Salva no estado da sessão
    if 'saved_analyses' not in st.session_state:
        st.session_state.saved_analyses = {}

    st.session_state.saved_analyses[analysis_name] = {
        'metadata': analysis_data,
        'working_hours_df': working_hours_df,
        'coord_df': coord_df,
        'merged_df': merged_df,
        'onix_df': onix_df,
        'jornada_df': jornada_df,
    }

    return analysis_data


def load_analysis(analysis_name):
    """Carrega os dados de uma análise a partir do estado da sessão.

    Args:
        analysis_name (str): O nome da análise a ser carregada.

    Returns:
        dict | None: Um dicionário contendo os dados da análise se encontrada,
                      caso contrário, None.
    """
    if 'saved_analyses' in st.session_state and analysis_name in st.session_state.saved_analyses:
        return st.session_state.saved_analyses[analysis_name]
    return None


def get_analysis_list():
    """Obtém a lista de nomes de análises salvas.

    Returns:
        list: Uma lista de strings contendo os nomes de todas as análises
              atualmente no estado da sessão.
    """
    if 'saved_analyses' in st.session_state:
        return list(st.session_state.saved_analyses.keys())
    return []


def create_summary_metrics(df):
    """Cria e exibe os cartões de métricas de resumo no Streamlit.

    Args:
        df (pd.DataFrame): O DataFrame de resumo das horas de trabalho.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total de Motoristas", value=len(df), delta=None)

    with col2:
        avg_onix = df['ONIX_TOTAL_HOURS'].mean()
        st.metric(label="Média Horas ONIX", value=f"{avg_onix:.1f}", delta=None)

    with col3:
        avg_jornada = df['JORNADA_TOTAL_HOURS'].mean()
        st.metric(label="Média Horas Jornada", value=f"{avg_jornada:.1f}", delta=None)

    with col4:
        avg_diff = df['HOURS_DIFFERENCE'].mean()
        st.metric(
            label="Média da Diferença", value=f"{avg_diff:.1f}", delta=f"{avg_diff:.1f} horas" if avg_diff > 0 else f"{avg_diff:.1f} horas"
        )


def create_hours_comparison_chart(df):
    """Cria um gráfico de dispersão comparando as horas do ONIX vs. Jornada.

    Args:
        df (pd.DataFrame): O DataFrame de resumo das horas de trabalho.

    Returns:
        go.Figure: Um objeto de figura do Plotly contendo o gráfico de dispersão.
    """
    fig = px.scatter(
        df,
        x='JORNADA_TOTAL_HOURS',
        y='ONIX_TOTAL_HOURS',
        hover_data=['DRIVER', 'HOURS_DIFFERENCE'],
        title="Comparação de Horas de Trabalho: ONIX vs. Jornada",
        labels={'JORNADA_TOTAL_HOURS': 'Horas Jornada', 'ONIX_TOTAL_HOURS': 'Horas ONIX'},
    )

    # Adiciona linha diagonal para correlação perfeita
    max_val = max(df['ONIX_TOTAL_HOURS'].max(), df['JORNADA_TOTAL_HOURS'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='red'), name='Correlação Perfeita', showlegend=True
        )
    )

    fig.update_layout(height=500, showlegend=True)

    return fig


def create_difference_distribution(df):
    """Cria um histograma da distribuição das diferenças de horas.

    Args:
        df (pd.DataFrame): O DataFrame de resumo das horas de trabalho.

    Returns:
        go.Figure: Um objeto de figura do Plotly contendo o histograma.
    """
    fig = px.histogram(
        df,
        x='HOURS_DIFFERENCE',
        nbins=20,
        title="Distribuição das Diferenças de Horas (ONIX - Jornada)",
        labels={'HOURS_DIFFERENCE': 'Diferença de Horas', 'count': 'Número de Motoristas'},
    )

    # Adiciona linha vertical no zero
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Diferença Zero")

    fig.update_layout(height=400)

    return fig


def create_top_drivers_chart(df, top_n=10):
    """Cria um gráfico de barras dos principais motoristas por diferença de horas.

    Args:
        df (pd.DataFrame): O DataFrame de resumo das horas de trabalho.
        top_n (int): O número de principais motoristas a serem exibidos.

    Returns:
        go.Figure: Um objeto de figura do Plotly contendo o gráfico de barras.
    """
    # Ordena pela diferença absoluta
    df_sorted = df.sort_values('HOURS_DIFFERENCE', key=abs, ascending=False).head(top_n)

    fig = go.Figure()

    # Adiciona barras para diferenças positivas
    positive_df = df_sorted[df_sorted['HOURS_DIFFERENCE'] > 0]
    if len(positive_df) > 0:
        fig.add_trace(
            go.Bar(
                x=positive_df['DRIVER'],
                y=positive_df['HOURS_DIFFERENCE'],
                name='ONIX > Jornada',
                marker_color='green',
                text=positive_df['HOURS_DIFFERENCE'].round(1),
                textposition='auto',
            )
        )

    # Adiciona barras para diferenças negativas
    negative_df = df_sorted[df_sorted['HOURS_DIFFERENCE'] < 0]
    if len(negative_df) > 0:
        fig.add_trace(
            go.Bar(
                x=negative_df['DRIVER'],
                y=negative_df['HOURS_DIFFERENCE'],
                name='ONIX < Jornada',
                marker_color='red',
                text=negative_df['HOURS_DIFFERENCE'].round(1),
                textposition='auto',
            )
        )

    fig.update_layout(
        title=f"Top {top_n} Motoristas por Diferença de Horas",
        xaxis_title="Motorista",
        yaxis_title="Diferença de Horas (ONIX - Jornada)",
        height=500,
        xaxis_tickangle=-45,
    )

    return fig


def create_work_sessions_chart(df):
    """Cria um gráfico comparando as sessões de trabalho entre os sistemas.

    Args:
        df (pd.DataFrame): O DataFrame de resumo das horas de trabalho.

    Returns:
        go.Figure: Um objeto de figura do Plotly contendo os subplots.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=('Inícios de Trabalho', 'Fins de Trabalho', 'Total de Horas', 'Diferenças de Horas'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
    )

    # Comparação de inícios de trabalho
    fig.add_trace(
        go.Scatter(x=df['DRIVER'], y=df['ONIX_WORK_STARTS'], mode='markers', name='Inícios ONIX', marker=dict(color='blue')), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['DRIVER'], y=df['JORNADA_WORK_STARTS'], mode='markers', name='Inícios Jornada', marker=dict(color='orange')),
        row=1,
        col=1,
    )

    # Comparação de fins de trabalho
    fig.add_trace(
        go.Scatter(x=df['DRIVER'], y=df['ONIX_WORK_ENDS'], mode='markers', name='Fins ONIX', marker=dict(color='blue'), showlegend=False),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=df['DRIVER'], y=df['JORNADA_WORK_ENDS'], mode='markers', name='Fins Jornada', marker=dict(color='orange'), showlegend=False
        ),
        row=1,
        col=2,
    )

    # Comparação de total de horas
    fig.add_trace(
        go.Scatter(
            x=df['DRIVER'], y=df['ONIX_TOTAL_HOURS'], mode='markers', name='Horas ONIX', marker=dict(color='blue'), showlegend=False
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df['DRIVER'], y=df['JORNADA_TOTAL_HOURS'], mode='markers', name='Horas Jornada', marker=dict(color='orange'), showlegend=False
        ),
        row=2,
        col=1,
    )

    # Diferenças de horas
    colors = ['green' if x > 0 else 'red' for x in df['HOURS_DIFFERENCE']]
    fig.add_trace(
        go.Bar(x=df['DRIVER'], y=df['HOURS_DIFFERENCE'], name='Diferença', marker=dict(color=colors), showlegend=False), row=2, col=2
    )

    fig.update_layout(height=800, title_text="Análise de Sessões de Trabalho", showlegend=True)

    # Atualiza os rótulos do eixo x para todos os subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(tickangle=-45, row=i, col=j)

    return fig


def create_coordinate_distance_chart(coord_df):
    """Cria um gráfico mostrando as distâncias de coordenadas entre os sistemas.

    Args:
        coord_df (pd.DataFrame): DataFrame com a análise de coordenadas.

    Returns:
        go.Figure: Um objeto de figura do Plotly contendo o gráfico de barras.
    """
    fig = px.bar(
        coord_df.sort_values('AVG_DISTANCE_KM', ascending=False).head(15),
        x='DRIVER',
        y='AVG_DISTANCE_KM',
        title="Top 15 Motoristas por Distância Média de Coordenadas (ONIX vs Jornada)",
        labels={'AVG_DISTANCE_KM': 'Distância Média (km)', 'DRIVER': 'Motorista'},
        color='AVG_DISTANCE_KM',
        color_continuous_scale='Reds',
    )

    fig.update_layout(height=500, xaxis_tickangle=-45, coloraxis_colorbar=dict(title="Distância (km)"))

    return fig


def create_coordinate_scatter_plot(coord_df):
    """Cria um gráfico de dispersão mostrando a precisão das coordenadas vs. horas de trabalho.

    Args:
        coord_df (pd.DataFrame): DataFrame com a análise de coordenadas.

    Returns:
        go.Figure: Um objeto de figura do Plotly contendo o gráfico de dispersão.
    """
    fig = px.scatter(
        coord_df,
        x='AVG_DISTANCE_KM',
        y='RECORDS_WITH_COORDS',
        size='MAX_DISTANCE_KM',
        color='AVG_DISTANCE_KM',
        hover_data=['DRIVER', 'AVG_DISTANCE_KM', 'MAX_DISTANCE_KM', 'STD_DISTANCE_KM'],
        title="Cobertura de Dados de Coordenadas vs. Precisão da Distância",
        labels={
            'AVG_DISTANCE_KM': 'Distância Média (km)',
            'RECORDS_WITH_COORDS': 'Registros com Coordenadas',
            'MAX_DISTANCE_KM': 'Distância Máxima (km)',
        },
    )

    fig.update_layout(height=500)

    return fig


def create_coordinate_distribution_chart(coord_df):
    """Cria um histograma das distâncias de coordenadas.

    Args:
        coord_df (pd.DataFrame): DataFrame com a análise de coordenadas.

    Returns:
        go.Figure: Um objeto de figura do Plotly contendo o histograma.
    """
    fig = px.histogram(
        coord_df,
        x='AVG_DISTANCE_KM',
        nbins=20,
        title="Distribuição das Distâncias Médias de Coordenadas",
        labels={'AVG_DISTANCE_KM': 'Distância Média (km)', 'count': 'Número de Motoristas'},
    )

    fig.update_layout(height=400)

    return fig


def create_path_timeline_map(consolidado_df, driver_name, max_points=200):
    """Cria um mapa mostrando a linha do tempo cronológica do trajeto para um motorista.

    Args:
        consolidado_df (pd.DataFrame): O DataFrame mesclado.
        driver_name (str): O nome do motorista a ser visualizado.
        max_points (int): O número máximo de pontos a serem exibidos no mapa para evitar lentidão.

    Returns:
        go.Figure | None: Um objeto de figura do Plotly com o mapa, ou None se não houver
                           dados de coordenadas para o motorista.
    """
    driver_data = consolidado_df[
        (consolidado_df['MOTORISTA'] == driver_name)
        & consolidado_df['LATITUDE'].notna()
        & consolidado_df['LONGITUDE'].notna()
        & consolidado_df['LATITUDE.1'].notna()
        & consolidado_df['LONGITUDE.1'].notna()
    ].copy()

    if len(driver_data) == 0:
        return None

    # Converte strings de tempo para datetime para ordenação adequada
    driver_data['DATETIME_ONIX'] = pd.to_datetime(driver_data['DATA_SEPARADA'] + ' ' + driver_data['HORA'], format='%d/%m/%Y %H:%M:%S')

    # Ordena por datetime do ONIX
    driver_data = driver_data.sort_values('DATETIME_ONIX')

    # Amostra os dados se houver muitos pontos
    if len(driver_data) > max_points:
        step = len(driver_data) // max_points
        driver_data = driver_data.iloc[::step]

    # Calcula as distâncias
    distances = []
    for _, row in driver_data.iterrows():
        dist = haversine_distance(row['LATITUDE'], row['LONGITUDE'], row['LATITUDE.1'], row['LONGITUDE.1'])
        distances.append(dist)

    driver_data['DISTANCE_KM'] = distances

    # Cria escala de cores baseada no tempo
    min_time = driver_data['DATETIME_ONIX'].min()
    max_time = driver_data['DATETIME_ONIX'].max()
    time_range = (max_time - min_time).total_seconds()

    # Normaliza o tempo para a escala de cores (0 a 1)
    driver_data['TIME_NORMALIZED'] = (driver_data['DATETIME_ONIX'] - min_time).dt.total_seconds() / time_range if time_range > 0 else 0

    fig = go.Figure()

    # Adiciona trajeto do ONIX com cores baseadas no tempo
    fig.add_trace(
        go.Scattermapbox(
            lat=driver_data['LATITUDE'],
            lon=driver_data['LONGITUDE'],
            mode='markers+lines',
            marker=dict(
                size=6,
                color=driver_data['TIME_NORMALIZED'],
                colorscale='Viridis',
                colorbar=dict(
                    title="Progresso do Tempo",
                    tickmode='array',
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    ticktext=[
                        min_time.strftime('%d/%m %H:%M'),
                        (min_time + pd.Timedelta(seconds=time_range * 0.25)).strftime('%d/%m %H:%M'),
                        (min_time + pd.Timedelta(seconds=time_range * 0.5)).strftime('%d/%m %H:%M'),
                        (min_time + pd.Timedelta(seconds=time_range * 0.75)).strftime('%d/%m %H:%M'),
                        max_time.strftime('%d/%m %H:%M'),
                    ],
                ),
                showscale=True,
            ),
            line=dict(width=2, color='blue'),
            name='Trajeto ONIX',
            text=driver_data['DATA_SEPARADA'] + ' ' + driver_data['HORA'],
            hovertemplate='<b>Trajeto ONIX</b><br>Tempo: %{text}<br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>',
        )
    )

    # Adiciona trajeto do Jornada com cores baseadas no tempo
    fig.add_trace(
        go.Scattermapbox(
            lat=driver_data['LATITUDE.1'],
            lon=driver_data['LONGITUDE.1'],
            mode='markers+lines',
            marker=dict(
                size=6,
                color=driver_data['TIME_NORMALIZED'],
                colorscale='Plasma',
                colorbar=dict(
                    title="Progresso do Tempo",
                    tickmode='array',
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    ticktext=[
                        min_time.strftime('%d/%m %H:%M'),
                        (min_time + pd.Timedelta(seconds=time_range * 0.25)).strftime('%d/%m %H:%M'),
                        (min_time + pd.Timedelta(seconds=time_range * 0.5)).strftime('%d/%m %H:%M'),
                        (min_time + pd.Timedelta(seconds=time_range * 0.75)).strftime('%d/%m %H:%M'),
                        max_time.strftime('%d/%m %H:%M'),
                    ],
                ),
                showscale=True,
            ),
            line=dict(width=2, color='red'),
            name='Trajeto Jornada',
            text=driver_data['DATA_SEPARADA.1'] + ' ' + driver_data['HORA.1'],
            hovertemplate='<b>Trajeto Jornada</b><br>Tempo: %{text}<br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>',
        )
    )

    # Adiciona linhas de conexão entre os pontos ONIX e Jornada
    for _, row in driver_data.iterrows():
        fig.add_trace(
            go.Scattermapbox(
                lat=[row['LATITUDE'], row['LATITUDE.1']],
                lon=[row['LONGITUDE'], row['LONGITUDE.1']],
                mode='lines',
                line=dict(width=1, color='gray'),
                showlegend=False,
                hoverinfo='skip',
            )
        )

    # Calcula o ponto central
    all_lats = list(driver_data['LATITUDE']) + list(driver_data['LATITUDE.1'])
    all_lons = list(driver_data['LONGITUDE']) + list(driver_data['LONGITUDE.1'])

    fig.update_layout(
        title=f"Mapa de Linha do Tempo do Trajeto - {driver_name}<br><sub>Azul: Trajeto ONIX | Vermelho: Trajeto Jornada | Cinza: Conexões</sub>",
        mapbox=dict(style="open-street-map", center=dict(lat=np.mean(all_lats), lon=np.mean(all_lons)), zoom=11),
        height=700,
        showlegend=True,
    )

    return fig


def main():
    """Função principal que executa a aplicação Streamlit.

    Configura a página, gerencia o estado da sessão, o upload de arquivos,
    a seleção de análises e renderiza a interface do usuário com abas
    para diferentes visualizações de dados.
    """
    st.set_page_config(page_title="Dashboard de Análise de Horas e Coordenadas", page_icon="⏰", layout="wide")

    st.title("⏰ Dashboard de Análise de Horas e Coordenadas")
    st.markdown("Compare horas de trabalho e precisão de coordenadas entre os sistemas ONIX e Jornada")

    # Barra lateral para gerenciamento de análises
    st.sidebar.header("📁 Gerenciamento de Análises")

    # Seção de upload de arquivos
    st.sidebar.subheader("📤 Carregar Novos Arquivos")

    uploaded_onix = st.sidebar.file_uploader("Carregar Arquivo ONIX (Excel)", type=['xlsx', 'xls'], key="onix_upload")

    uploaded_jornada = st.sidebar.file_uploader("Carregar Arquivo Jornada (Excel)", type=['xlsx', 'xls'], key="jornada_upload")

    # Criação de análise
    if uploaded_onix and uploaded_jornada:
        st.sidebar.subheader("🔬 Criar Nova Análise")

        analysis_name = st.sidebar.text_input(
            "Nome da Análise", value=f"Analise_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="analysis_name"
        )

        if st.sidebar.button("🚀 Processar Arquivos", key="process_files"):
            with st.spinner("Processando arquivos..."):
                try:
                    # Carrega e prepara os dados
                    onix_df = load_and_prepare_onix_data(uploaded_onix.read(), uploaded_onix.name)
                    jornada_df = load_and_prepare_jornada_data(uploaded_jornada.read(), uploaded_jornada.name)

                    # Calcula as horas de trabalho
                    working_hours_df = calculate_working_hours(onix_df, jornada_df)

                    # Mescla os dados
                    merged_df = merge_data(onix_df, jornada_df)

                    # Calcula a análise de coordenadas
                    coord_df = calculate_coordinate_analysis(merged_df)

                    # Salva a análise
                    analysis_data = save_analysis(analysis_name, onix_df, jornada_df, working_hours_df, coord_df, merged_df)

                    st.sidebar.success(f"✅ Análise '{analysis_name}' criada com sucesso!")
                    st.sidebar.json(analysis_data)

                    # Define como análise atual
                    st.session_state.current_analysis = analysis_name

                except Exception as e:
                    st.sidebar.error(f"❌ Erro ao processar arquivos: {str(e)}")

    # Seleção de análise
    st.sidebar.subheader("📊 Selecionar Análise")

    analysis_list = get_analysis_list()
    if analysis_list:
        selected_analysis = st.sidebar.selectbox("Escolher Análise", analysis_list, key="analysis_select")

        if st.sidebar.button("📈 Carregar Análise", key="load_analysis"):
            st.session_state.current_analysis = selected_analysis

        # Comparação de análises
        if len(analysis_list) > 1:
            st.sidebar.subheader("🔄 Comparar Análises")
            compare_analysis = st.sidebar.selectbox(
                "Comparar com", [None] + [a for a in analysis_list if a != selected_analysis], key="compare_analysis"
            )

            if compare_analysis and st.sidebar.button("📊 Comparar", key="compare_button"):
                st.session_state.compare_analysis = compare_analysis
    else:
        st.sidebar.info("Nenhuma análise disponível. Carregue arquivos para criar uma.")

    # Área de conteúdo principal
    if 'current_analysis' in st.session_state:
        current_analysis = st.session_state.current_analysis
        analysis_data = load_analysis(current_analysis)

        if analysis_data:
            df = analysis_data['working_hours_df']
            coord_df = analysis_data['coord_df']
            consolidado_df = analysis_data['merged_df']

            # Cabeçalho da análise
            st.header(f"📊 Análise: {current_analysis}")

            # Metadados da análise
            metadata = analysis_data['metadata']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Registros ONIX", metadata['onix_records'])
            with col2:
                st.metric("Registros Jornada", metadata['jornada_records'])
            with col3:
                st.metric("Motoristas (Horas)", metadata['working_hours_drivers'])
            with col4:
                st.metric("Motoristas (Coord.)", metadata['coordinate_drivers'])

            # Cria abas para diferentes análises
            tab1, tab2, tab3 = st.tabs(["📊 Análise de Horas de Trabalho", "🗺️ Análise de Coordenadas", "🔍 Detalhes do Motorista"])

            with tab1:
                # Análise de Horas de Trabalho
                st.header("📊 Análise de Horas de Trabalho")

                # Métricas de resumo
                create_summary_metrics(df)

                # Gráficos principais
                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(create_hours_comparison_chart(df), use_container_width=True)

                with col2:
                    st.plotly_chart(create_difference_distribution(df), use_container_width=True)

                # Gráfico dos principais motoristas
                st.header("🏆 Top Motoristas por Diferença de Horas")
                top_n = st.slider("Número de motoristas para exibir", 5, 20, 10, key="top_n_hours")
                st.plotly_chart(create_top_drivers_chart(df, top_n), use_container_width=True)

                # Análise detalhada das sessões de trabalho
                st.header("🔍 Análise Detalhada das Sessões de Trabalho")
                st.plotly_chart(create_work_sessions_chart(df), use_container_width=True)

            with tab2:
                # Análise de Coordenadas
                st.header("🗺️ Análise de Coordenadas")

                # Métricas de resumo de coordenadas
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(label="Total de Motoristas", value=len(coord_df), delta=None)

                with col2:
                    avg_distance = coord_df['AVG_DISTANCE_KM'].mean()
                    st.metric(label="Distância Média", value=f"{avg_distance:.1f} km", delta=None)

                with col3:
                    max_distance = coord_df['AVG_DISTANCE_KM'].max()
                    st.metric(label="Distância Máxima", value=f"{max_distance:.1f} km", delta=None)

                with col4:
                    min_distance = coord_df['AVG_DISTANCE_KM'].min()
                    st.metric(label="Distância Mínima", value=f"{min_distance:.1f} km", delta=None)

                # Gráficos de coordenadas
                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(create_coordinate_distance_chart(coord_df), use_container_width=True)

                with col2:
                    st.plotly_chart(create_coordinate_scatter_plot(coord_df), use_container_width=True)

                # Gráfico de distribuição
                st.plotly_chart(create_coordinate_distribution_chart(coord_df), use_container_width=True)

                # Seção de Mapas de Linha do Tempo
                st.header("🛣️ Mapas de Linha do Tempo do Trajeto")

                # Seleção do tipo de mapa
                map_type = st.radio("Selecionar Tipo de Mapa", ["Linha do Tempo de Um Motorista", "Comparação de Múltiplos Motoristas"], key="map_type")

                if map_type == "Linha do Tempo de Um Motorista":
                    # Mapa de linha do tempo de um motorista
                    selected_driver_timeline = st.selectbox(
                        "Selecionar Motorista para Mapa de Linha do Tempo", sorted(coord_df['DRIVER'].tolist()), key="timeline_driver"
                    )

                    if selected_driver_timeline:
                        # Controles do mapa
                        col1, col2 = st.columns(2)
                        with col1:
                            max_points = st.slider(
                                "Máximo de Pontos a Exibir", min_value=50, max_value=500, value=200, key="max_points_timeline"
                            )

                        with col2:
                            show_connections = st.checkbox("Mostrar Linhas de Conexão", value=True, key="show_connections")

                        # Gera o mapa de linha do tempo
                        timeline_map = create_path_timeline_map(consolidado_df, selected_driver_timeline, max_points)
                        if timeline_map:
                            st.plotly_chart(timeline_map, use_container_width=True)

                            # Estatísticas da linha do tempo do motorista
                            driver_data = consolidado_df[
                                (consolidado_df['MOTORISTA'] == selected_driver_timeline)
                                & consolidado_df['LATITUDE'].notna()
                                & consolidado_df['LONGITUDE'].notna()
                                & consolidado_df['LATITUDE.1'].notna()
                                & consolidado_df['LONGITUDE.1'].notna()
                            ]

                            if len(driver_data) > 0:
                                # Converte para datetime para análise de tempo
                                driver_data['DATETIME_ONIX'] = pd.to_datetime(
                                    driver_data['DATA_SEPARADA'] + ' ' + driver_data['HORA'], format='%d/%m/%Y %H:%M:%S'
                                )

                                time_span = (driver_data['DATETIME_ONIX'].max() - driver_data['DATETIME_ONIX'].min()).total_seconds() / 3600

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total de Registros", len(driver_data))
                                with col2:
                                    st.metric("Período de Tempo", f"{time_span:.1f} horas")
                                with col3:
                                    avg_dist = coord_df[coord_df['DRIVER'] == selected_driver_timeline]['AVG_DISTANCE_KM'].iloc[0]
                                    st.metric("Distância Média", f"{avg_dist:.1f} km")
                        else:
                            st.warning("Nenhum dado de coordenada disponível para este motorista.")

                # Tabela de dados de análise de coordenadas
                st.header("📋 Dados da Análise de Coordenadas")
                st.dataframe(coord_df.sort_values('AVG_DISTANCE_KM', ascending=False), use_container_width=True, height=400)

            with tab3:
                # Detalhes do Motorista
                st.header("🔍 Análise Individual do Motorista")

                # Seleção do motorista
                selected_driver_detail = st.selectbox(
                    "Selecionar Motorista para Análise Detalhada", sorted(df['DRIVER'].tolist()), key="driver_detail"
                )

                if selected_driver_detail:
                    # Obtém dados do motorista
                    driver_hours = df[df['DRIVER'] == selected_driver_detail].iloc[0]
                    driver_coords = coord_df[coord_df['DRIVER'] == selected_driver_detail].iloc[0]

                    # Resumo do motorista
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Horas ONIX", f"{driver_hours['ONIX_TOTAL_HOURS']:.1f}")
                        st.metric("Horas Jornada", f"{driver_hours['JORNADA_TOTAL_HOURS']:.1f}")
                        st.metric("Diferença de Horas", f"{driver_hours['HOURS_DIFFERENCE']:.1f}")

                    with col2:
                        st.metric("Distância Média", f"{driver_coords['AVG_DISTANCE_KM']:.1f} km")
                        st.metric("Distância Máxima", f"{driver_coords['MAX_DISTANCE_KM']:.1f} km")
                        st.metric("Registros com Coord.", f"{driver_coords['RECORDS_WITH_COORDS']}")

                    with col3:
                        st.metric("Inícios de Trab. ONIX", f"{driver_hours['ONIX_WORK_STARTS']}")
                        st.metric("Fins de Trab. ONIX", f"{driver_hours['ONIX_WORK_ENDS']}")
                        st.metric("Inícios de Trab. Jornada", f"{driver_hours['JORNADA_WORK_STARTS']}")

                    # Mapa de coordenadas
                    st.header("🗺️ Mapa de Coordenadas")
                    coord_map = create_path_timeline_map(consolidado_df, selected_driver_detail)
                    if coord_map:
                        st.plotly_chart(coord_map, use_container_width=True)
                    else:
                        st.warning("Nenhum dado de coordenada disponível para este motorista.")

            # Comparação de análises
            if 'compare_analysis' in st.session_state:
                st.header("🔄 Comparação de Análises")
                compare_data = load_analysis(st.session_state.compare_analysis)

                if compare_data:
                    compare_df = compare_data['working_hours_df']
                    compare_coord_df = compare_data['coord_df']

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader(f"📊 {current_analysis}")
                        create_summary_metrics(df)

                    with col2:
                        st.subheader(f"📊 {st.session_state.compare_analysis}")
                        create_summary_metrics(compare_df)

                    # Gráficos de comparação
                    st.subheader("📈 Gráficos de Comparação")

                    # Mescla dados para comparação
                    comparison_df = pd.merge(
                        df[['DRIVER', 'ONIX_TOTAL_HOURS', 'JORNADA_TOTAL_HOURS', 'HOURS_DIFFERENCE']],
                        compare_df[['DRIVER', 'ONIX_TOTAL_HOURS', 'JORNADA_TOTAL_HOURS', 'HOURS_DIFFERENCE']],
                        on='DRIVER',
                        suffixes=('_Atual', '_Comparada'),
                    )

                    # Comparação de horas
                    fig = px.scatter(
                        comparison_df,
                        x='ONIX_TOTAL_HOURS_Atual',
                        y='ONIX_TOTAL_HOURS_Comparada',
                        hover_data=['DRIVER'],
                        title="Comparação de Horas ONIX Entre Análises",
                        labels={
                            'ONIX_TOTAL_HOURS_Atual': f'{current_analysis} Horas ONIX',
                            'ONIX_TOTAL_HOURS_Comparada': f'{st.session_state.compare_analysis} Horas ONIX',
                        },
                    )

                    # Adiciona linha diagonal
                    max_val = max(comparison_df['ONIX_TOTAL_HOURS_Atual'].max(), comparison_df['ONIX_TOTAL_HOURS_Comparada'].max())
                    fig.add_trace(
                        go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='red'), name='Correspondência Perfeita')
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Limpa comparação
                    if st.button("❌ Limpar Comparação"):
                        del st.session_state.compare_analysis
                        st.rerun()
        else:
            st.error("Dados da análise não encontrados.")
    else:
        # Tela de boas-vindas
        st.header("👋 Bem-vindo ao Dashboard de Análise")
        st.markdown(
            """
        ### Para Começar:
        1. **Carregue os Arquivos**: Use a barra lateral para carregar seus arquivos Excel do ONIX e do Jornada.
        2. **Crie uma Análise**: Dê um nome à sua análise e processe os arquivos.
        3. **Explore os Dados**: Use as abas para analisar horas de trabalho e coordenadas.
        4. **Compare Análises**: Crie múltiplas análises e compare-as lado a lado.
        
        ### Funcionalidades:
        - ⚡ **Processamento em Cache**: Análise rápida com cache inteligente.
        - 📊 **Análise de Horas de Trabalho**: Compare o rastreamento de tempo entre sistemas.
        - 🗺️ **Análise de Coordenadas**: Visualize a precisão do GPS e as linhas do tempo dos trajetos.
        - 🔄 **Comparação de Análises**: Compare múltiplas análises lado a lado.
        - 📈 **Gráficos Interativos**: Visualizações dinâmicas com Plotly.
        """
        )

        # Mostra análises salvas, se houver
        analysis_list = get_analysis_list()
        if analysis_list:
            st.subheader("📁 Análises Salvas")
            for analysis in analysis_list:
                analysis_data = load_analysis(analysis)
                if analysis_data:
                    metadata = analysis_data['metadata']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**{analysis}**")
                    with col2:
                        st.write(f"Criada em: {metadata['created_at'][:19]}")
                    with col3:
                        if st.button(f"Carregar {analysis}", key=f"load_{analysis}"):
                            st.session_state.current_analysis = analysis
                            st.rerun()

    # Rodapé
    st.markdown("---")
    st.markdown("**Fonte de Dados**: Sistema de rastreamento GPS ONIX e sistema de status de motorista Jornada")
    st.markdown("**Nota**: Distâncias de coordenadas calculadas usando a fórmula de Haversine")


if __name__ == "__main__":
    main()