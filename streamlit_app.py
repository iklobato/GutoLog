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
    """Calculate the great circle distance between two points on Earth in kilometers."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_prepare_onix_data(file_content, filename):
    """Load and prepare ONIX data for merging."""
    df = pd.read_excel(io.BytesIO(file_content))

    # Standardize driver names for matching
    df['MOTORISTA_NORMALIZED'] = df['MOTORISTA'].str.strip().str.upper()

    # Convert date to datetime if not already
    df['DATA'] = pd.to_datetime(df['DATA'])

    # Create separate date and time columns
    df['DATA_SEPARADA'] = df['DATA'].dt.strftime('%d/%m/%Y')
    df['HORA'] = df['DATA'].dt.strftime('%H:%M:%S')

    # Rename columns for consistency
    df = df.rename(columns={'LATITU': 'LATITUDE', 'LONGIT': 'LONGITUDE', 'VELOCI': 'VELOCIDADE'})

    # Add source identifier
    df['ORIGEM'] = 'ONIX'

    return df


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_prepare_jornada_data(file_content, filename):
    """Load and prepare Relatorio de Jornada data for merging."""
    df = pd.read_excel(io.BytesIO(file_content))

    # Standardize driver names for matching
    df['CONDUTOR_NORMALIZED'] = df['Condutor'].str.strip().str.upper()

    # Convert date columns to datetime
    df['Data Cadastro'] = pd.to_datetime(df['Data Cadastro'], format='%d/%m/%Y %H:%M:%S')
    df['Data Geração'] = pd.to_datetime(df['Data Geração'], format='%d/%m/%Y %H:%M:%S')

    # Create separate date and time columns for both dates
    df['DATA_SEPARADA_CADASTRO'] = df['Data Cadastro'].dt.strftime('%d/%m/%Y')
    df['HORA_CADASTRO'] = df['Data Cadastro'].dt.strftime('%H:%M:%S')
    df['DATA_SEPARADA_GERACAO'] = df['Data Geração'].dt.strftime('%d/%m/%Y')
    df['HORA_GERACAO'] = df['Data Geração'].dt.strftime('%H:%M:%S')

    # Convert latitude and longitude to numeric (handle comma decimal separator)
    df['Latitude'] = df['Latitude'].astype(str).str.replace(',', '.').astype(float)
    df['Longitude'] = df['Longitude'].astype(str).str.replace(',', '.').astype(float)

    # Add source identifier
    df['ORIGEM'] = 'ATS'

    return df


@st.cache_data(ttl=3600)
def calculate_working_hours(onix_df, jornada_df):
    """Calculate working hours for each driver from both sources."""
    # Define work start/end indicators
    onix_work_starts = ['INICIO DE VIAGEM', 'INICIO DE VIAGEM CARREGADO', 'REINICIO DE VIAGEM']
    onix_work_ends = ['FIM DE VIAGEM', 'FIM DE JORNADA']
    jornada_work_starts = ['Inicio de jornada']
    jornada_work_ends = ['Fim de jornada']

    working_hours_summary = []

    # Find common drivers
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

        # ONIX working hours calculation
        onix_driver_data = onix_df[onix_df['MOTORISTA_NORMALIZED'] == driver].copy()
        onix_starts = onix_driver_data[onix_driver_data['MACRO'].isin(onix_work_starts)].sort_values('DATA')
        onix_ends = onix_driver_data[onix_driver_data['MACRO'].isin(onix_work_ends)].sort_values('DATA')

        driver_summary['ONIX_WORK_STARTS'] = len(onix_starts)
        driver_summary['ONIX_WORK_ENDS'] = len(onix_ends)

        # Calculate ONIX total working hours
        onix_total_hours = 0
        for _, start_row in onix_starts.iterrows():
            start_time = start_row['DATA']
            # Find the next work end after this start
            next_ends = onix_ends[onix_ends['DATA'] > start_time]
            if len(next_ends) > 0:
                end_time = next_ends.iloc[0]['DATA']
                hours_diff = (end_time - start_time).total_seconds() / 3600
                onix_total_hours += hours_diff

        driver_summary['ONIX_TOTAL_HOURS'] = round(onix_total_hours, 2)

        # Jornada working hours calculation
        jornada_driver_data = jornada_df[jornada_df['CONDUTOR_NORMALIZED'] == driver].copy()
        jornada_starts = jornada_driver_data[jornada_driver_data['Status Gerado'].isin(jornada_work_starts)].sort_values('Data Cadastro')
        jornada_ends = jornada_driver_data[jornada_driver_data['Status Gerado'].isin(jornada_work_ends)].sort_values('Data Cadastro')

        driver_summary['JORNADA_WORK_STARTS'] = len(jornada_starts)
        driver_summary['JORNADA_WORK_ENDS'] = len(jornada_ends)

        # Calculate Jornada total working hours
        jornada_total_hours = 0
        for _, start_row in jornada_starts.iterrows():
            start_time = start_row['Data Cadastro']
            # Find the next work end after this start
            next_ends = jornada_ends[jornada_ends['Data Cadastro'] > start_time]
            if len(next_ends) > 0:
                end_time = next_ends.iloc[0]['Data Cadastro']
                hours_diff = (end_time - start_time).total_seconds() / 3600
                jornada_total_hours += hours_diff

        driver_summary['JORNADA_TOTAL_HOURS'] = round(jornada_total_hours, 2)

        # Calculate difference
        driver_summary['HOURS_DIFFERENCE'] = round(driver_summary['ONIX_TOTAL_HOURS'] - driver_summary['JORNADA_TOTAL_HOURS'], 2)

        working_hours_summary.append(driver_summary)

    return pd.DataFrame(working_hours_summary)


@st.cache_data(ttl=3600)
def merge_data(onix_df, jornada_df):
    """Merge ONIX and Jornada data based on driver names and dates."""
    # Find common drivers
    onix_drivers = set(onix_df['MOTORISTA_NORMALIZED'])
    jornada_drivers = set(jornada_df['CONDUTOR_NORMALIZED'])
    common_drivers = onix_drivers.intersection(jornada_drivers)

    merged_records = []

    for driver in common_drivers:
        # Get data for this driver from both sources
        onix_driver_data = onix_df[onix_df['MOTORISTA_NORMALIZED'] == driver].copy()
        jornada_driver_data = jornada_df[jornada_df['CONDUTOR_NORMALIZED'] == driver].copy()

        # For each ONIX record, find the closest Jornada record by time
        for _, onix_row in onix_driver_data.iterrows():
            onix_time = onix_row['DATA']

            # Find the closest Jornada record by time (within same day)
            jornada_same_day = jornada_driver_data[jornada_driver_data['Data Cadastro'].dt.date == onix_time.date()]

            if len(jornada_same_day) > 0:
                # Find the closest record by time difference
                time_diffs = abs((jornada_same_day['Data Cadastro'] - onix_time).dt.total_seconds())
                closest_idx = time_diffs.idxmin()
                closest_jornada = jornada_same_day.loc[closest_idx]

                # Create merged record
                merged_record = {
                    # ONIX data
                    'MOTORISTA': onix_row['MOTORISTA'],
                    'DATA_SEPARADA': onix_row['DATA_SEPARADA'],
                    'HORA': onix_row['HORA'],
                    'CIDADE': onix_row['CIDADE'],
                    'LATITUDE': onix_row['LATITUDE'],
                    'LONGITUDE': onix_row['LONGITUDE'],
                    'MACRO': onix_row['MACRO'],
                    'ORIGEM': onix_row['ORIGEM'],
                    # Jornada data
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
                # No matching Jornada record for this day, use ONIX data only
                merged_record = {
                    # ONIX data
                    'MOTORISTA': onix_row['MOTORISTA'],
                    'DATA_SEPARADA': onix_row['DATA_SEPARADA'],
                    'HORA': onix_row['HORA'],
                    'CIDADE': onix_row['CIDADE'],
                    'LATITUDE': onix_row['LATITUDE'],
                    'LONGITUDE': onix_row['LONGITUDE'],
                    'MACRO': onix_row['MACRO'],
                    'ORIGEM': onix_row['ORIGEM'],
                    # Empty Jornada data
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

    # Convert to DataFrame
    merged_df = pd.DataFrame(merged_records)

    return merged_df


@st.cache_data(ttl=3600)
def calculate_coordinate_analysis(consolidado_df):
    """Calculate coordinate differences and statistics for each driver."""
    # Filter records with both coordinate sets
    coord_data = consolidado_df[
        consolidado_df['LATITUDE'].notna()
        & consolidado_df['LONGITUDE'].notna()
        & consolidado_df['LATITUDE.1'].notna()
        & consolidado_df['LONGITUDE.1'].notna()
    ].copy()

    # Calculate coordinate statistics by driver
    driver_coord_stats = []

    for driver in coord_data['MOTORISTA'].unique():
        driver_data = coord_data[coord_data['MOTORISTA'] == driver]

        distances = []
        lat_diffs = []
        lon_diffs = []

        for _, row in driver_data.iterrows():
            # Calculate distance
            dist = haversine_distance(row['LATITUDE'], row['LONGITUDE'], row['LATITUDE.1'], row['LONGITUDE.1'])
            distances.append(dist)

            # Calculate coordinate differences
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
    """Save analysis data to session state and local storage."""
    analysis_data = {
        'name': analysis_name,
        'created_at': datetime.now().isoformat(),
        'onix_records': len(onix_df),
        'jornada_records': len(jornada_df),
        'working_hours_drivers': len(working_hours_df),
        'coordinate_drivers': len(coord_df),
        'merged_records': len(merged_df),
    }

    # Save to session state
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
    """Load analysis data from session state."""
    if 'saved_analyses' in st.session_state and analysis_name in st.session_state.saved_analyses:
        return st.session_state.saved_analyses[analysis_name]
    return None


def get_analysis_list():
    """Get list of saved analyses."""
    if 'saved_analyses' in st.session_state:
        return list(st.session_state.saved_analyses.keys())
    return []


def create_summary_metrics(df):
    """Create summary metrics cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Drivers", value=len(df), delta=None)

    with col2:
        avg_onix = df['ONIX_TOTAL_HOURS'].mean()
        st.metric(label="Avg ONIX Hours", value=f"{avg_onix:.1f}", delta=None)

    with col3:
        avg_jornada = df['JORNADA_TOTAL_HOURS'].mean()
        st.metric(label="Avg Jornada Hours", value=f"{avg_jornada:.1f}", delta=None)

    with col4:
        avg_diff = df['HOURS_DIFFERENCE'].mean()
        st.metric(
            label="Avg Difference", value=f"{avg_diff:.1f}", delta=f"{avg_diff:.1f} hours" if avg_diff > 0 else f"{avg_diff:.1f} hours"
        )


def create_hours_comparison_chart(df):
    """Create a scatter plot comparing ONIX vs Jornada hours."""
    fig = px.scatter(
        df,
        x='JORNADA_TOTAL_HOURS',
        y='ONIX_TOTAL_HOURS',
        hover_data=['DRIVER', 'HOURS_DIFFERENCE'],
        title="ONIX vs Jornada Working Hours Comparison",
        labels={'JORNADA_TOTAL_HOURS': 'Jornada Hours', 'ONIX_TOTAL_HOURS': 'ONIX Hours'},
    )

    # Add diagonal line for perfect correlation
    max_val = max(df['ONIX_TOTAL_HOURS'].max(), df['JORNADA_TOTAL_HOURS'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='red'), name='Perfect Correlation', showlegend=True
        )
    )

    fig.update_layout(height=500, showlegend=True)

    return fig


def create_difference_distribution(df):
    """Create a histogram of hour differences."""
    fig = px.histogram(
        df,
        x='HOURS_DIFFERENCE',
        nbins=20,
        title="Distribution of Hour Differences (ONIX - Jornada)",
        labels={'HOURS_DIFFERENCE': 'Hour Difference', 'count': 'Number of Drivers'},
    )

    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Difference")

    fig.update_layout(height=400)

    return fig


def create_top_drivers_chart(df, top_n=10):
    """Create a bar chart of top drivers by hour difference."""
    # Sort by absolute difference
    df_sorted = df.sort_values('HOURS_DIFFERENCE', key=abs, ascending=False).head(top_n)

    fig = go.Figure()

    # Add bars for positive differences
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

    # Add bars for negative differences
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
        title=f"Top {top_n} Drivers by Hour Difference",
        xaxis_title="Driver",
        yaxis_title="Hour Difference (ONIX - Jornada)",
        height=500,
        xaxis_tickangle=-45,
    )

    return fig


def create_work_sessions_chart(df):
    """Create a chart comparing work sessions between systems."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=('Work Starts', 'Work Ends', 'Total Hours', 'Hour Differences'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
    )

    # Work starts comparison
    fig.add_trace(
        go.Scatter(x=df['DRIVER'], y=df['ONIX_WORK_STARTS'], mode='markers', name='ONIX Starts', marker=dict(color='blue')), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['DRIVER'], y=df['JORNADA_WORK_STARTS'], mode='markers', name='Jornada Starts', marker=dict(color='orange')),
        row=1,
        col=1,
    )

    # Work ends comparison
    fig.add_trace(
        go.Scatter(x=df['DRIVER'], y=df['ONIX_WORK_ENDS'], mode='markers', name='ONIX Ends', marker=dict(color='blue'), showlegend=False),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=df['DRIVER'], y=df['JORNADA_WORK_ENDS'], mode='markers', name='Jornada Ends', marker=dict(color='orange'), showlegend=False
        ),
        row=1,
        col=2,
    )

    # Total hours comparison
    fig.add_trace(
        go.Scatter(
            x=df['DRIVER'], y=df['ONIX_TOTAL_HOURS'], mode='markers', name='ONIX Hours', marker=dict(color='blue'), showlegend=False
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df['DRIVER'], y=df['JORNADA_TOTAL_HOURS'], mode='markers', name='Jornada Hours', marker=dict(color='orange'), showlegend=False
        ),
        row=2,
        col=1,
    )

    # Hour differences
    colors = ['green' if x > 0 else 'red' for x in df['HOURS_DIFFERENCE']]
    fig.add_trace(
        go.Bar(x=df['DRIVER'], y=df['HOURS_DIFFERENCE'], name='Difference', marker=dict(color=colors), showlegend=False), row=2, col=2
    )

    fig.update_layout(height=800, title_text="Work Sessions Analysis", showlegend=True)

    # Update x-axis labels for all subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(tickangle=-45, row=i, col=j)

    return fig


def create_coordinate_distance_chart(coord_df):
    """Create a chart showing coordinate distances between systems."""
    fig = px.bar(
        coord_df.sort_values('AVG_DISTANCE_KM', ascending=False).head(15),
        x='DRIVER',
        y='AVG_DISTANCE_KM',
        title="Top 15 Drivers by Average Coordinate Distance (ONIX vs Jornada)",
        labels={'AVG_DISTANCE_KM': 'Average Distance (km)', 'DRIVER': 'Driver'},
        color='AVG_DISTANCE_KM',
        color_continuous_scale='Reds',
    )

    fig.update_layout(height=500, xaxis_tickangle=-45, coloraxis_colorbar=dict(title="Distance (km)"))

    return fig


def create_coordinate_scatter_plot(coord_df):
    """Create a scatter plot showing coordinate accuracy vs working hours."""
    fig = px.scatter(
        coord_df,
        x='AVG_DISTANCE_KM',
        y='RECORDS_WITH_COORDS',
        size='MAX_DISTANCE_KM',
        color='AVG_DISTANCE_KM',
        hover_data=['DRIVER', 'AVG_DISTANCE_KM', 'MAX_DISTANCE_KM', 'STD_DISTANCE_KM'],
        title="Coordinate Data Coverage vs Distance Accuracy",
        labels={
            'AVG_DISTANCE_KM': 'Average Distance (km)',
            'RECORDS_WITH_COORDS': 'Records with Coordinates',
            'MAX_DISTANCE_KM': 'Max Distance (km)',
        },
    )

    fig.update_layout(height=500)

    return fig


def create_coordinate_distribution_chart(coord_df):
    """Create a histogram of coordinate distances."""
    fig = px.histogram(
        coord_df,
        x='AVG_DISTANCE_KM',
        nbins=20,
        title="Distribution of Average Coordinate Distances",
        labels={'AVG_DISTANCE_KM': 'Average Distance (km)', 'count': 'Number of Drivers'},
    )

    fig.update_layout(height=400)

    return fig


def create_path_timeline_map(consolidado_df, driver_name, max_points=200):
    """Create a map showing chronological path timeline for a specific driver."""
    driver_data = consolidado_df[
        (consolidado_df['MOTORISTA'] == driver_name)
        & consolidado_df['LATITUDE'].notna()
        & consolidado_df['LONGITUDE'].notna()
        & consolidado_df['LATITUDE.1'].notna()
        & consolidado_df['LONGITUDE.1'].notna()
    ].copy()

    if len(driver_data) == 0:
        return None

    # Convert time strings to datetime for proper sorting
    driver_data = driver_data.copy()
    driver_data['DATETIME_ONIX'] = pd.to_datetime(driver_data['DATA_SEPARADA'] + ' ' + driver_data['HORA'], format='%d/%m/%Y %H:%M:%S')

    # Sort by ONIX datetime
    driver_data = driver_data.sort_values('DATETIME_ONIX')

    # Sample data if too many points
    if len(driver_data) > max_points:
        step = len(driver_data) // max_points
        driver_data = driver_data.iloc[::step]

    # Calculate distances
    distances = []
    for _, row in driver_data.iterrows():
        dist = haversine_distance(row['LATITUDE'], row['LONGITUDE'], row['LATITUDE.1'], row['LONGITUDE.1'])
        distances.append(dist)

    driver_data['DISTANCE_KM'] = distances

    # Create time-based color scale
    min_time = driver_data['DATETIME_ONIX'].min()
    max_time = driver_data['DATETIME_ONIX'].max()
    time_range = (max_time - min_time).total_seconds()

    # Normalize time for color scale (0 to 1)
    driver_data['TIME_NORMALIZED'] = (driver_data['DATETIME_ONIX'] - min_time).dt.total_seconds() / time_range if time_range > 0 else 0

    fig = go.Figure()

    # Add ONIX path with time-based colors
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
                    title="Time Progress",
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
            name='ONIX Path',
            text=driver_data['DATA_SEPARADA'] + ' ' + driver_data['HORA'],
            hovertemplate='<b>ONIX Path</b><br>Time: %{text}<br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>',
        )
    )

    # Add Jornada path with time-based colors
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
                    title="Time Progress",
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
            name='Jornada Path',
            text=driver_data['DATA_SEPARADA.1'] + ' ' + driver_data['HORA.1'],
            hovertemplate='<b>Jornada Path</b><br>Time: %{text}<br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>',
        )
    )

    # Add connection lines between ONIX and Jornada points
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

    # Calculate bounds and center point
    all_lats = list(driver_data['LATITUDE']) + list(driver_data['LATITUDE.1'])
    all_lons = list(driver_data['LONGITUDE']) + list(driver_data['LONGITUDE.1'])

    # Calculate bounds
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)

    # Calculate center
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Calculate appropriate zoom level based on bounds
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    max_range = max(lat_range, lon_range)

    # Adjust zoom based on range (empirical values)
    if max_range > 1.0:
        zoom_level = 8
    elif max_range > 0.5:
        zoom_level = 9
    elif max_range > 0.1:
        zoom_level = 10
    elif max_range > 0.05:
        zoom_level = 11
    elif max_range > 0.01:
        zoom_level = 12
    else:
        zoom_level = 13

    fig.update_layout(
        title=f"Path Timeline Map - {driver_name}<br><sub>Blue: ONIX Path | Red: Jornada Path | Gray: Connections</sub>",
        mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=zoom_level),
        height=700,
        showlegend=True,
    )

    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Working Hours & Coordinates Analysis Dashboard", page_icon="⏰", layout="wide")

    st.title("⏰ Working Hours & Coordinates Analysis Dashboard")
    st.markdown("Compare working hours and coordinate accuracy between ONIX and Jornada systems")

    # Sidebar for analysis management
    st.sidebar.header("📁 Analysis Management")

    # File upload section
    st.sidebar.subheader("📤 Upload New Files")

    uploaded_onix = st.sidebar.file_uploader("Upload ONIX Excel File", type=['xlsx', 'xls'], key="onix_upload")

    uploaded_jornada = st.sidebar.file_uploader("Upload Jornada Excel File", type=['xlsx', 'xls'], key="jornada_upload")

    # Analysis creation
    if uploaded_onix and uploaded_jornada:
        st.sidebar.subheader("🔬 Create New Analysis")

        analysis_name = st.sidebar.text_input(
            "Analysis Name", value=f"Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="analysis_name"
        )

        if st.sidebar.button("🚀 Process Files", key="process_files"):
            with st.spinner("Processing files..."):
                try:
                    # Load and prepare data
                    onix_df = load_and_prepare_onix_data(uploaded_onix.read(), uploaded_onix.name)
                    jornada_df = load_and_prepare_jornada_data(uploaded_jornada.read(), uploaded_jornada.name)

                    # Calculate working hours
                    working_hours_df = calculate_working_hours(onix_df, jornada_df)

                    # Merge data
                    merged_df = merge_data(onix_df, jornada_df)

                    # Calculate coordinate analysis
                    coord_df = calculate_coordinate_analysis(merged_df)

                    # Save analysis
                    analysis_data = save_analysis(analysis_name, onix_df, jornada_df, working_hours_df, coord_df, merged_df)

                    st.sidebar.success(f"✅ Analysis '{analysis_name}' created successfully!")
                    st.sidebar.json(analysis_data)

                    # Set as current analysis
                    st.session_state.current_analysis = analysis_name

                except Exception as e:
                    st.sidebar.error(f"❌ Error processing files: {str(e)}")

    # Analysis selection
    st.sidebar.subheader("📊 Select Analysis")

    analysis_list = get_analysis_list()
    if analysis_list:
        selected_analysis = st.sidebar.selectbox("Choose Analysis", analysis_list, key="analysis_select")

        if st.sidebar.button("📈 Load Analysis", key="load_analysis"):
            st.session_state.current_analysis = selected_analysis

        # Analysis comparison
        if len(analysis_list) > 1:
            st.sidebar.subheader("🔄 Compare Analyses")
            compare_analysis = st.sidebar.selectbox(
                "Compare with", [None] + [a for a in analysis_list if a != selected_analysis], key="compare_analysis"
            )

            if compare_analysis and st.sidebar.button("📊 Compare", key="compare_button"):
                st.session_state.compare_analysis = compare_analysis
    else:
        st.sidebar.info("No analyses available. Upload files to create one.")

    # Main content area
    if 'current_analysis' in st.session_state:
        current_analysis = st.session_state.current_analysis
        analysis_data = load_analysis(current_analysis)

        if analysis_data:
            df = analysis_data['working_hours_df']
            coord_df = analysis_data['coord_df']
            consolidado_df = analysis_data['merged_df']

            # Analysis header
            st.header(f"📊 Analysis: {current_analysis}")

            # Analysis metadata
            metadata = analysis_data['metadata']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ONIX Records", metadata['onix_records'])
            with col2:
                st.metric("Jornada Records", metadata['jornada_records'])
            with col3:
                st.metric("Working Hours Drivers", metadata['working_hours_drivers'])
            with col4:
                st.metric("Coordinate Drivers", metadata['coordinate_drivers'])

            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["📊 Working Hours Analysis", "🗺️ Coordinate Analysis", "🔍 Driver Details"])

            with tab1:
                # Working Hours Analysis
                st.header("📊 Working Hours Analysis")

                # Summary metrics
                create_summary_metrics(df)

                # Main charts
                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(create_hours_comparison_chart(df), use_container_width=True, key="hours_comparison")

                with col2:
                    st.plotly_chart(create_difference_distribution(df), use_container_width=True, key="difference_distribution")

                # Top drivers chart
                st.header("🏆 Top Drivers by Hour Difference")
                top_n = st.slider("Number of top drivers to show", 5, 20, 10, key="top_n_hours")
                st.plotly_chart(create_top_drivers_chart(df, top_n), use_container_width=True, key="top_drivers_chart")

                # Detailed work sessions analysis
                st.header("🔍 Detailed Work Sessions Analysis")
                st.plotly_chart(create_work_sessions_chart(df), use_container_width=True, key="work_sessions_chart")

            with tab2:
                # Coordinate Analysis
                st.header("🗺️ Coordinate Analysis")

                # Coordinate summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(label="Total Drivers", value=len(coord_df), delta=None)

                with col2:
                    avg_distance = coord_df['AVG_DISTANCE_KM'].mean()
                    st.metric(label="Avg Distance", value=f"{avg_distance:.1f} km", delta=None)

                with col3:
                    max_distance = coord_df['AVG_DISTANCE_KM'].max()
                    st.metric(label="Max Distance", value=f"{max_distance:.1f} km", delta=None)

                with col4:
                    min_distance = coord_df['AVG_DISTANCE_KM'].min()
                    st.metric(label="Min Distance", value=f"{min_distance:.1f} km", delta=None)

                # Coordinate charts
                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(create_coordinate_distance_chart(coord_df), use_container_width=True, key="coord_distance_chart")

                with col2:
                    st.plotly_chart(create_coordinate_scatter_plot(coord_df), use_container_width=True, key="coord_scatter_chart")

                # Distribution chart
                st.plotly_chart(create_coordinate_distribution_chart(coord_df), use_container_width=True, key="coord_distribution_chart")

                # Path Timeline Maps Section
                st.header("🛣️ Path Timeline Maps")

                # Map type selection
                map_type = st.radio("Select Map Type", ["Single Driver Timeline", "Multi-Driver Comparison"], key="map_type")

                if map_type == "Single Driver Timeline":
                    # Single driver timeline map
                    selected_driver_timeline = st.selectbox(
                        "Select Driver for Timeline Map", sorted(coord_df['DRIVER'].tolist()), key="timeline_driver"
                    )

                    if selected_driver_timeline:
                        # Map controls
                        col1, col2 = st.columns(2)
                        with col1:
                            max_points = st.slider(
                                "Max Points to Display", min_value=50, max_value=500, value=200, key="max_points_timeline"
                            )

                        with col2:
                            show_connections = st.checkbox("Show Connection Lines", value=True, key="show_connections")

                        # Generate timeline map
                        timeline_map = create_path_timeline_map(consolidado_df, selected_driver_timeline, max_points)
                        if timeline_map:
                            st.plotly_chart(timeline_map, use_container_width=True, key=f"timeline_map_{selected_driver_timeline}")

                            # Driver timeline statistics
                            driver_data = consolidado_df[
                                (consolidado_df['MOTORISTA'] == selected_driver_timeline)
                                & consolidado_df['LATITUDE'].notna()
                                & consolidado_df['LONGITUDE'].notna()
                                & consolidado_df['LATITUDE.1'].notna()
                                & consolidado_df['LONGITUDE.1'].notna()
                            ]

                            if len(driver_data) > 0:
                                # Convert to datetime for time analysis
                                driver_data['DATETIME_ONIX'] = pd.to_datetime(
                                    driver_data['DATA_SEPARADA'] + ' ' + driver_data['HORA'], format='%d/%m/%Y %H:%M:%S'
                                )

                                time_span = (driver_data['DATETIME_ONIX'].max() - driver_data['DATETIME_ONIX'].min()).total_seconds() / 3600

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Records", len(driver_data))
                                with col2:
                                    st.metric("Time Span", f"{time_span:.1f} hours")
                                with col3:
                                    avg_dist = coord_df[coord_df['DRIVER'] == selected_driver_timeline]['AVG_DISTANCE_KM'].iloc[0]
                                    st.metric("Avg Distance", f"{avg_dist:.1f} km")
                        else:
                            st.warning("No coordinate data available for this driver.")

                # Coordinate data table
                st.header("📋 Coordinate Analysis Data")
                st.dataframe(coord_df.sort_values('AVG_DISTANCE_KM', ascending=False), use_container_width=True, height=400)

            with tab3:
                # Driver Details
                st.header("🔍 Individual Driver Analysis")

                # Driver selection
                selected_driver_detail = st.selectbox(
                    "Select Driver for Detailed Analysis", sorted(df['DRIVER'].tolist()), key="driver_detail"
                )

                if selected_driver_detail:
                    # Get driver data
                    driver_hours = df[df['DRIVER'] == selected_driver_detail].iloc[0]
                    driver_coords = coord_df[coord_df['DRIVER'] == selected_driver_detail].iloc[0]

                    # Driver summary
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("ONIX Hours", f"{driver_hours['ONIX_TOTAL_HOURS']:.1f}")
                        st.metric("Jornada Hours", f"{driver_hours['JORNADA_TOTAL_HOURS']:.1f}")
                        st.metric("Hour Difference", f"{driver_hours['HOURS_DIFFERENCE']:.1f}")

                    with col2:
                        st.metric("Avg Distance", f"{driver_coords['AVG_DISTANCE_KM']:.1f} km")
                        st.metric("Max Distance", f"{driver_coords['MAX_DISTANCE_KM']:.1f} km")
                        st.metric("Records with Coords", f"{driver_coords['RECORDS_WITH_COORDS']}")

                    with col3:
                        st.metric("ONIX Work Starts", f"{driver_hours['ONIX_WORK_STARTS']}")
                        st.metric("ONIX Work Ends", f"{driver_hours['ONIX_WORK_ENDS']}")
                        st.metric("Jornada Work Starts", f"{driver_hours['JORNADA_WORK_STARTS']}")

                    # Coordinate map
                    st.header("🗺️ Coordinate Map")
                    coord_map = create_path_timeline_map(consolidado_df, selected_driver_detail)
                    if coord_map:
                        st.plotly_chart(coord_map, use_container_width=True, key=f"driver_map_{selected_driver_detail}")
                    else:
                        st.warning("No coordinate data available for this driver.")

            # Analysis comparison
            if 'compare_analysis' in st.session_state:
                st.header("🔄 Analysis Comparison")
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

                    # Comparison charts
                    st.subheader("📈 Comparison Charts")

                    # Merge data for comparison
                    comparison_df = pd.merge(
                        df[['DRIVER', 'ONIX_TOTAL_HOURS', 'JORNADA_TOTAL_HOURS', 'HOURS_DIFFERENCE']],
                        compare_df[['DRIVER', 'ONIX_TOTAL_HOURS', 'JORNADA_TOTAL_HOURS', 'HOURS_DIFFERENCE']],
                        on='DRIVER',
                        suffixes=('_Current', '_Compare'),
                    )

                    # Hours comparison
                    fig = px.scatter(
                        comparison_df,
                        x='ONIX_TOTAL_HOURS_Current',
                        y='ONIX_TOTAL_HOURS_Compare',
                        hover_data=['DRIVER'],
                        title="ONIX Hours Comparison Between Analyses",
                        labels={
                            'ONIX_TOTAL_HOURS_Current': f'{current_analysis} ONIX Hours',
                            'ONIX_TOTAL_HOURS_Compare': f'{st.session_state.compare_analysis} ONIX Hours',
                        },
                    )

                    # Add diagonal line
                    max_val = max(comparison_df['ONIX_TOTAL_HOURS_Current'].max(), comparison_df['ONIX_TOTAL_HOURS_Compare'].max())
                    fig.add_trace(
                        go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='red'), name='Perfect Match')
                    )

                    st.plotly_chart(fig, use_container_width=True, key="comparison_chart")

                    # Clear comparison
                    if st.button("❌ Clear Comparison"):
                        del st.session_state.compare_analysis
                        st.rerun()
        else:
            st.error("Analysis data not found.")
    else:
        # Welcome screen
        st.header("👋 Welcome to the Analysis Dashboard")
        st.markdown(
            """
        ### Get Started:
        1. **Upload Files**: Use the sidebar to upload your ONIX and Jornada Excel files
        2. **Create Analysis**: Give your analysis a name and process the files
        3. **Explore Data**: Use the tabs to analyze working hours and coordinates
        4. **Compare Analyses**: Create multiple analyses and compare them
        
        ### Features:
        - ⚡ **Cached Processing**: Fast analysis with intelligent caching
        - 📊 **Working Hours Analysis**: Compare time tracking between systems
        - 🗺️ **Coordinate Analysis**: Visualize GPS accuracy and path timelines
        - 🔄 **Analysis Comparison**: Compare multiple analyses side-by-side
        - 📈 **Interactive Charts**: Dynamic visualizations with Plotly
        """
        )

        # Show saved analyses if any
        analysis_list = get_analysis_list()
        if analysis_list:
            st.subheader("📁 Saved Analyses")
            for analysis in analysis_list:
                analysis_data = load_analysis(analysis)
                if analysis_data:
                    metadata = analysis_data['metadata']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**{analysis}**")
                    with col2:
                        st.write(f"Created: {metadata['created_at'][:19]}")
                    with col3:
                        if st.button(f"Load {analysis}", key=f"load_{analysis}"):
                            st.session_state.current_analysis = analysis
                            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("**Data Source**: ONIX GPS tracking system and Jornada driver status system")
    st.markdown("**Note**: Coordinate distances calculated using Haversine formula")


if __name__ == "__main__":
    main()
