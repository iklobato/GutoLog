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


def load_data():
    """Load the working hours comparison data."""
    try:
        df = pd.read_excel('files/Working_Hours_Comparison.xlsx')
        return df
    except FileNotFoundError:
        st.error("Working_Hours_Comparison.xlsx file not found. Please run the merge script first.")
        return None


def load_coordinate_data():
    """Load the coordinate analysis data."""
    try:
        coord_df = pd.read_excel('files/Coordinate_Analysis.xlsx')
        return coord_df
    except FileNotFoundError:
        st.error("Coordinate_Analysis.xlsx file not found. Please run the coordinate analysis script first.")
        return None


def load_consolidated_data():
    """Load the consolidated data for detailed coordinate analysis."""
    try:
        df = pd.read_excel('files/Consolidado.xlsx')
        return df
    except FileNotFoundError:
        st.error("Consolidado.xlsx file not found. Please run the merge script first.")
        return None


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


def create_driver_coordinate_map(consolidado_df, driver_name):
    """Create a map showing coordinate differences for a specific driver."""
    driver_data = consolidado_df[
        (consolidado_df['MOTORISTA'] == driver_name)
        & consolidado_df['LATITUDE'].notna()
        & consolidado_df['LONGITUDE'].notna()
        & consolidado_df['LATITUDE.1'].notna()
        & consolidado_df['LONGITUDE.1'].notna()
    ].copy()

    if len(driver_data) == 0:
        return None

    # Calculate distances
    distances = []
    for _, row in driver_data.iterrows():
        dist = haversine_distance(row['LATITUDE'], row['LONGITUDE'], row['LATITUDE.1'], row['LONGITUDE.1'])
        distances.append(dist)

    driver_data['DISTANCE_KM'] = distances

    # Sample data for map (too many points would make it slow)
    sample_data = driver_data.sample(min(100, len(driver_data)))

    fig = go.Figure()

    # Add ONIX coordinates
    fig.add_trace(
        go.Scattermapbox(
            lat=sample_data['LATITUDE'],
            lon=sample_data['LONGITUDE'],
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='ONIX Coordinates',
            text=sample_data['DATA_SEPARADA'] + ' ' + sample_data['HORA'],
            hovertemplate='<b>ONIX</b><br>Date: %{text}<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>',
        )
    )

    # Add Jornada coordinates
    fig.add_trace(
        go.Scattermapbox(
            lat=sample_data['LATITUDE.1'],
            lon=sample_data['LONGITUDE.1'],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Jornada Coordinates',
            text=sample_data['DATA_SEPARADA.1'] + ' ' + sample_data['HORA.1'],
            hovertemplate='<b>Jornada</b><br>Date: %{text}<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>',
        )
    )

    fig.update_layout(
        title=f"Coordinate Comparison Map - {driver_name}",
        mapbox=dict(style="open-street-map", center=dict(lat=sample_data['LATITUDE'].mean(), lon=sample_data['LONGITUDE'].mean()), zoom=10),
        height=600,
    )

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
    driver_data['DATETIME_ONIX'] = pd.to_datetime(driver_data['DATA_SEPARADA'] + ' ' + driver_data['HORA'], format='%d/%m/%Y %H:%M:%S')
    driver_data['DATETIME_JORNADA'] = pd.to_datetime(
        driver_data['DATA_SEPARADA.1'] + ' ' + driver_data['HORA.1'], format='%d/%m/%Y %H:%M:%S'
    )

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

    # Calculate center point
    all_lats = list(driver_data['LATITUDE']) + list(driver_data['LATITUDE.1'])
    all_lons = list(driver_data['LONGITUDE']) + list(driver_data['LONGITUDE.1'])

    fig.update_layout(
        title=f"Path Timeline Map - {driver_name}<br><sub>Blue: ONIX Path | Red: Jornada Path | Gray: Connections</sub>",
        mapbox=dict(style="open-street-map", center=dict(lat=np.mean(all_lats), lon=np.mean(all_lons)), zoom=11),
        height=700,
        showlegend=True,
    )

    return fig


def create_multi_driver_timeline_map(consolidado_df, selected_drivers, max_points_per_driver=50):
    """Create a map showing path timelines for multiple drivers."""
    if not selected_drivers:
        return None

    fig = go.Figure()

    # Color palette for different drivers
    colors = px.colors.qualitative.Set3

    for i, driver_name in enumerate(selected_drivers):
        driver_data = consolidado_df[
            (consolidado_df['MOTORISTA'] == driver_name)
            & consolidado_df['LATITUDE'].notna()
            & consolidado_df['LONGITUDE'].notna()
            & consolidado_df['LATITUDE.1'].notna()
            & consolidado_df['LONGITUDE.1'].notna()
        ].copy()

        if len(driver_data) == 0:
            continue

        # Convert time strings to datetime for proper sorting
        driver_data['DATETIME_ONIX'] = pd.to_datetime(driver_data['DATA_SEPARADA'] + ' ' + driver_data['HORA'], format='%d/%m/%Y %H:%M:%S')

        # Sort by ONIX datetime
        driver_data = driver_data.sort_values('DATETIME_ONIX')

        # Sample data if too many points
        if len(driver_data) > max_points_per_driver:
            step = len(driver_data) // max_points_per_driver
            driver_data = driver_data.iloc[::step]

        # Create time-based color scale for this driver
        min_time = driver_data['DATETIME_ONIX'].min()
        max_time = driver_data['DATETIME_ONIX'].max()
        time_range = (max_time - min_time).total_seconds()

        driver_data['TIME_NORMALIZED'] = (driver_data['DATETIME_ONIX'] - min_time).dt.total_seconds() / time_range if time_range > 0 else 0

        color = colors[i % len(colors)]

        # Add ONIX path
        fig.add_trace(
            go.Scattermapbox(
                lat=driver_data['LATITUDE'],
                lon=driver_data['LONGITUDE'],
                mode='markers+lines',
                marker=dict(size=4, color=color),
                line=dict(width=2, color=color),
                name=f'{driver_name} - ONIX',
                text=driver_data['DATA_SEPARADA'] + ' ' + driver_data['HORA'],
                hovertemplate=f'<b>{driver_name} - ONIX</b><br>Time: %{{text}}<br>Lat: %{{lat:.6f}}<br>Lon: %{{lon:.6f}}<extra></extra>',
            )
        )

        # Add Jornada path
        fig.add_trace(
            go.Scattermapbox(
                lat=driver_data['LATITUDE.1'],
                lon=driver_data['LONGITUDE.1'],
                mode='markers+lines',
                marker=dict(size=4, color=color),
                line=dict(width=2, color=color, dash='dash'),
                name=f'{driver_name} - Jornada',
                text=driver_data['DATA_SEPARADA.1'] + ' ' + driver_data['HORA.1'],
                hovertemplate=f'<b>{driver_name} - Jornada</b><br>Time: %{{text}}<br>Lat: %{{lat:.6f}}<br>Lon: %{{lon:.6f}}<extra></extra>',
            )
        )

    # Calculate center point
    all_lats = []
    all_lons = []
    for driver_name in selected_drivers:
        driver_data = consolidado_df[
            (consolidado_df['MOTORISTA'] == driver_name)
            & consolidado_df['LATITUDE'].notna()
            & consolidado_df['LONGITUDE'].notna()
            & consolidado_df['LATITUDE.1'].notna()
            & consolidado_df['LONGITUDE.1'].notna()
        ]
        if len(driver_data) > 0:
            all_lats.extend(driver_data['LATITUDE'].tolist())
            all_lats.extend(driver_data['LATITUDE.1'].tolist())
            all_lons.extend(driver_data['LONGITUDE'].tolist())
            all_lons.extend(driver_data['LONGITUDE.1'].tolist())

    if all_lats and all_lons:
        fig.update_layout(
            title=f"Multi-Driver Path Timeline Map<br><sub>Solid: ONIX Paths | Dashed: Jornada Paths</sub>",
            mapbox=dict(style="open-street-map", center=dict(lat=np.mean(all_lats), lon=np.mean(all_lons)), zoom=10),
            height=700,
            showlegend=True,
        )

    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Working Hours & Coordinates Comparison Dashboard", page_icon="⏰", layout="wide")

    st.title("⏰ Working Hours & Coordinates Comparison Dashboard")
    st.markdown("Compare working hours and coordinate accuracy between ONIX and Jornada systems")

    # Load data
    df = load_data()
    coord_df = load_coordinate_data()
    consolidado_df = load_consolidated_data()

    if df is None or coord_df is None or consolidado_df is None:
        return

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📊 Working Hours Analysis", "🗺️ Coordinate Analysis", "🔍 Driver Details"])

    with tab1:
        # Working Hours Analysis (existing code)
        st.header("📊 Working Hours Analysis")

        # Sidebar filters for working hours
        st.sidebar.header("Working Hours Filters")

        # Driver filter
        all_drivers = ['All'] + sorted(df['DRIVER'].tolist())
        selected_driver = st.sidebar.selectbox("Select Driver", all_drivers, key="hours_driver")

        # Difference range filter
        min_diff = df['HOURS_DIFFERENCE'].min()
        max_diff = df['HOURS_DIFFERENCE'].max()

        diff_range = st.sidebar.slider(
            "Hour Difference Range",
            min_value=float(min_diff),
            max_value=float(max_diff),
            value=(float(min_diff), float(max_diff)),
            step=1.0,
            key="hours_range",
        )

        # Apply filters
        filtered_df = df.copy()

        if selected_driver != 'All':
            filtered_df = filtered_df[filtered_df['DRIVER'] == selected_driver]

        filtered_df = filtered_df[(filtered_df['HOURS_DIFFERENCE'] >= diff_range[0]) & (filtered_df['HOURS_DIFFERENCE'] <= diff_range[1])]

        # Summary metrics
        create_summary_metrics(filtered_df)

        # Main charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_hours_comparison_chart(filtered_df), use_container_width=True)

        with col2:
            st.plotly_chart(create_difference_distribution(filtered_df), use_container_width=True)

        # Top drivers chart
        st.header("🏆 Top Drivers by Hour Difference")
        top_n = st.slider("Number of top drivers to show", 5, 20, 10, key="top_n_hours")
        st.plotly_chart(create_top_drivers_chart(filtered_df, top_n), use_container_width=True)

        # Detailed work sessions analysis
        st.header("🔍 Detailed Work Sessions Analysis")
        st.plotly_chart(create_work_sessions_chart(filtered_df), use_container_width=True)

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
            st.plotly_chart(create_coordinate_distance_chart(coord_df), use_container_width=True)

        with col2:
            st.plotly_chart(create_coordinate_scatter_plot(coord_df), use_container_width=True)

        # Distribution chart
        st.plotly_chart(create_coordinate_distribution_chart(coord_df), use_container_width=True)

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
                    max_points = st.slider("Max Points to Display", min_value=50, max_value=500, value=200, key="max_points_timeline")

                with col2:
                    show_connections = st.checkbox("Show Connection Lines", value=True, key="show_connections")

                # Generate timeline map
                timeline_map = create_path_timeline_map(consolidado_df, selected_driver_timeline, max_points)
                if timeline_map:
                    st.plotly_chart(timeline_map, use_container_width=True)

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

        else:  # Multi-Driver Comparison
            # Multi-driver comparison map
            available_drivers = sorted(coord_df['DRIVER'].tolist())
            selected_drivers_multi = st.multiselect(
                "Select Drivers for Comparison (max 5)",
                available_drivers,
                default=available_drivers[:3] if len(available_drivers) >= 3 else available_drivers,
                max_selections=5,
                key="multi_drivers",
            )

            if selected_drivers_multi:
                col1, col2 = st.columns(2)
                with col1:
                    max_points_per_driver = st.slider(
                        "Max Points per Driver", min_value=20, max_value=100, value=50, key="max_points_multi"
                    )

                with col2:
                    st.info(f"Selected: {len(selected_drivers_multi)} drivers")

                # Generate multi-driver map
                multi_map = create_multi_driver_timeline_map(consolidado_df, selected_drivers_multi, max_points_per_driver)
                if multi_map:
                    st.plotly_chart(multi_map, use_container_width=True)

                    # Multi-driver statistics
                    st.subheader("📊 Multi-Driver Statistics")
                    multi_stats = coord_df[coord_df['DRIVER'].isin(selected_drivers_multi)].sort_values('AVG_DISTANCE_KM', ascending=False)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(
                            multi_stats[['DRIVER', 'AVG_DISTANCE_KM', 'MAX_DISTANCE_KM', 'RECORDS_WITH_COORDS']],
                            use_container_width=True,
                            height=300,
                        )

                    with col2:
                        # Summary statistics
                        st.metric("Total Drivers", len(selected_drivers_multi))
                        st.metric("Avg Distance", f"{multi_stats['AVG_DISTANCE_KM'].mean():.1f} km")
                        st.metric("Max Distance", f"{multi_stats['AVG_DISTANCE_KM'].max():.1f} km")
                        st.metric("Total Records", multi_stats['RECORDS_WITH_COORDS'].sum())
                else:
                    st.warning("No coordinate data available for selected drivers.")
            else:
                st.info("Please select at least one driver for comparison.")

        # Coordinate data table
        st.header("📋 Coordinate Analysis Data")
        st.dataframe(coord_df.sort_values('AVG_DISTANCE_KM', ascending=False), use_container_width=True, height=400)

    with tab3:
        # Driver Details
        st.header("🔍 Individual Driver Analysis")

        # Driver selection
        selected_driver_detail = st.selectbox("Select Driver for Detailed Analysis", sorted(df['DRIVER'].tolist()), key="driver_detail")

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
            coord_map = create_driver_coordinate_map(consolidado_df, selected_driver_detail)
            if coord_map:
                st.plotly_chart(coord_map, use_container_width=True)
            else:
                st.warning("No coordinate data available for this driver.")

    # Footer
    st.markdown("---")
    st.markdown("**Data Source**: ONIX GPS tracking system and Jornada driver status system")
    st.markdown("**Note**: Coordinate distances calculated using Haversine formula")


if __name__ == "__main__":
    main()
