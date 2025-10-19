from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth in kilometers."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers
    r = 6371
    return c * r


def calculate_coordinate_analysis():
    """Calculate coordinate differences and statistics for each driver."""
    print("Calculating coordinate analysis...")

    # Load consolidated data
    consolidado_df = pd.read_excel('files/Consolidado.xlsx')

    # Filter records with both coordinate sets
    coord_data = consolidado_df[
        consolidado_df['LATITUDE'].notna()
        & consolidado_df['LONGITUDE'].notna()
        & consolidado_df['LATITUDE.1'].notna()
        & consolidado_df['LONGITUDE.1'].notna()
    ].copy()

    print(f"Records with both coordinate sets: {len(coord_data)}")

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

    coord_df = pd.DataFrame(driver_coord_stats)

    # Save coordinate analysis
    coord_df.to_excel('files/Coordinate_Analysis.xlsx', index=False)
    print(f"Coordinate analysis saved to: files/Coordinate_Analysis.xlsx")

    return coord_df


def get_driver_coordinate_details(driver_name):
    """Get detailed coordinate data for a specific driver."""
    consolidado_df = pd.read_excel('files/Consolidado.xlsx')

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

    return driver_data


if __name__ == "__main__":
    # Run coordinate analysis
    coord_df = calculate_coordinate_analysis()

    print("\n=== COORDINATE ANALYSIS SUMMARY ===")
    print(f"Total drivers with coordinate data: {len(coord_df)}")
    print(f"Average distance between systems: {coord_df['AVG_DISTANCE_KM'].mean():.3f} km")
    print(f"Maximum average distance: {coord_df['AVG_DISTANCE_KM'].max():.3f} km")
    print(f"Minimum average distance: {coord_df['AVG_DISTANCE_KM'].min():.3f} km")

    print("\nTop 5 drivers with highest average distances:")
    top_distances = coord_df.nlargest(5, 'AVG_DISTANCE_KM')
    for _, row in top_distances.iterrows():
        print(f"  {row['DRIVER']}: {row['AVG_DISTANCE_KM']} km")
