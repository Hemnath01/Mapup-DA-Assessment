import pandas as pd


import numpy as np


def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): DataFrame with columns ['from_id', 'to_id', 'distance']

    Returns:
        pandas.DataFrame: Distance matrix
    """

    unique_ids = pd.unique(df[['from_id', 'to_id']].values.ravel('K'))


    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)


    for _, row in df.iterrows():
        distance_matrix.at[row['from_id'], row['to_id']] = row['distance']
        distance_matrix.at[row['to_id'], row['from_id']] = row['distance']  # Symmetric distances


    np.fill_diagonal(distance_matrix.values, 0)


    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]


    df['distance_matrix'] = distance_matrix.values.flatten()

    return df




def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): Distance matrix DataFrame with index and columns as IDs.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """

    unrolled_data = []


    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:  # Exclude same id_start to id_end
                distance = df.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})


    unrolled_df = pd.DataFrame(unrolled_data)


    df = unrolled_df

    return df





def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame with columns ['id_start', 'id_end', 'distance']
        reference_id (int): The ID to compare against

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """

    reference_distances = df[df['id_start'] == reference_id]['distance']

    if reference_distances.empty:
        return df

    average_distance = reference_distances.mean()


    lower_bound = average_distance * 0.90
    upper_bound = average_distance * 1.10


    avg_distances = df.groupby('id_start')['distance'].mean()
    filtered_ids = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)].index.tolist()


    result_df = pd.DataFrame({'id_start': filtered_ids})


    df = result_df

    return df




def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with columns ['id_start', 'id_end', 'distance']

    Returns:
        pandas.DataFrame: Updated DataFrame with toll rates for each vehicle type.
    """

    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }


    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df


import pandas as pd
import numpy as np
from datetime import time, timedelta


def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): DataFrame with columns ['id_start', 'id_end', 'distance', 'moto', 'car', 'rv', 'bus', 'truck']

    Returns:
        pandas.DataFrame: Updated DataFrame with time-based toll rates.
    """

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


    new_rows = []


    weekday_discount_factors = {
        (time(0, 0), time(10, 0)): 0.8,
        (time(10, 0), time(18, 0)): 1.2,
        (time(18, 0), time(23, 59, 59)): 0.8
    }

    weekend_discount_factor = 0.7


    for _, row in df.iterrows():
        for day in days_of_week:
            for hour in range(24):
                for minute in [0, 30]:
                    current_time = time(hour, minute)

                    if day in ['Saturday', 'Sunday']:
                        # Apply weekend discount
                        new_rows.append({
                            'id_start': row['id_start'],
                            'id_end': row['id_end'],
                            'start_day': day,
                            'start_time': current_time,
                            'end_day': day,
                            'end_time': current_time,
                            'moto': row['moto'] * weekend_discount_factor,
                            'car': row['car'] * weekend_discount_factor,
                            'rv': row['rv'] * weekend_discount_factor,
                            'bus': row['bus'] * weekend_discount_factor,
                            'truck': row['truck'] * weekend_discount_factor,
                        })
                    else:
                        # Apply weekday discounts
                        for (start, end), factor in weekday_discount_factors.items():
                            if start <= current_time < end:
                                new_rows.append({
                                    'id_start': row['id_start'],
                                    'id_end': row['id_end'],
                                    'start_day': day,
                                    'start_time': current_time,
                                    'end_day': day,
                                    'end_time': current_time,
                                    'moto': row['moto'] * factor,
                                    'car': row['car'] * factor,
                                    'rv': row['rv'] * factor,
                                    'bus': row['bus'] * factor,
                                    'truck': row['truck'] * factor,
                                })
                                break


    new_df = pd.DataFrame(new_rows)


    df = new_df

    return df


