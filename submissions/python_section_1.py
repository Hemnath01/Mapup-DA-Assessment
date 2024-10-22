from typing import Dict, List

import pandas as pd
import polyline
import numpy as np



def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.


    """
    # Your code goes here.
    length = len(lst)
    for i in range(0, length, n):
        group = []
    for j in range(n):
        if i + j < length:
            group.append(lst[i + j])
    for j in range(len(group) - 1, -1, -1):
        lst[i + j] = group[j]
    return lst

if __name__ == "__main__":
    # Example input for the reverse function
    input_list = [1, 2, 3, 4, 5, 6, 7, 8]
    n = 3
    output = reverse_by_n_elements(input_list, n)
    print(f"Input: {input_list}, n={n} => Output: {output}")


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}

    for string in lst:
        length = len(string)

        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))

if __name__ == "__main__":
    input_list_1 = ["apple", "bat", "car", "elephant", "dog", "bear"]
    output_1 = group_by_length(input_list_1)
    print(f"Input: {input_list_1} => Output: {output_1}")

    input_list_2 = ["one", "two", "three", "four"]
    output_2 = group_by_length(input_list_2)
    print(f"Input: {input_list_2} => Output: {output_2}")

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:

    # Your code here
    flat_dict = {}

    def _flatten(current_dict: Dict[str, Any], parent_key: str = ''):
        for key, value in current_dict.items():

            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):

                _flatten(value, new_key)
            elif isinstance(value, list):

                for index, item in enumerate(value):
                    item_key = f"{new_key}[{index}]"
                    if isinstance(item, dict):
                        _flatten(item, item_key)
                    else:
                        flat_dict[item_key] = item
            else:
                flat_dict[new_key] = value

    _flatten(nested_dict)
    return dict(flat_dict)


# Example test cases
if __name__ == "__main__":
    nested_dict = {
        "road": {
            "name": "Highway 1",
            "length": 350,
            "sections": [
                {
                    "id": 1,
                    "condition": {
                        "pavement": "good",
                        "traffic": "moderate"
                    }
                }
            ]
        }
    }

    flattened = flatten_dict(nested_dict)
    print(flattened)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  # Append a copy of the current permutation
            return

        for i in range(start, len(nums)):
            # Skip duplicates
            if i > start and nums[i] == nums[start]:
                continue

            # Swap to create a new permutation
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)  # Recurse
            # Swap back to backtrack
            nums[start], nums[i] = nums[i], nums[start]

            # Sort the numbers to handle duplicates
        nums.sort()
        result = []
        backtrack(0)
        return result  # Return the list of unique permutations


if __name__ == "__main__":
        input_list = [1, 1, 2]
        output = unique_permutations(input_list)
        print(output)




def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """

    patterns = [
        r'\b(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{4})\b',  # dd-mm-yyyy
        r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])-(\d{4})\b',  # mm/dd/yyyy
        r'\b(\d{4})\.(0[1-9]|1[0-2])\.(0[1-9]|[12][0-9]|3[01])\b'  # yyyy.mm.dd
    ]

    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if pattern == patterns[0]:  # dd-mm-yyyy
                dates.append(f"{match[0]}-{match[1]}-{match[2]}")
            elif pattern == patterns[1]:  # mm/dd/yyyy
                dates.append(f"{match[0]}/{match[1]}-{match[2]}")
            elif pattern == patterns[2]:  # yyyy.mm.dd
                dates.append(f"{match[0]}.{match[1]}.{match[2]}")

    return dates

if __name__ == "__main__":
    text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
    output = find_all_dates(text)
    print(output)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in meters (mean radius)
    r = 6371000
    return c * r

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Calculate the distance between consecutive points
    distances = [0.0]  # First distance is 0
    for i in range(1, len(df)):
        dist = haversine(df.latitude[i - 1], df.longitude[i - 1], df.latitude[i], df.longitude[i])
        distances.append(dist)

    # Add the distances to the DataFrame
    df['distance'] = distances

    return df

if __name__ == "__main__":
    polyline_str = "u{~wFzj}w@h@pCj@zC"  # Example polyline
    df = polyline_to_dataframe(polyline_str)
    print(df)




def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Create a new matrix for the final transformation
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Calculate the sum of the row and column, excluding the current element
            row_sum = sum(rotated_matrix[i])  # Sum of the i-th row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the j-th column
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude current element

    return final_matrix

if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    transformed_matrix = rotate_and_multiply_matrix(matrix)
    print(transformed_matrix)


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create a multi-index boolean series
    completeness = pd.Series(index=df.set_index(['id', 'id_2']).index, dtype=bool)

    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    for (id_val, id_2_val), group in grouped:
        # Get the start and end of the timestamp range for the group
        min_time = group['timestamp'].min()
        max_time = group['timestamp'].max()

        # Check if it spans 7 days
        spans_seven_days = (max_time - min_time).days >= 6
        all_hours_covered = (group['timestamp'].dt.floor('H').nunique() == 24)  # 24 unique hours

        # Set the boolean value in the series
        completeness.loc[(id_val, id_2_val)] = spans_seven_days and all_hours_covered

    return completeness

    # Example usage
    if __name__ == "__main__":
        # Load dataset-1.csv
        df = pd.read_csv('dataset-1.csv')
        result = time_check(df)
        print(result)

    return pd.Series()


