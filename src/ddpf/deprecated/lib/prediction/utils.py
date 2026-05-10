# Functions
import ast
import pandas as pd
import numpy as np

def obsToList(obs,isint=False):
    # Transforms one observation in a list of strings or integers based on isint
    # Safely evaluate the string as a list
    parsed_list = ast.literal_eval(obs)
    if isint:
        return [int(x) for x in parsed_list]
    else:
        return [str(x) for x in parsed_list]

def getTopMax(df_observations, top=10):
    # For each observation in df_observations, filter the 'c.id', 'c.name' and 'intensity' to only include the
    # top highest values based on variable top
    for index, row in df_observations.iterrows():
        newsort_value = []
        newsort_names = []
        newsort_id = []
        for (x,y,z) in zip( obsToList(row['KB_IDs']),obsToList(row['intensity'],True),obsToList(row['KB_Components'])):
            if z not in newsort_names:
                newsort_names.append(z)
                newsort_value.append(y)
                newsort_id.append(x)
        newsort_names = [x for _, x in sorted(zip(newsort_value, newsort_names ))]
        newsort_id = [x for _, x in sorted(zip(newsort_value, newsort_id ))]
        newsort_value.sort()
        df_observations.at[index,'KB_IDs'] = str(newsort_id[-min(top,len(newsort_id)):])
        df_observations.at[index,'intensity'] = str(newsort_value[-min(top,len(newsort_id)):])
        df_observations.at[index,'KB_Components'] = str(newsort_names[-min(top,len(newsort_id)):])
    return df_observations

def update_dict_values_by_column_order(dictionary, dataframe, column_name):
    """
    Updates the values of a dictionary based on the order and occurrence of its keys
    in a specified column of a given DataFrame.

    Parameters:
    dictionary (dict): The dictionary whose values need updating.
    dataframe (pd.DataFrame): The pandas DataFrame containing the reference column.
    column_name (str): The name of the column in the dataframe to use for ordering.

    Returns:
    dict: The updated dictionary with values assigned sequentially for each occurrence.
    """
    # Ensure the column exists
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Standardize the DataFrame column to strings for consistency
    dataframe[column_name] = dataframe[column_name].astype(str)

    # Track the first appearance order of keys from the dictionary in the DataFrame column
    first_appearance_order = []
    seen_keys = set()
    for value in dataframe[column_name]:
        if value in dictionary and value not in seen_keys:
            first_appearance_order.append(value)
            seen_keys.add(value)

    # Update the dictionary values based on the new order
    updated_dict = {key: i for i, key in enumerate(first_appearance_order)}

    return updated_dict