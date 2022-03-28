import os
import pandas as pd

from config import *
from tqdm import tqdm

import csv


def get_user_ids(rankings_path):
    """
    Given the path to a folder containing the stored rankings
    for each model, it returns a list of the user_id's

    Parameters
    ----------

    rankings_path: path to a folder containing the stored rankings
    for each model
    """

    # Get the list of files with rankings
    rankings = os.listdir(rankings_path)

    # Open one file as a dataframe
    one_rec_df = pd.read_csv(rankings_path + rankings[0])

    # Get the unique values of the user column
    return one_rec_df['user'].unique()


def filter_ranking_df(rankings_df, user_id):
    """
    Function that, given a df containing the rankings
    for all users, filters out only those concerning
    the given user_id

    Parameters
    ----------

    rankings_df: pandas DataFrame. The dataframe containing
    the user-wise rankings for the specific model.
    the df has columns user, item, rank
    
    user_id: string. The user_id

    Returns
    -------
    user_rankings_df: pandas DataFrame. The dataframe containing
    the rankings for the specific model for the individual user.
    Columns: user, item, rank
    """

    return rankings_df.loc[rankings_df['user'] == user_id]


def get_points_df(rankings_df, k=10):
    """
    Function that, given a df containing the rankings
    (for all users or for a single one),
    converts them to points

    Parameters
    ----------

    rankings_df: pandas DataFrame. The dataframe containing
    the user-wise rankings for the specific model.
    the df has columns user, item, rank
    
    Returns
    -------
    user_points_df: pandas DataFrame. The dataframe containing
    the rankings as well as points for the specific model.
    the df has columns: user, item, rank, points
    """

    rankings_df['points'] = k + 1 - rankings_df['rank']
    return rankings_df


def sum_borda_points(rankings_path, user_id, user_dataframe_columns):
    """
    Function that aggregates the rankings of different recommenders into 
    a single ranking for a specific user according to Borda count. 

    Parameters
    ----------

    rankings_path: string. Path to the folder containing
    the recommendations of the different recommenders
    stored each in a .csv file,
    the CSVs have columns user, item, rank
    
    user_id: string. The user_id of the user for whom to fuse
    the recommenders
    
    user_dataframe_columns: list of strings. The columns of the
    returned dataframe

    
    Returns
    -------
    user_points_df: pandas DataFrame. The dataframe containing
    the rankings as well as points after fusing the models.
    the df has columns: user, item, rank, points
    """

    # Define the single user dataframe
    user_df = pd.DataFrame(columns=user_dataframe_columns)

    # Iterate over the files in the rankings_path folder
    for ranking_file in os.listdir(rankings_path):
        # Open a single rankings file as a pandas dataframe
        one_rec_df = pd.read_csv(rankings_path + ranking_file, sep=',')

        # Filter out the rows for this user
        one_rec_df = filter_ranking_df(one_rec_df, user_id)

        # Add the points
        one_rec_df = get_points_df(one_rec_df)

        # Drop rank column
        one_rec_df = one_rec_df.drop(columns=['rank'])

        # Drop user_id column
        one_rec_df = one_rec_df.drop(columns=['user'])

        # Concatenate with the user_df 
        user_df = pd.concat([user_df, one_rec_df], ignore_index=True)

    # Sum points of equal items
    user_df = user_df.groupby('item').sum()

    # Sort on descending points
    user_df = user_df.sort_values(['points'], ascending=[False])

    # Reset user column
    user_df['user'] = user_id

    # Get the ranking as an index+1
    user_df = user_df.reset_index()

    user_df['rank'] = user_df.index + 1

    return user_df[['user', 'rank', 'item', 'points']]


def append_user_top_k_to_fused_pandas(rankings_path, k=10):
    """
    Appends the top_k of the given user to the
    dataframe containing the fused recommendations
    for all the users.

    Parameters
    ----------

    rankings_path: string. Path to the folder containing
    the recommendations of the different recommenders
    stored each in a .csv file, with columns
    the df has columns user, item, rank
    
    k: int. The number of elements to recommend

    
    Returns
    -------
    fused_recommendations: pandas DataFrame. The dataframe containing
    the fused recommendations for all users
    
    Columns: user, top_k
    """

    # Get the unique user_ids
    users = get_user_ids(rankings_path)
    # print( 'There are ', len(users), 'unique users.' )

    # Columns of the final dataframe
    fused_dataframe_columns = ['user', 'top_k']

    # Initialize the final dataframe
    fused_recommendations = pd.DataFrame(columns=fused_dataframe_columns)

    # columns of the individual user dataframe
    user_dataframe_columns = ['user', 'item', 'points']

    for user_id in tqdm(users):
        user_df = sum_borda_points(rankings_path, user_id, user_dataframe_columns)
        user_data = {'user': user_df.user.values[0], 'top_k': user_df.user.values[:k]}
        fused_recommendations = pd.concat([fused_recommendations,
                                           pd.DataFrame.from_dict(user_data, orient='columns')],
                                          ignore_index=True)

    return fused_recommendations


def append_user_top_k_to_fused(rankings_path, k=10):
    """
    Appends the top_k of all the users to the
    file containing the fused recommendations.

    Parameters
    ----------

    rankings_path: string. Path to the folder containing
    the recommendations of the different recommenders
    stored each in a .csv file, with columns
    the df has columns user, item, rank
    
    k: int. The number of elements to recommend

    
    Returns
    -------
    fused_recommendations: pandas DataFrame. The dataframe containing
    the fused recommendations for all users
    
    Columns: user, top_k
    """

    # Get the unique user_ids
    users = get_user_ids(rankings_path)
    # print( 'There are ', len(users), 'unique users.' )

    # Columns of the final dataframe
    fused_dataframe_columns = ['user', 'top_k']

    # columns of the individual user dataframe
    user_dataframe_columns = ['user', 'item', 'points']

    with open(output_path, 'w') as output_file:
        # Open the output file
        writer = csv.writer(output_file, delimiter='\t')
        # Write the header: user_id and list of top_k
        writer.writerow(fused_dataframe_columns)

        # Iterate over all users
        for user_id in tqdm(users):
            # Get the borda points as a dataframe and order them descending
            user_df = sum_borda_points(rankings_path, user_id, user_dataframe_columns)
            # Write the user_id as well as the first k elements in the final dataframe
            user_data = [user_df.user.values[0], list(user_df.item.values[:k])]
            writer.writerow(user_data)
