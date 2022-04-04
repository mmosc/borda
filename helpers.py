import os
import pandas as pd
pd.options.mode.chained_assignment = None

from tqdm import tqdm

import csv
import numpy as np

import json
from datetime import datetime
from itertools import combinations


def generate_id(prefix=None, postfix=None):
    date_time_obj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(
        date_time_obj.year,
        date_time_obj.month,
        date_time_obj.day,
        date_time_obj.hour,
        date_time_obj.minute,
        date_time_obj.second,
        date_time_obj.microsecond
    )
    if prefix is None:
        pass
    else:
        uid = prefix + "_" + uid

    if postfix is None:
        pass
    else:
        uid = uid + "_" + postfix

    return uid


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

    k:
    
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


def append_user_top_k_to_fused(rankings_path, out_path, k=10):
    """
    Appends the top_k of all the users to the
    file containing the fused recommendations.

    Parameters
    ----------

    rankings_path: string. Path to the folder containing
    the recommendations of the different recommenders
    stored each in a .csv file, with columns
    the df has columns user, item, rank

    out_path:

    k: int. The number of elements to recommend


    Returns
    -------
    fused_recommendations: pandas DataFrame. The dataframe containing
    the fused recommendations for all users

    Columns: user, top_k
    """

    # Get the unique user_ids
    users = get_user_ids(rankings_path)

    # Columns of the final dataframe
    fused_dataframe_columns = ['user', 'top_k']

    # columns of the individual user dataframe
    user_dataframe_columns = ['user', 'item', 'points']

    with open(out_path, 'w') as output_file:
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


def pandas_simple_borda(
        rankings_path,
        output_csv,
        k_init=50,
        k_final=10
):
    """
    Computes the simple Borda count from a list of rankings provided by
    recommender users. The ranked list are passed in as individual .csv
    stored in the ranking_path folder and the resulting ranked lists
    after voting are both returned as a dataframe and stored to a .csv
    under the output_csv path.

    Parameters
    ----------

    rankings_path: string. Path to the folder containing
    the recommendations of the different recommenders
    stored each in a .csv file, with columns
    the df has columns user, item, rank

    output_csv: path to the file where the results of the voting are stored

    k_init: the length of the initial lists stored in rankings_path

    k_final: the length of the final recommended list


    Returns
    -------
    ranking_df: pandas DataFrame. The dataframe containing
    the fused recommendations for all users

    """
    # Columns of the final dataframe
    fused_dataframe_columns = ['user', 'item', 'rank']

    # List of file names
    ranking_files = os.listdir(rankings_path)

    ###
    # Concatenate all dataframes
    ###
    list_of_dataframes = []

    for individual_ranking in ranking_files:
        df = pd.read_csv(rankings_path + individual_ranking, index_col=False, sep=',')
        list_of_dataframes.append(df)

    all_rankings_together = pd.concat(list_of_dataframes, ignore_index=True)

    # Assign points
    all_rankings_together['points'] = k_init + 1 - all_rankings_together['rank']

    # Drop individual recommenders' rankings
    all_rankings_together = all_rankings_together.drop(columns=['rank'])

    # group by user and item and sum points
    # I.e. for each user, get the sum of the points
    # of each unique item
    all_rankings_together = all_rankings_together.groupby(['user', 'item']).sum()

    # Unset item from index
    # leave only user as index
    all_rankings_together = all_rankings_together.reset_index(level=[1])

    # Sort rows according to user id
    # and for each user id sort rows according to points
    all_rankings_together = all_rankings_together.reset_index().sort_values(['user', 'points'], ascending=False)

    # Get only the first final_k elements for each user
    top_k_points = all_rankings_together.groupby('user').head(k_final)

    # Assign the ranking of each item
    top_k_points['rank'] = top_k_points.groupby(['user']).cumcount().add(1).values

    # Select the relevant columns and write the .csv
    ranking_df = top_k_points[fused_dataframe_columns]
    ranking_df.to_csv(output_csv, index=False, sep=',')

    return ranking_df


def pandas_weighted_borda_all_models(
        rankings_path,
        output_folder,
        k_init=50,
        k_final=10,
        weights=None,
        norm=1
):
    """
    Computes the weighted Borda count from a list of rankings provided by
    recommender users. The ranked list are passed in as individual .csv
    stored in the ranking_path folder and the resulting ranked lists
    after voting are both returned as a dataframe and stored to a .csv
    under the output_csv path.

    The weighted voting is computed by assigning the first weight
    to the first file retrieved by os.listdir, and so on.
    If no weights are passed, the weights are all given the same
    value.

    The weights are normalized according to Lp norm passed as argument.
    If none is given, L1 is applied (i.e. sum=1).

    Parameters
    ----------

    rankings_path: string. Path to the folder containing
    the recommendations of the different recommenders
    stored each in a .csv file, with columns
    the df has columns user, item, rank

    output_folder: path to the file where the results of the voting are stored

    k_init: the length of the initial lists stored in rankings_path

    k_final: the length of the final recommended list

    weights: the list of weights to be used in the voting

    norm: the L_norm norm to be used in the voting


    Returns
    -------
    ranking_df: pandas DataFrame. The dataframe containing
    the fused recommendations for all users

    """

    # Define the path of the final .csv
    output_csv = output_folder + '/weighted_list.csv'

    # Define the path to the .json storing info on the grid element
    output_json = output_folder + '/grid_params.json'

    # Initialize the params dictionary to be stored and store the norm in it
    params = {'norm': norm, 'models': []}

    # List of file names
    ranking_files = os.listdir(rankings_path)

    # If no weights are passed, set weights to a list of ones.
    # This is equivalent to a non-weighted Borda
    if weights is None:
        weights = np.ones(len(ranking_files))

    # print("Computing weighted average with L{} norm".format(norm))

    # If the weights are not normalised, normalise them

    # Convert the weights to a numpy array
    weights = np.array(weights)

    # Compute the LP norm
    normalization = np.linalg.norm(weights, ord=norm)

    # If the norm is not 1, re-normalize the weights
    if normalization != 1:
        #       warnings.warn("The norm of the given weights under the given norm is not 1! Re-normalizing the weights...")
        weights = weights / normalization

    # The weights are considered to the power of norm (the p in Lp)
    weights = weights ** norm

    # Columns of the final dataframe
    fused_dataframe_columns = ['user', 'item', 'rank']

    ###
    # Concatenate all dataframes
    ###
    list_of_dataframes = []

    for idx, individual_ranking in enumerate(ranking_files):
        # Open the individual ranking
        one_ranking_df = pd.read_csv(rankings_path + individual_ranking, index_col=False, sep=',')
        # Assign points according to the ranking and to the weight of this recommender
        one_ranking_df['points'] = (k_init + 1 - one_ranking_df['rank']) * weights[idx]
        list_of_dataframes.append(one_ranking_df)

        # Store the model info and weight in the param dictionary
        model_params = get_model_info_from_filename(individual_ranking)
        # Add weight to the param dictionary of the specific model
        model_params['weight'] = weights[idx]

        # Add the specific model params to the overall dictionary of params
        params['models'].append(model_params)

    # Write model params and weights to the .json
    with open(output_json, 'w') as fp:
        json.dump(params, fp)

    # Concatenate all dataframes
    all_rankings_together = pd.concat(list_of_dataframes, ignore_index=True)

    # Drop individual recommenders' rankings
    all_rankings_together = all_rankings_together.drop(columns=['rank'])

    # group by user and item and sum points
    # I.e. for each user, get the sum of the points
    # of each unique item
    all_rankings_together = all_rankings_together.groupby(['user', 'item']).sum()

    # Unset item from index
    # leave only user as index
    all_rankings_together = all_rankings_together.reset_index(level=[1])

    # Sort rows according to user id
    # and for each user id sort rows according to points
    all_rankings_together = all_rankings_together.reset_index().sort_values(['user', 'points'], ascending=False)

    # Get only the first final_k elements for each user
    top_k_points = all_rankings_together.groupby('user').head(k_final)

    # Assign the ranking of each item
    top_k_points['rank'] = top_k_points.groupby(['user']).cumcount().add(1).values

    # Select the relevant columns and write the .csv
    ranking_df = top_k_points[fused_dataframe_columns]
    ranking_df.to_csv(output_csv, index=False, sep=',')

    return ranking_df


def pandas_weighted_borda_pairwise(
        rankings_path,
        output_folder,
        k_init=50,
        k_final=10,
        weights_grid_onemodel=None,
        norm=1
):
    """
    Computes the weighted Borda count for pairs of files
    from a list of rankings provided by
    recommender users. The ranked list are passed in as individual .csv
    stored in the ranking_path folder and the resulting ranked lists
    after voting are both returned as a dataframe and stored to a .csv
    under the output_csv path.

    The weighted voting is computed by assigning the first weight
    to the first file retrieved by os.listdir, and
    getting the weights of the second one by requiring the
    normalisation to be one under the specified Lp norm.
    If no weights are passed, the weights are all given the same
    value.

    The weights are normalized according to Lp norm passed as argument.
    If none is given, L1 is applied (i.e. sum=1).

    Parameters
    ----------

    rankings_path: string. Path to the folder containing
    the recommendations of the different recommenders
    stored each in a .csv file, with columns
    the df has columns user, item, rank

    output_folder: path to the file where the results of the voting are stored

    k_init: the length of the initial lists stored in rankings_path

    k_final: the length of the final recommended list

    weights_grid_onemodel: the list of weights of a single model to be used in the voting

    norm: the L_norm norm to be used in the voting


    Returns
    -------
    ranking_df: pandas DataFrame. The dataframe containing
    the fused recommendations for all users

    """

    output_folder = output_folder + '/pairwise/'

    # If no weights_grid_onemodel are passed, set weights_grid_onemodel
    # to a list with one 2**(-norm).
    # This is equivalent to a non-weighted Borda
    # for two models.
    if weights_grid_onemodel is None:
        weights_grid_onemodel = [2 ** (-norm)]

    # List of file names
    ranking_files = os.listdir(rankings_path)

    # generate pairs of models without repetition
    model_pairs = combinations(ranking_files, 2)

    # print("Computing weighted average with L{} norm".format(norm))
    for model_1, model_2 in tqdm(model_pairs):

        for model_1_weight in weights_grid_onemodel:
            model_2_weight = (1 - model_1_weight ** norm) ** (1 / norm)
            # print("Model 1: ", model_1, "\t weight: ", model_1_weight)
            # print("Model 2: ", model_2, "\t weight: ", model_2_weight)

            # Columns of the final dataframe
            fused_dataframe_columns = ['user', 'item', 'rank']

            ###
            # Concatenate all dataframes
            ###
            list_of_dataframes = []

            # Initialize the params dictionary to be stored and store the norm in it
            params = {'norm': norm, 'models': []}

            for (individual_ranking, weight) in ((model_1, model_1_weight), (model_2, model_2_weight)):
                # Open the individual ranking
                one_ranking_df = pd.read_csv(rankings_path + individual_ranking, index_col=False, sep=',')
                # Assign points according to the ranking and to the weight of this recommender
                one_ranking_df['points'] = (k_init + 1 - one_ranking_df['rank']) * weight
                list_of_dataframes.append(one_ranking_df)

                # Store the model info and weight in the param dictionary
                model_params = get_model_info_from_filename(individual_ranking)
                # Add weight to the param dictionary of the specific model
                model_params['weight'] = round(weight, 2)

                # Add the specific model params to the overall dictionary of params
                params['models'].append(model_params)

            # model pair folder
            model_pair_folder = output_folder
            for model_params in params['models']:
                for value in model_params.values():
                    model_pair_folder += str(value)
            model_pair_folder += '/'

            # Create the folder for this specific pair
            if not os.path.exists(model_pair_folder):
                os.makedirs(model_pair_folder)

            # Define the path to the final ranked list
            model_pair_folder_csv = model_pair_folder + 'weighted_list.csv'

            # Define the path to the .json storing info on the grid element
            output_json = model_pair_folder + 'voting_params.json'

            # Write model params and weights to the .json
            with open(output_json, 'w') as fp:
                json.dump(params, fp)

            # Concatenate all dataframes
            all_rankings_together = pd.concat(list_of_dataframes, ignore_index=True)

            # Drop individual recommenders' rankings
            all_rankings_together = all_rankings_together.drop(columns=['rank'])

            # group by user and item and sum points
            # I.e. for each user, get the sum of the points
            # of each unique item
            all_rankings_together = all_rankings_together.groupby(['user', 'item']).sum()

            # Unset item from index
            # leave only user as index
            all_rankings_together = all_rankings_together.reset_index(level=[1])

            # Sort rows according to user id
            # and for each user id sort rows according to points
            all_rankings_together = all_rankings_together.reset_index().sort_values(['user', 'points'], ascending=False)

            # Get only the first final_k elements for each user
            top_k_points = all_rankings_together.groupby('user').head(k_final)

            # Assign the ranking of each item
            top_k_points['rank'] = top_k_points.groupby(['user']).cumcount().add(1).values

            # Select the relevant columns and write the .csv
            ranking_df = top_k_points[fused_dataframe_columns]
            ranking_df.to_csv(model_pair_folder_csv, index=False, sep=',')

    return 1


def get_model_info_from_filename(filename):
    """
    Function fetching the model information from the filename string. 
    The input string is of the form
    model=BiVAECF_audio=VAD_textual=None_rec_per_user.csv    
    
    The function returns a dictionary of the type
    {'model': 'BiVAECF', 'audio': 'VAD', 'textual': 'None'}

    Parameters
    ----------

    filename: path to the ranking file of the model
    
    Returns
    -------
    model_info: dictionary storing the parameters of the model
    """

    model_info = {param[0]: param[1] for param in
                  [single_entry.split("=") for single_entry in filename.split("_")[:-3]]}
    return model_info


def generate_grid(single_weights, num_models):
    """
    Given a list of weights, it replicates it and returns it
    as a dictionary to be used as grid in the search of weights
    """

    return {'model_' + str(index): single_weights for index in range(num_models)}
