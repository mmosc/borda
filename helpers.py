import os
import pandas as pd


def get_user_ids( rankings_path ):
    '''
    Given the path to a folder containing the stored rankings 
    for each model, it returns a list of the user_id's
    '''

    # Get the list of files with rankings
    rankings = os.listdir( rankings_path )
    
    # Open one file it as a dataframe
    one_rec_df = pd.read_csv( rankings_path + rankings[0] )
    
    # Get the unique values of the user column
    return one_rec_df['user'].unique()

    

def filter_ranking_df( rankings_df, user_id ):
    '''
    Function that, given a df containing the rankings
    for all users, filters out only those concerning
    the given user_id

    Parameters
    ----------

    rankings_df: pandas DataFrame.  The dataframe containing
    the user-wise rankings for the specific model.
    Columns: user, item, rank
    
    user_id: string. The user_id

    Returns
    -------
    user_rankings_df: pandas DataFrame.  The dataframe containing
    the rankings for the specific model for the individual user.
    Columns: user, item, rank
    '''
    
    
    return rankings_df.loc[rankings_df['user'] == user_id]



def get_points_df( rankings_df ):
    '''
    Function that, given a df containing the rankings
    ( for all users or for a single one), 
    converts them to points

    Parameters
    ----------

    rankings_df: pandas DataFrame.  The dataframe containing
    the user-wise rankings for the specific model.
    Columns: user, item, rank
    
    Returns
    -------
    user_points_df: pandas DataFrame.  The dataframe containing
    the rankings as well as pointsfor the specific model.
    
    Columns: user, item, rank, points
    '''
    
    
    rankings_df['points'] = 11 - rankings_df['rank']
    return rankings_df


def sum_borda_points( rankings_path, user_id, user_dataframe_columns ):
    '''
    Function that aggregates the rankings of different recommenders into 
    a single ranking for a specific user according to Borda count. 

    Parameters
    ----------

    rankings_path: string. Path to the folder containing
    the recommendations of the different recommenders
    stored each in a .csv file, with columns
    Columns: user, item, rank
    
    user_id: string. The user_id ofthe user for whom to fuse
    the recommenders
    
    user_dataframe_columns: list of strings. The columns of the
    returned dataframe

    
    Returns
    -------
    user_points_df: pandas DataFrame.  The dataframe containing
    the rankings as well as points after fusing the models.
    
    Columns: user, item, rank, points
    '''
    
    

    # Define the single user dataframe
    user_df = pd.DataFrame( columns=user_dataframe_columns )
    
    # Iterate over the files in the rankings_path folder
    for ranking_file in os.listdir( rankings_path ):
        # Open a single rankings file as a pandas dataframe
        one_rec_df = pd.read_csv( rankings_path + ranking_file, sep=',' )

        # Filter out the rows for this user
        one_rec_df = filter_ranking_df( one_rec_df, user_id )

        # Add the points
        one_rec_df = get_points_df( one_rec_df )

        # Drop rank column
        one_rec_df = one_rec_df.drop( columns=['rank'] )

        # Drop user_id column
        one_rec_df = one_rec_df.drop( columns=['user'] )

        # Concatenate with the user_df 
        user_df = pd.concat( [user_df, one_rec_df], ignore_index=True )
    
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