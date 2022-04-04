from helpers import *
from config import *
from itertools import product

if __name__ == "__main__":
    # Old version: takes forever
    # append_user_top_k_to_fused(rankings_folder_path)

    # Generate the output folder with ID given by the tmestamp
    #     output_path = generate_id()
    #     if not os.path.exists(output_path):
    #         os.makedirs('grid_timestamp_' + output_path)i

    # Get the number of models
    num_models = len(os.listdir(rankings_folder_path))
    #print("there are {} models".format(num_models))

    # Generate the grid with values according to
    # weights_for_one_model
    weights_grid = generate_grid(
        single_weights=weights_for_one_model,
        num_models=num_models
    )

    # Iterate over norms
    for norm in tqdm(lp_norms):
        # Iterate over the combinations of weights
        for weights in tqdm(product(*weights_grid.values()), leave=bool(norm == 2)):
            # Generate the output folder with ID given by the tmestamp
            grid_id = generate_id()
            grid_path = output_path + grid_id
            if not os.path.exists(grid_path):
                os.makedirs(grid_path)
            # convert the tuple to a list
            weights = list(weights)

            pandas_weighted_borda(
                rankings_path=rankings_folder_path,
                output_folder=grid_path,
                k_init=50,
                k_final=10,
                weights=weights,
                norm=norm
            )
