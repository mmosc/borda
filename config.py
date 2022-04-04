jku_path = '/home/marta/jku/'
voting_path = jku_path + 'voting/'
rankings_folder_path = voting_path + 'rec_per_user/'

output_path = voting_path

# Length of the list from Davide
initial_k = 50
# Final k for the NDCG@k or whatever@k
final_k = 10

# Logarithmic
#weights_for_one_model = [1., 0.3, 0.1, 0.03, 0.01]

# Linear
weights_for_one_model = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.]

lp_norms = [1, 2]

# Parameter to choose whether to merge all models or two at the time
pairwise = True
