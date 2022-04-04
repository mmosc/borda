jku_path = '/home/marta/jku/'
voting_path = jku_path + 'voting/'
rankings_folder_path = voting_path + 'rec_per_user/'

output_path = voting_path

# Length of the list from Davide
initial_k = 50
# Final k for the NDCG@k or whatever@k
final_k = 10

weights_for_one_model = [1., 0.3, 0.1, 0.03, 0.01]

lp_norms = [1, 2]
