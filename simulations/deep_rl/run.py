"""Script to run deep_rl.py with args passed through command line.

Created on October 16, 2023
@author: anonymous
"""

from deep_rl import deep_rl_kfold
import sys

_, save_folder, sim_num, n_iters, model_type, reward_function, n_steps, n_folds = sys.argv
reward_function = int(reward_function)
n_steps = int(n_steps)
n_folds = int(n_folds)
for sim_iter in range(int(n_iters)):
    try:
        random_seed = int(sim_iter)
        df_obs_file = f'{save_folder}/sim_{sim_num}_iter_{sim_iter}_df_obs.csv'  # absolute path to the data file
        df_true_file = f'{save_folder}/sim_{sim_num}_iter_{sim_iter}_df_true.csv'  # relative path to the data file
        # model_save_file = f'{save_folder}/deep_rl/models/sim_{sim_num}_r{reward_function}_iter_{sim_iter}'  # absolute path where to save fitted model
        model_save_file = None  # don't save trained models, comment out and uncomment above line to save models.
        outcomes_save_file = f'{save_folder}/sim_{sim_num}_iter_{sim_iter}_{model_type}_r{reward_function}_outcomes.csv'  # absolute path where to save outcomes

        outcomes = deep_rl_kfold(df_obs_file, df_true_file, model_type=model_type, reward_function=reward_function,
                                 n_steps=n_steps, n_folds=n_folds,
                                 model_save_file=model_save_file, outcomes_save_file=outcomes_save_file,
                                 random_seed=random_seed)
    except AssertionError as ae:
        print(f'Did not run iter {sim_iter} due to discrete action space error:')
        print(ae)
        print()
