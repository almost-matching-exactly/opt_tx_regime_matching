"""Script to run create_df.py and generate data for simulation setup passed through command line.

Created on October 16, 2023
@author: anonymous
"""

import datetime
import numpy as np
import sys
import time

from simulations.create_df import create_df

_, data_save_folder, sim_num, n_iters, dgp_version, n_patients, \
    n_covs, timestep_setting, time_drop_setting, binary_doses, n_folds, n_neighbors, n_bins = sys.argv

sim_num = int(sim_num)
n_iters = int(n_iters)
n_patients = int(n_patients)
n_covs = int(n_covs)
timestep_setting = int(timestep_setting)
time_drop_setting = int(time_drop_setting)

t_drop_bounds = [0, 0]
if timestep_setting == 0:
    t_bounds = [2, 2]
    if time_drop_setting == 1:
        t_drop_bounds = [0, 1]
elif timestep_setting == 1:
    t_bounds = [10, 15]
    if time_drop_setting == 1:
        t_drop_bounds = [2, 5]
binary_doses = (binary_doses == 'True')

# OptMatch params
n_neighbors = int(n_neighbors)
n_folds = int(n_folds)

# Multilevel params
n_bins = int(n_bins)
multi_bins = list(np.linspace(100 / (2*(n_bins-1)), 100 - (100 / (2*(n_bins-1))), n_bins-1))
multi_interval = multi_bins[-1] - multi_bins[-2]

start = time.time()

for sim_iter in range(n_iters):
    create_df(n_patients, n_covs, t_bounds, t_drop_bounds, data_save_folder, sim_num, sim_iter, n_folds, n_neighbors,
              dgp_version, binary_doses, multi_bins, multi_interval, random_seed=None, verbose=False)
    print(f'Data created for {sim_iter}: {str(datetime.timedelta(seconds = time.time() - start))} total seconds.')
