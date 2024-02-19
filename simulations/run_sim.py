"""Script to run sim.py and generate optimal treatment regime estimation results for our method, inf horizon methods,
and qlearning methods. See methods in methods/ folder.

Created on October 16, 2023
@author: anonymous
"""

import datetime
import numpy as np
import sys
import time

from simulations.sim import sim

_, data_save_folder, sim_num, n_iters, binary_doses, n_folds, n_neighbors, n_bins = sys.argv

sim_num = int(sim_num)
n_iters = int(n_iters)
binary_doses = (binary_doses == 'True')

# OptMatch params
n_folds = int(n_folds)
n_neighbors = int(n_neighbors)

# Multilevel params
n_bins = int(n_bins)
multi_bins = list(np.linspace(100 / (2*(n_bins-1)), 100 - (100 / (2*(n_bins-1))), n_bins-1))
multi_interval = multi_bins[-1] - multi_bins[-2]

start = time.time()

for method in ['baseline', 'matching', 'q', 'inf']:
    for sim_iter in range(n_iters):
        sim(data_save_folder, sim_num, sim_iter, n_folds, n_neighbors, binary_doses, multi_bins, multi_interval,
            method=method, random_seed=None, verbose=False)
        print(f'{method} iter {sim_iter} complete: {str(datetime.timedelta(seconds = time.time() - start))}')
