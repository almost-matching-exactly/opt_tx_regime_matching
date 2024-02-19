"""Functions to generate data for the simulation study according to the passed arguments.

Created on October 16, 2023
@author: anonymous
"""
import numpy as np
import pandas as pd

from simulations.sim_helpers import simulate


def create_df(n_patients, n_covs, t_bounds, t_drop_bounds, data_save_folder, sim_num, sim_iter, n_folds, n_neighbors,
              dgp_version, binary_doses, multi_bins, multi_interval, random_seed=None, verbose=False):
    """Simulates data according to the passed arguments. See details in paper.

    Parameters
    ----------
    n_patients : int
        number of patients
    n_covs : int
        number of pre-treatment covariates
    t_bounds : list(int)
        list of two integers, where the first value is the minimum number of timesteps and the second value is the
        maximum number of timesteps. Both values are inclusive.
    t_drop_bounds : list(int)
        list of two integers, where the first value is the minimum number of unobserved timesteps and the second value
        is the maximum number of unobserved timesteps. Both values are inclusive.
    data_save_folder : str
        where to save the generated datasets
    sim_num : int
        number assigned to this simulation setup
    sim_iter : int
        iteration of the simulation setup. Also used as the random seed if random_seed is None
    n_folds : int
        number of folds used for our method and deep rl. Not used to generate data, but written to the config file.
    n_neighbors : int
        number of neighbors to use for our method. Not used to generate data, but written to the config file.
    dgp_version : str
        policy type to use when generated data. "random" for random policy or "semi-random" for informed policy.
    binary_doses : bool
        whether the dose space is binary or not.
    multi_bins : list[float]
        cutoffs to use when creating bins for multi-treatment methods. Not used to generate data, but written to
        the config file.
    multi_interval : float
        interval between bins for multi-treatment methods. Not used to generate data, but written to the config file.
    random_seed : None or int, default=None
        random state to run on. If not specified, random_seed is set to sim_iter.
    verbose : bool, default=False
        whether to print progress or not.
    """

    if random_seed is None:
        random_seed = sim_iter
    np.random.seed(random_seed)
    with open(f'{data_save_folder}/sim_{sim_num}_iter_{sim_iter}_config.txt', 'w') as f:
        f.write(f'# samples: {n_patients}\n')
        f.write(f'# covs: {n_covs}\n')
        f.write(f'Timestep bounds: {t_bounds}\n')
        f.write(f'Timestep drop bounds: {t_drop_bounds}\n')
        f.write(f'Random seed: {random_seed}\n')
        f.write(f'Analysis folds: {n_folds}\n')
        f.write(f'QLearner multi bins: {multi_bins}\n')
        f.write(f'QLearner multi interval: {multi_interval}\n')
        f.write(f'# Neighbors: {n_neighbors}\n')
        f.write(f'DGP: {dgp_version}')
        f.write(f'Binary doses: {binary_doses}')

    X = np.random.normal(0, 1, size=(n_patients, n_covs))
    beta = X[:, 0]*10 + np.random.normal(100, 5, size=(n_patients,))
    alpha = np.random.normal(1, 0.1, size=(n_patients,))
    ed50 = np.clip(-X[:, 2]*2 + np.random.normal(15, 1, size=(n_patients,)), a_min=1, a_max=None)
    gamma = np.random.normal(1, 0.1, size=(n_patients,))
    timesteps = np.random.randint(t_bounds[0], t_bounds[1]+1, size=(n_patients,))
    random_drop = np.random.randint(t_drop_bounds[0], t_drop_bounds[1]+1, size=(n_patients,))

    init_burden = X[:, 1]*5 + np.random.normal(75, 5, size=(n_patients,))
    init_burden = np.maximum(np.minimum(init_burden, beta), 0)  # make sure that the initial burden isn't larger than the max burden

    df_true = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    df_true['init_burden'] = init_burden
    df_true['beta'] = beta
    df_true['alpha'] = alpha
    df_true['ed50'] = ed50
    df_true['gamma'] = gamma
    df_true['timesteps'] = timesteps
    df_true['missing_timesteps'] = random_drop
    df_true.to_csv(f'{data_save_folder}/sim_{sim_num}_iter_{sim_iter}_df_true.csv', index=False)
    if verbose:
        print(f'Saved true df to {data_save_folder}/sim_{sim_num}_iter_{sim_iter}_df_true.csv')

    cont_outcomes = []
    ea_outcomes = []
    conc_outcomes = []
    full_burdens = []
    burdens = []
    doses = []
    concentrations = []
    features = []
    preset_polices = []
    for pat in range(n_patients):
        outcome, burden, dose, concentration, feature, policy_array, ea_out, conc_out = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                  init_burden[pat], timesteps[pat], policy=dgp_version, preset_doses=None, binary_doses=binary_doses, return_hist=True)
        cont_outcomes.append(outcome)
        ea_outcomes.append(ea_out)
        conc_outcomes.append(conc_out)
        full_burdens.append(burden[:len(burden)-random_drop[pat]])
        concentrations.append(concentration)
        burden = burden[:-1]
        burdens.append(burden[:len(burden)-random_drop[pat]])
        doses.append(dose[:len(dose)-random_drop[pat]])
        features.append(feature[:len(feature)-random_drop[pat]])
        preset_polices.append(policy_array)

    cont_outcomes = np.array(cont_outcomes)

    # binarize the outcomes
    # outcomes = (1 / (1 + np.exp(-((cont_outcomes - np.mean(cont_outcomes)) / np.std(cont_outcomes))))) > 0.5
    outcomes = cont_outcomes > 3

    df_obs = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    df_obs = df_obs.join(pd.DataFrame(full_burdens,
                                      columns=[f'Burden{i}' for i in range(max([len(t) for t in full_burdens]))]
                                      )
                         )
    df_obs = df_obs.join(pd.DataFrame(doses,
                                      columns=[f'Dose{i}' for i in range(max([len(t) for t in doses]))]
                                      )
                         )
    df_obs = df_obs.join(pd.DataFrame(concentrations,
                                      columns=[f'Conc{i}' for i in range(max([len(t) for t in concentrations]))]
                                      )
                         )
    df_obs = df_obs.join(pd.DataFrame(
        [f.ravel() for f in features],
        columns=[f'T{j}F{i}' for j in range(max([h.shape[0] for h in features])) for i in range(features[0].shape[1])]
    ))

    df_obs['ea_outcome'] = ea_outcomes
    df_obs['conc_outcome'] = conc_outcomes
    df_obs['cont_outcome'] = cont_outcomes
    df_obs['binary_outcome'] = outcomes
    df_obs.to_csv(f'{data_save_folder}/sim_{sim_num}_iter_{sim_iter}_df_obs.csv', index=False)
    if verbose:
        print(f'Saved observational df to {data_save_folder}/sim_{sim_num}_iter_{sim_iter}_df_obs.csv')
