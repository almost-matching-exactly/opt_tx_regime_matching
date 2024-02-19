"""Script to generate optimal treatment regime estimates and calculate the outcomes under these estimates.

Created on October 16, 2023
@author: anonymous
"""

import numpy as np
import pandas as pd

from simulations.methods.opt_matching import opt_matx_kfold
from simulations.methods.qlearning import QLearner
from simulations.methods.inf_horizon import InfHorizon
from simulations.sim_helpers import simulate


def sim(data_save_folder, sim_num, sim_iter, n_folds, n_neighbors,
        binary_doses, multi_bins, multi_interval, method='all', random_seed=None, verbose=False):
    """Generates optimal policies for specified method, simulates outcomes under these policies, and saved results.

    Parameters
    ----------
    data_save_folder : str
        location of datasets generated from create_df.py and also where results from this script are saved.
    sim_num : int
        number assigned to this simulation setup
    sim_iter : int
        iteration of the simulation setup. Also used as the random seed if random_seed is None
    n_folds : int
        number of folds used for our method
    n_neighbors : int
        number of neighbors to use for our method
    binary_doses : bool
        whether the dose space is binary or not.
    multi_bins : list[float]
        cutoffs to use when creating bins for multi-treatment methods.
    multi_interval : float
        interval between bins for multi-treatment methods.
    method : str
        which methods to run. "baseline" runs preset policy methods. "matching" runs our method. "q" runs q-learning
        methods. "inf" runs infinite horizon methods. "all" runs all methods.
    random_seed : None or int, default=None
        random state to run on. If not specified, random_seed is set to sim_iter.
    verbose : bool, default=False
        whether to print progress or not.
    """

    if random_seed is None:
        random_seed = sim_iter
    np.random.seed(random_seed)

    df_true = pd.read_csv(f'{data_save_folder}/sim_{sim_num}_iter_{sim_iter}_df_true.csv')
    df_obs = pd.read_csv(f'{data_save_folder}/sim_{sim_num}_iter_{sim_iter}_df_obs.csv')

    alpha = df_true['alpha'].to_numpy()
    beta = df_true['beta'].to_numpy()
    gamma = df_true['gamma'].to_numpy()
    ed50 = df_true['ed50'].to_numpy()
    init_burden = df_obs['Burden0'].to_numpy()
    timesteps = df_true['timesteps'].to_numpy()

    X = df_obs[[c for c in df_obs.columns if 'X' in c]].to_numpy()
    outcomes = df_obs['binary_outcome'].to_numpy()
    feature_cols = [c for c in df_obs.columns if 'T' in c]
    dose_cols = [c for c in df_obs.columns if 'Dose' in c]
    burden_cols = [c for c in df_obs.columns if 'Burden' in c]
    concentration_cols = [c for c in df_obs.columns if 'Conc' in c]
    n_features = int(len(feature_cols) /
                     max(df_true['timesteps'] - df_true['missing_timesteps']))
    features = []
    doses = []
    full_burdens = []
    concentrations = []
    for i, row in df_obs.iterrows():
        features.append(row[feature_cols].dropna().astype(int).to_numpy().reshape(-1, n_features))
        doses.append(row[dose_cols].dropna().astype(float).to_numpy().reshape(-1,))
        full_burdens.append(row[burden_cols].dropna().astype(float).to_numpy().reshape(-1, ))
        concentrations.append(row[concentration_cols].dropna().astype(float).to_numpy()[:1+df_true.loc[i, 'timesteps']-df_true.loc[i, 'missing_timesteps']].reshape(-1, ))
    burdens = [np.array(f[:-1]) for f in full_burdens]

    if method == 'baseline' or method == 'all':
        coi_outcomes = {}
        coa_outcomes = {}
        random_outcomes = {}
        expert_outcomes = {}
        if binary_doses:
            full_dose = 50
        else:
            full_dose = 100
        for pat in range(df_obs.shape[0]):
            coi_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat], init_burden[pat],
                                         timesteps[pat], preset_doses=np.array([0]),
                                         binary_doses=binary_doses, dose_interval=0, return_hist=False, burden_noise=0)
            coa_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat], init_burden[pat],
                                         timesteps[pat], preset_doses=np.array([1]),
                                         binary_doses=binary_doses, dose_interval=full_dose, return_hist=False, burden_noise=0)
            random_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat], init_burden[pat],
                                         timesteps[pat], policy='random',
                                            binary_doses=binary_doses, return_hist=False, burden_noise=0)
            expert_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat], init_burden[pat],
                                            timesteps[pat], policy='expert',
                                            binary_doses=binary_doses, return_hist=False, burden_noise=0)
        coi_outcomes = pd.DataFrame.from_dict([coi_outcomes]).T
        coi_outcomes.columns = ['Inaction']
        coa_outcomes = pd.DataFrame.from_dict([coa_outcomes]).T
        coa_outcomes.columns = ['Action']
        random_outcomes = pd.DataFrame.from_dict([random_outcomes]).T
        random_outcomes.columns = ['Random']
        expert_outcomes = pd.DataFrame.from_dict([expert_outcomes]).T
        expert_outcomes.columns = ['Expert']
        baseline_outcomes = coi_outcomes.join(coa_outcomes, how='outer').join(random_outcomes, how='outer')\
            .join(expert_outcomes, how='outer')
        baseline_outcomes['Observed'] = df_obs['cont_outcome'].to_numpy()
        baseline_outcomes.to_csv(f'{data_save_folder}/sim_{sim_num}_iter_{sim_iter}_baseline_outcomes.csv', index=False)

    if method == 'matching' or method == 'all':
        if verbose:
            print('Running OptMatch w/o Params...')
        opt_match_policies = opt_matx_kfold(X, outcomes, features, doses, n_folds=n_folds, n_neighbors=n_neighbors,
                                            binary_doses=binary_doses, random_seed=random_seed)
        opt_matching_outcomes = {}
        for pat in range(df_obs.shape[0]):
            opt_matching_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat], init_burden[pat],
                                                  timesteps[pat], policy_array=opt_match_policies.loc[pat].to_numpy(),
                                                  binary_doses=binary_doses, return_hist=False, burden_noise=0)

        opt_matching_outcomes = pd.DataFrame.from_dict([opt_matching_outcomes]).T
        opt_matching_outcomes.columns = ['Matching']
        opt_matching_outcomes.to_csv(f'{data_save_folder}/sim_{sim_num}_iter_{sim_iter}_matching_outcomes.csv',
                                     index=False)

    if method == 'q' or method == 'all':
        q_linear_binary = QLearner(X, burdens, doses, outcomes, treatment_bins=[25], method='linear')
        q_linear_binary.fit()
        q_linear_binary_final_model, q_linear_binary_scalers = q_linear_binary.get_final_policy_model()

        q_rf_binary = QLearner(X, burdens, doses, outcomes, treatment_bins=[25], method='RF')
        q_rf_binary.fit()
        q_rf_binary_final_model, q_rf_binary_scalers = q_rf_binary.get_final_policy_model()

        q_sv_binary = QLearner(X, burdens, doses, outcomes, treatment_bins=[25], method='SV')
        q_sv_binary.fit()
        q_sv_binary_final_model, q_sv_binary_scalers = q_sv_binary.get_final_policy_model()

        q_linear_binary_outcomes = {}
        q_rf_binary_outcomes = {}
        q_sv_binary_outcomes = {}

        for pat in range(df_obs.shape[0]):
            q_linear_binary_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                    init_burden[pat],
                                                    timesteps[pat],
                                                    preset_doses=q_linear_binary.opt_policy[pat],
                                                    policy_model=q_linear_binary_final_model,
                                                    model_scalers=q_linear_binary_scalers,
                                                    dose_interval=50,
                                                    return_hist=False, burden_noise=0)
            q_rf_binary_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                init_burden[pat],
                                                timesteps[pat],
                                                preset_doses=q_rf_binary.opt_policy[pat],
                                                policy_model=q_rf_binary_final_model,
                                                model_scalers=q_rf_binary_scalers,
                                                dose_interval=50,
                                                return_hist=False, burden_noise=0)
            q_sv_binary_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                init_burden[pat],
                                                timesteps[pat],
                                                preset_doses=q_sv_binary.opt_policy[pat],
                                                policy_model=q_sv_binary_final_model,
                                                model_scalers=q_sv_binary_scalers,
                                                dose_interval=50,
                                                return_hist=False, burden_noise=0)

        q_linear_binary_outcomes = pd.DataFrame.from_dict([q_linear_binary_outcomes]).T
        q_linear_binary_outcomes.columns = ['Q Linear binary']
        q_rf_binary_outcomes = pd.DataFrame.from_dict([q_rf_binary_outcomes]).T
        q_rf_binary_outcomes.columns = ['Q RF binary']
        q_sv_binary_outcomes = pd.DataFrame.from_dict([q_sv_binary_outcomes]).T
        q_sv_binary_outcomes.columns = ['Q SV binary']

        q_outcomes = q_linear_binary_outcomes.join(q_rf_binary_outcomes, how='outer')\
            .join(q_sv_binary_outcomes, how='outer')

        if not binary_doses:
            if verbose:
                print('Running Q Learning methods...')
            q_linear_multi = QLearner(X, burdens, doses, outcomes, treatment_bins=multi_bins, method='linear')
            q_linear_multi.fit()
            q_linear_multi_final_model, q_linear_multi_scalers = q_linear_multi.get_final_policy_model()

            q_rf_multi = QLearner(X, burdens, doses, outcomes, treatment_bins=multi_bins, method='RF')
            q_rf_multi.fit()
            q_rf_multi_final_model, q_rf_multi_scalers = q_rf_multi.get_final_policy_model()

            q_sv_multi = QLearner(X, burdens, doses, outcomes, treatment_bins=multi_bins, method='SV')
            q_sv_multi.fit()
            q_sv_multi_final_model, q_sv_multi_scalers = q_sv_multi.get_final_policy_model()

            q_linear_multi_outcomes = {}
            q_rf_multi_outcomes = {}
            q_sv_multi_outcomes = {}

            for pat in range(df_obs.shape[0]):
                q_linear_multi_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                        init_burden[pat],
                                                        timesteps[pat],
                                                        preset_doses=q_linear_multi.opt_policy[pat],
                                                        policy_model=q_linear_multi_final_model,
                                                        model_scalers=q_linear_multi_scalers,
                                                        dose_interval=multi_interval,
                                                        return_hist=False, burden_noise=0)
                q_rf_multi_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                    init_burden[pat],
                                                    timesteps[pat],
                                                    preset_doses=q_rf_multi.opt_policy[pat],
                                                    policy_model=q_rf_multi_final_model,
                                                    model_scalers=q_rf_multi_scalers,
                                                    dose_interval=multi_interval,
                                                    return_hist=False, burden_noise=0)
                q_sv_multi_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                    init_burden[pat],
                                                    timesteps[pat],
                                                    preset_doses=q_sv_multi.opt_policy[pat],
                                                    policy_model=q_sv_multi_final_model,
                                                    model_scalers=q_sv_multi_scalers,
                                                    dose_interval=multi_interval,
                                                    return_hist=False, burden_noise=0)

            q_linear_multi_outcomes = pd.DataFrame.from_dict([q_linear_multi_outcomes]).T
            q_linear_multi_outcomes.columns = ['Q Linear Multi']
            q_rf_multi_outcomes = pd.DataFrame.from_dict([q_rf_multi_outcomes]).T
            q_rf_multi_outcomes.columns = ['Q RF Multi']
            q_sv_multi_outcomes = pd.DataFrame.from_dict([q_sv_multi_outcomes]).T
            q_sv_multi_outcomes.columns = ['Q SV Multi']
            q_outcomes = q_outcomes.join(q_linear_multi_outcomes, how='outer')\
                .join(q_rf_multi_outcomes, how='outer').join(q_sv_multi_outcomes, how='outer')

        q_outcomes.to_csv(f'{data_save_folder}/sim_{sim_num}_iter_{sim_iter}_q_multi_outcomes.csv', index=False)

    if method == 'inf' or method == 'all':
        if verbose:
            print('Running inf horizon methods...')
        inf_linear_binary_r1 = InfHorizon(X, full_burdens, doses, concentrations=concentrations,
                                          method='linear', treatment_bins=[25], reward_function=1)
        inf_linear_binary_r1.fit()

        inf_rf_binary_r1 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='RF',
                                      treatment_bins=[25], reward_function=1)
        inf_rf_binary_r1.fit()

        inf_sv_binary_r1 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='SV',
                                      treatment_bins=[25], reward_function=1)
        inf_sv_binary_r1.fit()

        inf_linear_binary_r2 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='linear',
                                          treatment_bins=[25], reward_function=2)
        inf_linear_binary_r2.fit()

        inf_rf_binary_r2 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='RF',
                                      treatment_bins=[25], reward_function=2)
        inf_rf_binary_r2.fit()

        inf_sv_binary_r2 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='SV',
                                      treatment_bins=[25], reward_function=2)
        inf_sv_binary_r2.fit()

        inf_linear_binary_r3 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='linear',
                                          treatment_bins=[25], reward_function=3)
        inf_linear_binary_r3.fit()

        inf_rf_binary_r3 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='RF',
                                      treatment_bins=[25], reward_function=3)
        inf_rf_binary_r3.fit()

        inf_sv_binary_r3 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='SV',
                                      treatment_bins=[25], reward_function=3)
        inf_sv_binary_r3.fit()

        inf_linear_binary_r1_outcomes = {}
        inf_rf_binary_r1_outcomes = {}
        inf_sv_binary_r1_outcomes = {}
        inf_linear_binary_r2_outcomes = {}
        inf_rf_binary_r2_outcomes = {}
        inf_sv_binary_r2_outcomes = {}
        inf_linear_binary_r3_outcomes = {}
        inf_rf_binary_r3_outcomes = {}
        inf_sv_binary_r3_outcomes = {}

        if not binary_doses:
            inf_linear_multi_r1 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='linear',
                                             treatment_bins=multi_bins, reward_function=1)
            inf_linear_multi_r1.fit()

            inf_rf_multi_r1 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='RF',
                                         treatment_bins=multi_bins, reward_function=1)
            inf_rf_multi_r1.fit()

            inf_sv_multi_r1 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='SV',
                                         treatment_bins=multi_bins, reward_function=1)
            inf_sv_multi_r1.fit()

            inf_linear_multi_r2 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='linear',
                                             treatment_bins=multi_bins, reward_function=2)
            inf_linear_multi_r2.fit()

            inf_rf_multi_r2 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='RF',
                                         treatment_bins=multi_bins, reward_function=2)
            inf_rf_multi_r2.fit()

            inf_sv_multi_r2 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='SV',
                                         treatment_bins=multi_bins, reward_function=2)
            inf_sv_multi_r2.fit()

            inf_linear_multi_r3 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='linear',
                                             treatment_bins=multi_bins, reward_function=3)
            inf_linear_multi_r3.fit()

            inf_rf_multi_r3 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='RF',
                                         treatment_bins=multi_bins, reward_function=3)
            inf_rf_multi_r3.fit()

            inf_sv_multi_r3 = InfHorizon(X, full_burdens, doses, concentrations=concentrations, method='SV',
                                         treatment_bins=multi_bins, reward_function=3)
            inf_sv_multi_r3.fit()

            inf_linear_multi_r1_outcomes = {}
            inf_rf_multi_r1_outcomes = {}
            inf_sv_multi_r1_outcomes = {}
            inf_linear_multi_r2_outcomes = {}
            inf_rf_multi_r2_outcomes = {}
            inf_sv_multi_r2_outcomes = {}
            inf_linear_multi_r3_outcomes = {}
            inf_rf_multi_r3_outcomes = {}
            inf_sv_multi_r3_outcomes = {}

        for pat in range(df_obs.shape[0]):
            inf_linear_binary_r1_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                          init_burden[pat], timesteps[pat],
                                                          inf_policy_model=inf_linear_binary_r1.q_model,
                                                          prop_model=inf_linear_binary_r1.prop_model,
                                                          model_scalers=inf_linear_binary_r1.scalers,
                                                          dose_interval=50, return_hist=False,
                                                          burden_noise=0)
            inf_rf_binary_r1_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                      init_burden[pat], timesteps[pat],
                                                      inf_policy_model=inf_rf_binary_r1.q_model,
                                                      prop_model=inf_rf_binary_r1.prop_model,
                                                      model_scalers=inf_rf_binary_r1.scalers,
                                                      dose_interval=50, return_hist=False,
                                                      burden_noise=0)
            inf_sv_binary_r1_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                      init_burden[pat], timesteps[pat],
                                                      inf_policy_model=inf_sv_binary_r1.q_model,
                                                      prop_model=inf_sv_binary_r1.prop_model,
                                                      model_scalers=inf_sv_binary_r1.scalers,
                                                      dose_interval=50, return_hist=False, burden_noise=0)
            inf_linear_binary_r2_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                          init_burden[pat], timesteps[pat],
                                                          inf_policy_model=inf_linear_binary_r2.q_model,
                                                          prop_model=inf_linear_binary_r2.prop_model,
                                                          model_scalers=inf_linear_binary_r2.scalers,
                                                          dose_interval=50, return_hist=False,
                                                          burden_noise=0)
            inf_rf_binary_r2_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                      init_burden[pat], timesteps[pat],
                                                      inf_policy_model=inf_rf_binary_r2.q_model,
                                                      prop_model=inf_rf_binary_r2.prop_model,
                                                      model_scalers=inf_rf_binary_r2.scalers,
                                                      dose_interval=50, return_hist=False,
                                                      burden_noise=0)
            inf_sv_binary_r2_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                      init_burden[pat], timesteps[pat],
                                                      inf_policy_model=inf_sv_binary_r2.q_model,
                                                      prop_model=inf_sv_binary_r2.prop_model,
                                                      model_scalers=inf_sv_binary_r2.scalers,
                                                      dose_interval=50, return_hist=False, burden_noise=0)
            inf_linear_binary_r3_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                          init_burden[pat], timesteps[pat],
                                                          inf_policy_model=inf_linear_binary_r3.q_model,
                                                          prop_model=inf_linear_binary_r3.prop_model,
                                                          model_scalers=inf_linear_binary_r3.scalers,
                                                          dose_interval=50, return_hist=False,
                                                          burden_noise=0)
            inf_rf_binary_r3_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                      init_burden[pat], timesteps[pat],
                                                      inf_policy_model=inf_rf_binary_r3.q_model,
                                                      prop_model=inf_rf_binary_r3.prop_model,
                                                      model_scalers=inf_rf_binary_r3.scalers,
                                                      dose_interval=50, return_hist=False,
                                                      burden_noise=0)
            inf_sv_binary_r3_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                      init_burden[pat], timesteps[pat],
                                                      inf_policy_model=inf_sv_binary_r3.q_model,
                                                      prop_model=inf_sv_binary_r3.prop_model,
                                                      model_scalers=inf_sv_binary_r3.scalers,
                                                      dose_interval=50, return_hist=False, burden_noise=0)
            if not binary_doses:
                inf_linear_multi_r1_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                             init_burden[pat], timesteps[pat],
                                                             inf_policy_model=inf_linear_multi_r1.q_model,
                                                             prop_model=inf_linear_multi_r1.prop_model,
                                                             model_scalers=inf_linear_multi_r1.scalers,
                                                             dose_interval=multi_interval, return_hist=False,
                                                             burden_noise=0)
                inf_rf_multi_r1_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                         init_burden[pat], timesteps[pat],
                                                         inf_policy_model=inf_rf_multi_r1.q_model,
                                                         prop_model=inf_rf_multi_r1.prop_model,
                                                         model_scalers=inf_rf_multi_r1.scalers,
                                                         dose_interval=multi_interval, return_hist=False,
                                                         burden_noise=0)
                inf_sv_multi_r1_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                         init_burden[pat], timesteps[pat],
                                                         inf_policy_model=inf_sv_multi_r1.q_model,
                                                         prop_model=inf_sv_multi_r1.prop_model,
                                                         model_scalers=inf_sv_multi_r1.scalers,
                                                         dose_interval=multi_interval, return_hist=False, burden_noise=0)
                inf_linear_multi_r2_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                             init_burden[pat], timesteps[pat],
                                                             inf_policy_model=inf_linear_multi_r2.q_model,
                                                             prop_model=inf_linear_multi_r2.prop_model,
                                                             model_scalers=inf_linear_multi_r2.scalers,
                                                             dose_interval=multi_interval, return_hist=False,
                                                             burden_noise=0)
                inf_rf_multi_r2_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                         init_burden[pat], timesteps[pat],
                                                         inf_policy_model=inf_rf_multi_r2.q_model,
                                                         prop_model=inf_rf_multi_r2.prop_model,
                                                         model_scalers=inf_rf_multi_r2.scalers,
                                                         dose_interval=multi_interval, return_hist=False,
                                                         burden_noise=0)
                inf_sv_multi_r2_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                         init_burden[pat], timesteps[pat],
                                                         inf_policy_model=inf_sv_multi_r2.q_model,
                                                         prop_model=inf_sv_multi_r2.prop_model,
                                                         model_scalers=inf_sv_multi_r2.scalers,
                                                         dose_interval=multi_interval, return_hist=False, burden_noise=0)
                inf_linear_multi_r3_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                             init_burden[pat], timesteps[pat],
                                                             inf_policy_model=inf_linear_multi_r3.q_model,
                                                             prop_model=inf_linear_multi_r3.prop_model,
                                                             model_scalers=inf_linear_multi_r3.scalers,
                                                             dose_interval=multi_interval, return_hist=False,
                                                             burden_noise=0)
                inf_rf_multi_r3_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                         init_burden[pat], timesteps[pat],
                                                         inf_policy_model=inf_rf_multi_r3.q_model,
                                                         prop_model=inf_rf_multi_r3.prop_model,
                                                         model_scalers=inf_rf_multi_r3.scalers,
                                                         dose_interval=multi_interval, return_hist=False,
                                                         burden_noise=0)
                inf_sv_multi_r3_outcomes[pat] = simulate(X[pat], alpha[pat], beta[pat], gamma[pat], ed50[pat],
                                                         init_burden[pat], timesteps[pat],
                                                         inf_policy_model=inf_sv_multi_r3.q_model,
                                                         prop_model=inf_sv_multi_r3.prop_model,
                                                         model_scalers=inf_sv_multi_r3.scalers,
                                                         dose_interval=multi_interval, return_hist=False, burden_noise=0)

        inf_linear_binary_r1_outcomes = pd.DataFrame.from_dict([inf_linear_binary_r1_outcomes]).T
        inf_linear_binary_r1_outcomes.columns = ['Inf Linear Binary r1']
        inf_rf_binary_r1_outcomes = pd.DataFrame.from_dict([inf_rf_binary_r1_outcomes]).T
        inf_rf_binary_r1_outcomes.columns = ['Inf RF Binary r1']
        inf_sv_binary_r1_outcomes = pd.DataFrame.from_dict([inf_sv_binary_r1_outcomes]).T
        inf_sv_binary_r1_outcomes.columns = ['Inf SV Binary r1']
        inf_linear_binary_r2_outcomes = pd.DataFrame.from_dict([inf_linear_binary_r2_outcomes]).T
        inf_linear_binary_r2_outcomes.columns = ['Inf Linear Binary r2']
        inf_rf_binary_r2_outcomes = pd.DataFrame.from_dict([inf_rf_binary_r2_outcomes]).T
        inf_rf_binary_r2_outcomes.columns = ['Inf RF Binary r2']
        inf_sv_binary_r2_outcomes = pd.DataFrame.from_dict([inf_sv_binary_r2_outcomes]).T
        inf_sv_binary_r2_outcomes.columns = ['Inf SV Binary r2']
        inf_linear_binary_r3_outcomes = pd.DataFrame.from_dict([inf_linear_binary_r3_outcomes]).T
        inf_linear_binary_r3_outcomes.columns = ['Inf Linear Binary r3']
        inf_rf_binary_r3_outcomes = pd.DataFrame.from_dict([inf_rf_binary_r3_outcomes]).T
        inf_rf_binary_r3_outcomes.columns = ['Inf RF Binary r3']
        inf_sv_binary_r3_outcomes = pd.DataFrame.from_dict([inf_sv_binary_r3_outcomes]).T
        inf_sv_binary_r3_outcomes.columns = ['Inf SV Binary r3']

        inf_outcomes = inf_linear_binary_r1_outcomes.join(inf_rf_binary_r1_outcomes, how='outer')\
            .join(inf_sv_binary_r1_outcomes, how='outer')\
            .join(inf_linear_binary_r2_outcomes, how='outer').join(inf_rf_binary_r2_outcomes, how='outer')\
            .join(inf_sv_binary_r2_outcomes, how='outer').join(inf_linear_binary_r3_outcomes, how='outer')\
            .join(inf_rf_binary_r3_outcomes, how='outer').join(inf_sv_binary_r3_outcomes, how='outer')

        if not binary_doses:
            inf_linear_multi_r1_outcomes = pd.DataFrame.from_dict([inf_linear_multi_r1_outcomes]).T
            inf_linear_multi_r1_outcomes.columns = ['Inf Linear Multi r1']
            inf_rf_multi_r1_outcomes = pd.DataFrame.from_dict([inf_rf_multi_r1_outcomes]).T
            inf_rf_multi_r1_outcomes.columns = ['Inf RF Multi r1']
            inf_sv_multi_r1_outcomes = pd.DataFrame.from_dict([inf_sv_multi_r1_outcomes]).T
            inf_sv_multi_r1_outcomes.columns = ['Inf SV Multi r1']
            inf_linear_multi_r2_outcomes = pd.DataFrame.from_dict([inf_linear_multi_r2_outcomes]).T
            inf_linear_multi_r2_outcomes.columns = ['Inf Linear Multi r2']
            inf_rf_multi_r2_outcomes = pd.DataFrame.from_dict([inf_rf_multi_r2_outcomes]).T
            inf_rf_multi_r2_outcomes.columns = ['Inf RF Multi r2']
            inf_sv_multi_r2_outcomes = pd.DataFrame.from_dict([inf_sv_multi_r2_outcomes]).T
            inf_sv_multi_r2_outcomes.columns = ['Inf SV Multi r2']
            inf_linear_multi_r3_outcomes = pd.DataFrame.from_dict([inf_linear_multi_r3_outcomes]).T
            inf_linear_multi_r3_outcomes.columns = ['Inf Linear Multi r3']
            inf_rf_multi_r3_outcomes = pd.DataFrame.from_dict([inf_rf_multi_r3_outcomes]).T
            inf_rf_multi_r3_outcomes.columns = ['Inf RF Multi r3']
            inf_sv_multi_r3_outcomes = pd.DataFrame.from_dict([inf_sv_multi_r3_outcomes]).T
            inf_sv_multi_r3_outcomes.columns = ['Inf SV Multi r3']
            inf_outcomes = inf_outcomes.join(inf_linear_multi_r1_outcomes, how='outer')\
                .join(inf_rf_multi_r1_outcomes, how='outer').join(inf_sv_multi_r1_outcomes, how='outer')\
                .join(inf_linear_multi_r2_outcomes, how='outer').join(inf_rf_multi_r2_outcomes, how='outer')\
                .join(inf_sv_multi_r2_outcomes, how='outer').join(inf_linear_multi_r3_outcomes, how='outer')\
                .join(inf_rf_multi_r3_outcomes, how='outer').join(inf_sv_multi_r3_outcomes, how='outer')

        inf_outcomes.to_csv(f'{data_save_folder}/sim_{sim_num}_iter_{sim_iter}_inf_outcomes.csv', index=False)
