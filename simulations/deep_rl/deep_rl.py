"""Functions to train and evaluate deep RL methods from d3rlpy Python package.

Created on October 16, 2023
@author: anonymous
"""

import d3rlpy
import importlib
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from sim_helpers import parse_rf_data, simulate


def deep_rl_kfold(df_obs_file, df_true_file, model_type='BCQ', reward_function=1, n_steps=10000, n_folds=5,
                  model_save_file=None, outcomes_save_file=None, random_seed=42):
    """Evaluate Deep RL methods from d3rlpy by splitting data into folds."""
    d3rlpy.seed(random_seed)
    train_df = pd.read_csv(df_obs_file)
    eval_df = pd.read_csv(df_true_file)

    i = 0
    outcomes = []
    for s in list(KFold(n_splits=n_folds).split(train_df)):
        print(f'Running fold {i+1} of {n_folds}.')
        m = train_model(train_df.loc[s[0]], model_type=model_type, reward_function=reward_function, n_steps=n_steps)
        if model_save_file is not None:
            m.save(f'{model_save_file}_{model_type}_fold{i}.d3')
            print(f'Model saved to {model_save_file}_{model_type}_fold{i}.d3')
        outcomes.append(eval_model(m, eval_df.loc[s[1]]))
        i += 1

    outcomes = pd.concat(outcomes).sort_index().groupby(level=0).mean()
    outcomes.columns = [f'{model_type}_r{reward_function}']
    if outcomes_save_file is not None:
        outcomes.to_csv(outcomes_save_file, index=False)
        print(f'Outcomes saved to {outcomes_save_file}')
    return outcomes


def train_model(df, model_type, reward_function=1, n_steps=10000):
    """Train d3rlpy model."""
    observations, actions, rewards, terminals = parse_rf_data(df, reward_function)

    # create dataset
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    # create scalers to scale all values
    observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()
    action_scaler = d3rlpy.preprocessing.MinMaxActionScaler()
    reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()

    # create the metric evaluator
    evaluators = {'td_error': d3rlpy.metrics.TDErrorEvaluator(episodes=dataset.episodes)}


    # init model on GPU and with the scalers
    m = getattr(importlib.import_module('d3rlpy.algos'), f'{model_type.upper()}Config')
    m = m(observation_scaler=observation_scaler, action_scaler=action_scaler,
          reward_scaler=reward_scaler).create(device="cuda:0")

    # fit the model
    m.fit(dataset, n_steps=n_steps, evaluators=evaluators)
    return m


def eval_model(m, df):
    """Evaluate d3rlpy model using simulation procedure."""
    x_cols = [c for c in df.columns if 'X' in c]

    opt_outcomes = {}
    for i, row in tqdm(df.iterrows()):
        opt_outcomes[i] = simulate(row[x_cols].to_numpy(), row['alpha'], row['beta'], row['gamma'], row['ed50'],
                                   row['beta'], int(row['timesteps']), policy_model=m, return_hist=False,
                                   burden_noise=0)

    opt_outcomes = pd.DataFrame.from_dict([opt_outcomes]).T
    opt_outcomes.columns = ['Outcomes']

    return opt_outcomes
