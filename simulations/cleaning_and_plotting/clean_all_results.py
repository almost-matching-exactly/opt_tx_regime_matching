"""Script to clean results generated from slurm_sim.q script.

Created on October 16, 2023
@author: anonymous
"""

import numpy as np
import os
import pandas as pd
import time

filepath = '{FOLDER WHERE RESULTS ARE SAVED}'  # Folder where results are saved
sims = 32  # Number of simulation setups to look for
iters = 20  # Number of iterations run for each setup
binary_threshold = 3  # Thresholding value for binary outcomes

all_methods = ['Observed', 'Matching', 'Expert', 'q-linear', 'q-rpart', 'Q Linear binary', 'Q RF binary', 'Q SV binary', 'Q Linear Multi',
               'Q RF Multi', 'Q SV Multi', 'optclass-linear', 'optclass-rpart', 'bowl-linear-r1', 'bowl-linear-r2', 'bowl-linear-r3',
               'bowl-poly-r1', 'bowl-poly-r2', 'bowl-poly-r3', 'Inf Linear Binary r1', 'Inf Linear Binary r2', 'Inf Linear Binary r3',
               'Inf RF Binary r1', 'Inf RF Binary r2', 'Inf RF Binary r3', 'Inf SV Binary r1', 'Inf SV Binary r2', 'Inf SV Binary r3',
               'Inf Linear Multi r1', 'Inf Linear Multi r2', 'Inf Linear Multi r3', 'Inf RF Multi r1', 'Inf RF Multi r2', 'Inf RF Multi r3',
               'Inf SV Multi r1', 'Inf SV Multi r2', 'Inf SV Multi r3', 'BCQ_r1', 'BCQ_r2', 'BCQ_r3', 'CQL_r1', 'CQL_r2', 'CQL_r3', 'CRR_r1',
               'CRR_r2', 'CRR_r3', 'DDPG_r1', 'DDPG_r2', 'DDPG_r3', 'SAC_r1', 'SAC_r2', 'SAC_r3', 'TD3_r1', 'TD3_r2', 'TD3_r3', 'Random',
               'Inaction', 'Action']
method_rename = {'Observed': 'Observed', 'Matching': 'Our Method', 'Expert': 'Expert', 'q-linear': 'Linear Q-learning (R)', 'q-rpart': 'DTree Q-learning (R)',
                 'Q Linear binary': 'Linear Q-learning (Python)', 'Q RF binary': 'RF Q-learning (Python)', 'Q SV binary': 'SV Q-learning (Python)',
                 'Q Linear Multi': 'Linear Q-learning Multi', 'Q RF Multi': 'RF Q-learning Multi', 'Q SV Multi': 'SV Q-learning Multi',
                 'optclass-linear': 'Linear OptClass', 'optclass-rpart': 'DTree OptClass', 'bowl-linear-r1': 'Linear BOWL R1',
                 'bowl-linear-r2': 'Linear BOWL R2', 'bowl-linear-r3': 'Linear BOWL R3', 'bowl-poly-r1': 'Poly BOWL R1',
                 'bowl-poly-r2': 'Poly BOWL R2', 'bowl-poly-r3': 'Poly BOWL R3', 'Inf Linear Binary r1': 'Linear Inf R1',
                 'Inf Linear Binary r2': 'Linear Inf R2', 'Inf Linear Binary r3': 'Linear Inf R3', 'Inf RF Binary r1': 'RF Inf R1',
                 'Inf RF Binary r2': 'RF Inf R2', 'Inf RF Binary r3': 'RF Inf R3', 'Inf SV Binary r1': 'SV Inf R1',
                 'Inf SV Binary r2': 'SV Inf R2', 'Inf SV Binary r3': 'SV Inf R3', 'Inf Linear Multi r1': 'Linear Inf Multi R1',
                 'Inf Linear Multi r2': 'Linear Inf Multi R2', 'Inf Linear Multi r3': 'Linear Inf Multi R3', 'Inf RF Multi r1': 'RF Inf Multi R1',
                 'Inf RF Multi r2': 'RF Inf Multi R2', 'Inf RF Multi r3': 'RF Inf Multi R3', 'Inf SV Multi r1': 'SV Inf Multi R1',
                 'Inf SV Multi r2': 'SV Inf Multi R2', 'Inf SV Multi r3': 'SV Inf Multi R3', 'BCQ_r1': 'BCQ R1', 'BCQ_r2': 'BCQ R2',
                 'BCQ_r3': 'BCQ R3', 'CQL_r1': 'CQL R1', 'CQL_r2': 'CQL R2', 'CQL_r3': 'CQL R3', 'CRR_r1': 'CRR R1', 'CRR_r2': 'CRR R2',
                 'CRR_r3': 'CRR R3', 'DDPG_r1': 'DDPG R1', 'DDPG_r2': 'DDPG R2', 'DDPG_r3': 'DDPG R3', 'SAC_r1': 'SAC R1', 'SAC_r2': 'SAC R2',
                 'SAC_r3': 'SAC R3', 'TD3_r1': 'TD3 R1', 'TD3_r2': 'TD3 R2', 'TD3_r3': 'TD3 R3', 'Random': 'Random',
                 'Inaction': 'Inaction', 'Action': 'Full Dosing'}
col_order = ['Sim', 'Iter', 'Covs', 'T Setting', 'T Drop Setting', 'DGP', 'Binary Dose'] + list(method_rename.values())
nan_col_order = [c for c in col_order if c != 'Iter']


def summarize_sim(sim, iters, binary_threshold=3, all_methods=None):
    """Load in all simulation results."""
    all_dfs = []
    all_dfs_binary = []
    for i in range(iters):
        df = []
        for f in os.listdir(filepath):
            if f'sim_{sim}_' in f and f'iter_{i}_' in f and 'outcomes' in f:
                df.append(pd.read_csv(f'{filepath}/{f}'))
        df = pd.concat(df, axis=1)
        df_binary = (df > binary_threshold).astype(int)
        df['Iter'] = i
        df_binary['Iter'] = i
        all_dfs.append(df)
        all_dfs_binary.append(df_binary)

        if len(all_dfs) == 0:
            raise Exception
        df = pd.concat(all_dfs)
        df_binary = pd.concat(all_dfs_binary)
        df_groups = df.groupby('Iter').mean()
        df_bin_groups = df_binary.groupby('Iter').mean()
        df_nans = pd.DataFrame(df_groups.isna().sum(axis=0)[df_groups.isna().sum(axis=0) > 0].sort_values()).T
        if all_methods is not None:
            df_nans[[c for c in all_methods if c not in df.columns]] = iters

    return df_groups, df_bin_groups, df_nans


def add_config(df, filepath, sim):
    """Add in config information for each simulation."""
    with open(f'{filepath}/sim_{sim}_iter_0_config.txt') as f:
        lines = f.readlines()
    df.insert(0, 'Binary Dose', lines[-1].split(':')[2].strip())
    df.insert(0, 'DGP', lines[-1].split(':')[1].split('B')[0].strip())
    df['DGP'] = df['DGP'].apply(lambda x: 'informed' if x == 'semi-random' else x)
    df.insert(0, 'T Drop Setting', 1 if int(lines[3].split(':')[1].replace(']\n', '')[-1]) == 0 else 2)
    df['T Drop Setting'] = df['T Drop Setting'].apply(lambda x: 'a' if int(x) == 1 else 'b')
    df.insert(0, 'T Setting', 1 if int(lines[2].split(':')[1][2]) == 2 else 2)
    df['T Setting'] = df['T Setting'].apply(lambda x: 'a' if int(x) == 1 else 'b')
    df.insert(0, 'Covs', int(lines[1].split(':')[1].replace('\n', '').strip()))
    if df.index.name == 'Iter':
        df = df.reset_index()
    df.insert(0, "Sim", s)
    return df


all_sims = []
all_sims_bin = []
all_sims_nan = []
start = time.time()
for s in range(sims):
    try:
        this_df, this_df_bin, this_df_nan = summarize_sim(s, iters, binary_threshold, all_methods)
        this_df = add_config(this_df, filepath, s)
        this_df_bin = add_config(this_df_bin, filepath, s)
        this_df_nan = add_config(this_df_nan, filepath, s)
        all_sims.append(this_df.copy(deep=True))
        all_sims_bin.append(this_df_bin.copy(deep=True))
        all_sims_nan.append(this_df_nan.copy(deep=True))
        print(f'Sim {s} read: {time.time() - start}')
    except Exception:
        print(f'No results for sim #{s}: {time.time() - start}')

print()
all_sims = pd.concat(all_sims).reset_index(drop=True).rename(columns=method_rename)
for c in col_order:
    if c not in all_sims.columns:
        print(f'No results for {c} for any simulation')
        col_order.remove(c)

all_sims = all_sims[col_order]
all_sims_bin = pd.concat(all_sims_bin).reset_index(drop=True).rename(columns=method_rename)[col_order]
all_sims_nan = pd.concat(all_sims_nan).reset_index(drop=True).rename(columns=method_rename)
for c in nan_col_order:
    if c not in all_sims_nan.columns:
        all_sims_nan[c] = np.nan
all_sims_nan = all_sims_nan[nan_col_order]

all_sims.to_csv(f'{filepath}/all_sims_cont_outcomes.csv', index=False)
all_sims_bin.to_csv(f'{filepath}/all_sims_binary_outcomes.csv', index=False)
all_sims_nan.to_csv(f'{filepath}/all_sims_nan.csv', index=False)

# Make summary files for continuous outcomes
all_sims.drop(columns=['Iter']).groupby(
    ['Sim', 'Covs', 'T Setting', 'T Drop Setting', 'Binary Dose', 'DGP']).mean().to_csv(
    f'{filepath}/sims_cont_outcomes_mean.csv')
all_sims.drop(columns=['Iter']).groupby(
    ['Sim', 'Covs', 'T Setting', 'T Drop Setting', 'Binary Dose', 'DGP']).std().to_csv(
    f'{filepath}/sims_cont_outcomes_std.csv')
all_sims.drop(columns=['Iter']).groupby(
    ['Sim', 'Covs', 'T Setting', 'T Drop Setting', 'Binary Dose', 'DGP']).median().to_csv(
    f'{filepath}/sims_cont_outcomes_median.csv')

# Make summary files for binary outcomes
all_sims_bin.drop(columns=['Iter']).groupby(
    ['Sim', 'Covs', 'T Setting', 'T Drop Setting', 'Binary Dose', 'DGP']).mean().to_csv(
    f'{filepath}/sims_binary_outcomes_mean.csv')
all_sims_bin.drop(columns=['Iter']).groupby(
    ['Sim', 'Covs', 'T Setting', 'T Drop Setting', 'Binary Dose', 'DGP']).std().to_csv(
    f'{filepath}/sims_binary_outcomes_std.csv')
all_sims_bin.drop(columns=['Iter']).groupby(
    ['Sim', 'Covs', 'T Setting', 'T Drop Setting', 'Binary Dose', 'DGP']).median().to_csv(
    f'{filepath}/sims_binary_outcomes_median.csv')

print(f'All results saved to {filepath}')
