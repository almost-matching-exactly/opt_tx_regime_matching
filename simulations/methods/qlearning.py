"""Class for QLearner: Uses finite-timestep backward induction Q-learning to estimate optimal treatment regimes.

Created on October 16, 2023
@author: anonymous
"""
import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler


class QLearner:
    def __init__(self, X, states, actions, outcomes, method='linear', treatment_bins=None, random_seed=0):
        self.scalers = []
        states = pd.DataFrame(states).dropna(axis=1)
        self.actions = pd.DataFrame(actions).dropna(axis=1)
        self.n_samples, self.n_timesteps = self.actions.shape
        if treatment_bins:
            self.actions = self.actions.apply(pd.cut,
                                              bins=[-np.inf, *treatment_bins, np.inf],
                                              labels=list(range(len(treatment_bins)+1))
                                              )
        self.unique_actions = np.sort(np.unique(self.actions.to_numpy()))
        self.n_unique_actions = len(self.unique_actions)
        self.outcomes = outcomes
        self.method = method
        rolling_hist = np.concatenate([X, states.loc[:, [0]].to_numpy()], axis=1)
        self.scalers.append(StandardScaler().fit(rolling_hist))
        rolling_hist = self.scalers[0].transform(rolling_hist)
        self.hist = [np.array(rolling_hist)]
        for t in range(self.n_timesteps-1):
            self.scalers.append(StandardScaler().fit(states.loc[:, [t+1]].to_numpy()))
            these_actions = self.build_action_vector(t)
            rolling_hist = np.concatenate(
                [rolling_hist,
                 these_actions,
                 self.scalers[t+1].transform(states.loc[:, [t+1]].to_numpy())],
                axis=1)
            self.hist.append(np.array(rolling_hist))

        self.q_functions = []
        self.opt_policy = np.empty([self.n_samples, 0])
        self.random_seed = random_seed

    def fit(self):
        opt_value = self.outcomes
        for t in range(self.n_timesteps-1, -1, -1):
            drop_actions = [i for i, v in (self.actions.loc[:, t].value_counts() < 10).to_dict().items() if v]
            mm = LogisticRegressionCV(max_iter=5000, random_state=self.random_seed)
            mm.fit(self.hist[t][~self.actions.loc[:, t].isin(drop_actions), :],
                   self.actions.loc[~self.actions.loc[:, t].isin(drop_actions), t].to_numpy())
            props = mm.predict_proba(self.hist[t])
            for d in drop_actions:
                props = np.insert(props, [d], np.zeros(shape=(self.n_samples, 1)), axis=1)
            vals = []
            for a in self.unique_actions:
                if a not in drop_actions:
                    m, model_type = self.get_value_model(t == self.n_timesteps-1)
                    m = m.fit(
                        np.concatenate([self.hist[t][self.actions.loc[:, t] == a, :],
                                        np.expand_dims(props[self.actions.loc[:, t] == a, a], axis=1)], axis=1),
                        opt_value[self.actions.loc[:, t] == a]
                    )
                    if model_type == 'regression':
                        vals.append(m.predict(np.concatenate([self.hist[t], np.expand_dims(props[:, a], axis=1)], axis=1)).reshape(-1, 1))
                    elif model_type == 'classification':
                        vals.append(m.predict_proba(np.concatenate([self.hist[t], np.expand_dims(props[:, a], axis=1)], axis=1))[:, 1].reshape(-1, 1))
                else:
                    vals.append(np.ones(shape=(self.n_samples, 1))*np.inf)
            vals = np.concatenate(vals, axis=1)
            self.opt_policy = np.concatenate(
                [np.argmin(vals, axis=1).reshape(-1, 1),
                 self.opt_policy],
                axis=1)
            opt_value = np.min(vals, axis=1)

    def get_value_model(self, last_step):
        if last_step:
            if self.method == 'linear':
                value_model = LogisticRegressionCV(max_iter=5000, random_state=self.random_seed)
            elif self.method == 'RF':
                value_model = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=self.random_seed)
            elif self.method == 'SV':
                value_model = SVC(probability=True)
            model_type = 'classification'
        else:
            if self.method == 'linear':
                value_model = RidgeCV()
            elif self.method == 'RF':
                value_model = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=self.random_seed)
            elif self.method == 'SV':
                value_model = SVR()
            model_type = 'regression'
        return value_model, model_type

    def get_final_policy_model(self):
        final_models = dict()
        drop_actions = [i for i, v in
                        (self.actions.loc[:, self.n_timesteps-1].value_counts() < 10).to_dict().items() if v]
        mm = LogisticRegressionCV(max_iter=5000, random_state=self.random_seed, multi_class="multinomial").fit(
            self.hist[self.n_timesteps-1][~self.actions.loc[:, self.n_timesteps-1].isin(drop_actions), :],
            self.actions.loc[~self.actions.loc[:, self.n_timesteps-1].isin(drop_actions), self.n_timesteps-1].to_numpy())
        final_models['Prop'] = {'model': mm, 'drop_actions': drop_actions}
        props = mm.predict_proba(self.hist[self.n_timesteps-1])
        for d in drop_actions:
            props = np.insert(props, [d], np.zeros(shape=(self.n_samples, 1)), axis=1)
        for a in self.unique_actions:
            these_outcomes = self.outcomes[self.actions.loc[:, self.n_timesteps-1] == a]
            if these_outcomes.shape[0] >= 10:
                m, _ = self.get_value_model(True)

                m = m.fit(
                    np.concatenate(
                        [self.hist[self.n_timesteps - 1][self.actions.loc[:, self.n_timesteps - 1] == a, :],
                         np.expand_dims(props[self.actions.loc[:, self.n_timesteps - 1] == a, a], axis=1)], axis=1),
                    these_outcomes
                )
                final_models[a] = copy.deepcopy(m)
            else:
                final_models[a] = None
        return final_models, self.scalers

    def build_action_vector(self, t):
        if self.n_unique_actions > 2:
            these_actions = np.zeros(shape=(self.n_samples, self.n_unique_actions))
            these_actions[np.arange(self.n_samples), self.actions.loc[:, t].to_numpy()] += 1
            return these_actions[:, 1:]
        else:
            return self.actions.loc[:, [t]].to_numpy()
