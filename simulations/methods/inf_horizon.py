"""Class for InfHorizon: Uses Fitted Q-iteration to estimate optimal treatment regimes.

Created on October 16, 2023
@author: anonymous
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class InfHorizon:
    def __init__(self, X, states, actions, concentrations=None, method='linear', treatment_bins=None, reward_function=1, tol=1e-3, random_seed=0):
        self.n_samples = len(actions)
        if reward_function == 1:
            self.rewards = np.concatenate([-np.diff(states[i]) + (0.25*(50 - actions[i])) for i in range(self.n_samples)])
        elif reward_function == 2:
            self.rewards = -np.concatenate(
                [(states[i][1:]*np.exp(X[i, 0])) + (actions[i]*np.exp(X[i, 2])) for i in range(self.n_samples)])
        elif reward_function == 3:
            self.rewards = -np.concatenate(
                [((np.exp(states[i][1:] / 50) - 1) *
                  np.exp(X[i, :2].mean())) + ((np.exp(concentrations[i][1:] / 50) - 1) *
                                              np.exp(X[i, 2:4].mean())) for i in range(self.n_samples)]
            )

        X = np.concatenate([np.repeat(X[[i]], states[i].shape[0]-1, axis=0) for i in range(self.n_samples)])
        states = np.concatenate([s[:-1] for s in states]).reshape(-1, 1)
        x_scaler = StandardScaler().fit(X)
        self.scalers = {'x': x_scaler}
        state_scaler = StandardScaler().fit(states)
        self.scalers['state'] = state_scaler
        self.hists = np.concatenate([x_scaler.transform(X), state_scaler.transform(states)], axis=1)
        actions = pd.DataFrame(np.concatenate(actions))
        self.actions = actions.apply(pd.cut, bins=[-np.inf, *treatment_bins, np.inf],
                                     labels=list(range(len(treatment_bins)+1))).to_numpy()
        self.unique_actions = np.sort(np.unique(self.actions))
        self.n_unique_actions = len(self.unique_actions)
        self.method = method
        self.n_treatments = len(treatment_bins) + 1
        self.n_obs = self.rewards.shape[0]
        self.tol = tol
        self.random_seed = random_seed
        self.q_model = None
        self.prop_model = None
        self.prop_vals = None
        if (method != 'linear') or self.n_treatments > 2:
            self.prop_model = LogisticRegressionCV(max_iter=5000, random_state=self.random_seed).fit(self.hists, self.actions)
            self.prop_vals = self.prop_model.predict_proba(self.hists)

    def fit(self, max_iters=100):
        self.fit_value_model()
        q_vals, opt_actions = self.get_q_vals()
        diff = np.inf
        iter = 0
        while diff > self.tol:
            self.fit_value_model(q_vals=q_vals)
            q_vals, these_opt_actions = self.get_q_vals()
            diff = 1 - (((opt_actions == these_opt_actions).sum()) /self.n_obs)
            opt_actions = these_opt_actions
            iter += 1
            if iter > max_iters:
                break

    def fit_value_model(self, q_vals=None):
        rewards = np.array(self.rewards)
        if q_vals is not None:
            rewards += q_vals
        hist_act = self.get_hist_act_vector()
        if self.n_treatments == 2:
            if self.method == 'linear':
                self.q_model = RidgeCV().fit(hist_act, rewards)
            elif self.method == 'tree':
                self.q_model = DecisionTreeRegressor(max_depth=4, random_state=self.random_seed).fit(hist_act, rewards)
            elif self.method == 'RF':
                self.q_model = RandomForestRegressor(n_estimators=20, max_depth=4,
                                                     random_state=self.random_seed).fit(hist_act, rewards)
            elif self.method == 'SV':
                self.q_model = SVR().fit(hist_act, rewards)
        else:
            self.q_model = []
            for i in range(self.n_treatments):
                these_rewards = rewards[np.ravel(self.actions == i)]
                if self.method == 'linear':
                    self.q_model.append(RidgeCV().fit(hist_act[i], these_rewards))
                elif self.method == 'tree':
                    self.q_model.append(DecisionTreeRegressor(max_depth=4, random_state=self.random_seed).fit(hist_act[i],
                                                                                                         these_rewards))
                elif self.method == 'RF':
                    self.q_model.append(RandomForestRegressor(n_estimators=20, max_depth=4,
                                                              random_state=self.random_seed).fit(hist_act[i],
                                                                                                 these_rewards))
                elif self.method == 'SV':
                    self.q_model.append(SVR().fit(hist_act[i], these_rewards))

    def get_dummy_act_vector(self):
        hists = []
        for i in range(self.n_treatments):
            if self.method == 'linear' and self.n_treatments == 2:
                hists.append(np.concatenate([self.hists, self.hists * i], axis=1))
            elif self.n_treatments == 2:
                hists.append(np.concatenate([self.hists,
                                             self.prop_vals[:, [1]],
                                             np.ones(shape=(self.n_obs, 1)) * i], axis=1))
            else:
                hists.append(np.concatenate([self.hists, self.prop_vals[:, [i]]], axis=1))
        return hists

    def get_hist_act_vector(self):
        if self.n_treatments == 2:
            if self.method == 'linear':
                return np.concatenate([self.hists, self.hists * self.actions], axis=1)
            else:
                return np.concatenate([self.hists, self.prop_vals[:, [1]], self.actions], axis=1)
        else:
            hists = []
            for i in range(self.n_treatments):
                hists.append(np.concatenate([self.hists[np.ravel(self.actions == i), :],
                                             self.prop_vals[np.ravel(self.actions == i), [i]].reshape(-1, 1)],
                                            axis=1))
            return hists

    def get_q_vals(self):
        hist_act = self.get_dummy_act_vector()
        if self.n_treatments == 2:
            q_vals = [self.q_model.predict(hist_act[0]).reshape(-1, 1),
                      self.q_model.predict(hist_act[1]).reshape(-1, 1)]
        else:
            q_vals = []
            for i in range(self.n_treatments):
                q_vals.append(self.q_model[i].predict(hist_act[i]).reshape(-1, 1))
        q_vals = np.concatenate(q_vals, axis=1)
        opt_actions = np.argmax(q_vals, axis=1)
        return np.amax(q_vals, axis=1), opt_actions
