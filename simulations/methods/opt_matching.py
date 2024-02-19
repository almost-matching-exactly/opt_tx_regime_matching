"""Class for OptMaTx: A matching method for optimal treatment regime estimation.

Created on October 16, 2023
@author: anonymous
"""
import numpy as np
import pandas as pd
import sklearn.ensemble as en
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


class OptMaTx:
    """
    Learns a distance metric and uses the learned distance metric to match samples together. Then,
    estimates optimal treatment regimes by interpolating over the samples with good outcomes in
    each matched group.

    Parameters
    ----------
    binary_dose : bool, default=False
        Indicates whether the treatment is binary or not.
    random_seed : None or int, default=None
        Random state to run on.

    Attributes
    -------
    scaler : sklearn.MinMaxScaler
        Fit to training data when set_dist_metric() is called. Used to scale training and estimation data.
    dist_metric : None or np.array
        Initialized as None and assigned to numpy array of learned distance metric when set_dist_metric()
        is called.
    binary_outcome : bool
    random_seed : None or int
    """
    def __init__(self, binary_doses=False, random_seed=None):
        self.scaler = MinMaxScaler()
        self.dist_metric = None
        self.binary_outcome = binary_doses
        self.random_seed = random_seed

    def set_dist_metric(self, X, outcomes, method='linear'):
        """
        Calculate variable importances to use for the distance metric. Assigns distance metric to
        self.dist_metric. Also fits self.scaler to training data.

        Parameters
        ----------
        X : np.array
            n x p numpy array of n samples and p pre-treatment covariates.
        Y : np.array
            outcome values corresponding to the samples.
        method : str, default='linear'
            Denotes which machine learning method to use to learn the feature importances for the distance metric.
        """
        x_normal = pd.DataFrame(self.scaler.fit_transform(X))
        y = pd.DataFrame(outcomes)

        if method == 'linear':
            m_dist_metric = lm.LogisticRegressionCV(penalty='l2').fit(x_normal, y.values.ravel())
            self.dist_metric = (np.abs(m_dist_metric.coef_) / np.abs(m_dist_metric.coef_).max())
        elif method == 'boosting':
            m_dist_metric = en.GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=self.random_seed).fit(
                x_normal, y.values.ravel()
            )
            self.dist_metric = (
                m_dist_metric.feature_importances_ / m_dist_metric.feature_importances_.max()
            )

    def get_opt_policy(self, X, outcomes, features, doses, n_neighbors=10):
        """Create match groups using the learned distance metric (if available, else use standard
        euclidean distance). Then calculate the optimal policy for each sample via a linear
        interpolation over the policies of the matched samples who had a good outcome.

        Parameters
        ----------
        X : np.array
            n x p numpy array of n samples and p pre-treatment covariates. p must be the same dimension as
            the X passed to get_dist_metric() if that method was called first.
        outcomes : np.array
            outcome values corresponding to the samples.
        features : list(np.array)
            list of length n, where each entry corresponds to a T_i x f dimensional np.array where T_i is the
            number of observed timesteps for sample i and f is the dimension of the additive policy model space.
        doses : list(np.array)
            list of length n, where each entry corresponds to a T_i dimensional np.array where T_i is the
            number of observed timesteps for sample i and the values are the action value taken on sample i
            at each timestep.
        n_neighbors : int, default=10
            number of neighbors with good outcomes to match to.
        Returns
        ------
        optimal policies : pd.DataFrame
            dataframe with the estimated optimal policy for each sample in the passed estimation set.
        """
        x_normal = pd.DataFrame(self.scaler.transform(X))
        y = pd.DataFrame(outcomes)
        y.columns = ['Y']
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(
            x_normal.to_numpy()[~y.to_numpy().reshape(-1,)] * self.dist_metric)
        neighbors = nn.kneighbors(x_normal.to_numpy() * self.dist_metric, return_distance=False)
        MG = np.zeros(shape=(X.shape[0], X.shape[0]))
        match_idxs = y.index[~y.to_numpy().reshape(-1, )].to_numpy()
        if n_neighbors > 15:
            for i in range(X.shape[0]):
                MG[i][np.random.choice(match_idxs, replace=False, size=(5,))] = 1
        else:
            for i in range(X.shape[0]):
                MG[i][match_idxs[neighbors[i]]] = 1

        policy_est = []
        if self.binary_outcome:
            min_dose = np.min(np.concatenate(doses))
            doses = [np.array([-1 if d == min_dose else 1 for d in pd]) for pd in doses]
        for pat in range(X.shape[0]):
           policy_m = lm.RidgeCV(fit_intercept=False).fit(features[pat], doses[pat])
           policy_est.append(policy_m.coef_.reshape(-1,))
        policy_est = pd.DataFrame(policy_est)
        T = policy_est.round(1)
        opt_matching_policy = pd.concat(
            [
                T[
                    MG[i] > 0
                ]  # get treatment assignment for all units in the matched group of unit i
                .join(
                    y[MG[i] > 0]
                )  # join outcomes for all units in the MG of i with treatments
                .groupby("Y")  # group by outcomes, here the outcome is binary
                .mean()  # get the average treatment for each outcome
                .iloc[[0]]  # choose the treatment with the minimum outcome
                for i in range(MG.shape[0])
            ],
            axis=0,
        ).reset_index()

        return opt_matching_policy[~opt_matching_policy['Y']].drop(columns=['Y'])


def opt_matx_kfold(X, outcomes, features, doses, n_folds=10, n_neighbors=10, binary_doses=False, random_seed=0):
    """Splits dataset into K folds to learn optimal treatment regimes for all samples in a dataset.

    Parameters
    ----------
    X : np.array
        n x p matrix with n samples and p pre-treatment covariates
    outcomes : np.array
        outcomes corresponding to samples
    features : list(np.array)
        list of length n, where each entry corresponds to a T_i x f dimensional np.array where T_i is the
        number of observed timesteps for sample i and f is the dimension of the additive policy model space.
    doses : list(np.array)
        list of length n, where each entry corresponds to a T_i dimensional np.array where T_i is the
        number of observed timesteps for sample i and the values are the action value taken on sample i
        at each timestep.
    n_fold : int, default=10
        number of folds to split data into. One fold used to learn distance metric and remaining folds used
        in estimation set to estimate optimal policies.
    n_neighbors : int, default=10
        number of neighbors with good outcomes to match to.
    binary_dose : bool, default=False
        Indicates whether the treatment is binary or not.
    random_seed : None or int, default=None
        Random state to run on.

    Returns
    -------
    optimal policies : pd.DataFrame
        dataframe with the estimated optimal policy for each sample in the passed dataset.
    """
    opt_policies = []
    for s in list(KFold(n_splits=n_folds).split(X)):
        m = OptMaTx(binary_doses=binary_doses, random_seed=random_seed)
        m.set_dist_metric(X[s[1], :], outcomes[s[1]])
        these_policies = m.get_opt_policy(X[s[0], :], outcomes[s[0]], [features[i] for i in s[0]],
                                          [doses[i] for i in s[0]], n_neighbors=n_neighbors)
        these_policies.index = s[0]
        opt_policies.append(these_policies.copy(deep=True))
    return pd.concat(opt_policies).sort_index().groupby(level=0).mean()
