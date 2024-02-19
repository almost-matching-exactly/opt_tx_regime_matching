"""Functions to help with simulation.

Created on October 16, 2023
@author: anonymous
"""

import numpy as np

# Policies used to generate observed data when dgp == 'semi-random' (i.e. informed)
cont_aggressive = np.array([10, 10, 20, 20, 20, 20, 0, 0, 20, 0])
cont_moderate = np.array([0, 0, 10, 10, 20, 20, -10, -20, 20, -20])
cont_conservative = np.array([0, 0, 0, 0, 10, 20, -10, -20, 20, -20])
continuous_policies = [cont_aggressive, cont_moderate, cont_conservative]

# Same as above but for when doses are binary (not continuous).
disc_aggressive = np.array([0, 50, 0, 0, 0, 0, 0, 0, 0, 0])
disc_moderate = np.array([0, 0, 0, 0, 50, 0, 0, 0, 0, 0])
disc_conservative = np.array([0, 0, 0, 0, 50, 0, -50, 0, 0, 0])
discrete_policies = [disc_aggressive, disc_moderate, disc_conservative]


def simulate(x, alpha, beta, gamma, ed50, burden, timesteps, policy=None, policy_array=None, preset_doses=None,
             policy_model=None, inf_policy_model=None, prop_model=None, model_scalers=None, binary_doses=False,
             dose_interval=None, burden_noise=2.5, return_hist=False):
    """Simulates outcomes given the PK/PD parameters and the policy."""
    conc = 0  # init concentration of drug is 0
    ea_burdens = np.array([burden])
    concentrations = np.array([conc])
    doses = np.array([])
    all_features = []
    if (policy == 'semi-random') or (policy == 'expert'):
        policy_array = get_preset_policy(x, policy, binary_doses)

    if inf_policy_model is not None:
        x = model_scalers['x'].transform(x.reshape(1, -1))[0]

    for t in range(timesteps):
        if preset_doses is not None:
            if t < preset_doses.shape[0]:
                dose = preset_doses[t]*dose_interval
            elif policy_model is not None:
                n_unique_actions = len(policy_model)
                this_hist = model_scalers[0].transform(np.append(x, ea_burdens[0]).reshape(1, -1))[0]
                for j in range(t-preset_doses.shape[0]+1, t):
                    if n_unique_actions > 2:
                        these_actions = np.zeros(shape=(n_unique_actions-1,))
                        these_actions[int(doses[j]/dose_interval)] += 1
                        these_actions = these_actions[1:]
                    else:
                        these_actions = int(doses[j]/dose_interval)
                    this_hist = np.append(
                        np.append(this_hist, these_actions),
                        model_scalers[j-(t-preset_doses.shape[0])].transform(np.array(ea_burdens[j+1]).reshape(1, -1))[0]
                    )
                this_hist = this_hist.reshape(1, -1)
                if n_unique_actions > 2:
                    action_probas = {}
                    props = policy_model['Prop']['model'].predict_proba(this_hist)[0]
                    for d in policy_model['Prop']['drop_actions']:
                        props = np.insert(props, d, 0)
                    for a, model in policy_model.items():
                        if a != 'Prop':
                            if model is not None:
                                action_probas[a] = model.predict_proba(np.append(this_hist, props[int(a)]).reshape(1, -1))[0][1]
                            else:
                                action_probas[a] = np.inf
                    dose = min(action_probas, key=action_probas.get)*dose_interval
                else:
                    if 'linear' in policy_model['Single'].__module__:
                        control_x = np.concatenate([this_hist, np.zeros(this_hist.shape)], axis=1)
                        treat_x = np.concatenate([this_hist, this_hist], axis=1)
                    else:
                        control_x = np.concatenate([this_hist, np.zeros(shape=(1,1))], axis=1)
                        treat_x = np.concatenate([this_hist, np.ones(shape=(1,1))], axis=1)
                    no_treatment = policy_model['Single'].predict_proba(control_x)[0][1]
                    treatment = policy_model['Single'].predict_proba(treat_x)[0][1]
                    if treatment < no_treatment:
                        dose = dose_interval
                    else:
                        dose = 0
        elif inf_policy_model is not None:
            burden = model_scalers['state'].transform(np.array(burden).reshape(1, -1))[0][0]
            this_hist = np.append(np.array(x), burden).reshape(1, -1)
            if type(inf_policy_model) == list:
                action_probas = {}
                props = prop_model.predict_proba(this_hist)[0]
                for i in range(len(inf_policy_model)):
                    action_probas[i] = inf_policy_model[i].predict(np.append(this_hist, props[i]).reshape(1, -1))[0]
                dose = max(action_probas, key=action_probas.get) * dose_interval
            else:
                if 'linear' in inf_policy_model.__module__:
                    control_x = np.concatenate([this_hist, np.zeros(this_hist.shape)], axis=1)
                    treat_x = np.concatenate([this_hist, this_hist], axis=1)
                else:
                    prop_score = prop_model.predict_proba(this_hist)[0][[1]].reshape(1, 1)
                    control_x = np.concatenate([this_hist, prop_score, np.zeros(shape=(1, 1))], axis=1)
                    treat_x = np.concatenate([this_hist, prop_score, np.ones(shape=(1, 1))], axis=1)
                no_treatment = inf_policy_model.predict(control_x)[0]
                treatment = inf_policy_model.predict(treat_x)[0]
                if treatment > no_treatment:
                    dose = dose_interval
                else:
                    dose = 0

        elif policy_model is not None:
            dose = policy_model.predict(np.append(np.array(x), burden).reshape(1, -1))[0]
        else:
            features = get_features(ea_burdens, doses)
            all_features.append(features.reshape(1, -1))
            if policy == 'random':
                if binary_doses:
                    dose = np.random.choice([0, 50])
                else:
                    dose = np.random.uniform(low=0, high=100)
            elif policy == 'semi-random':
                if binary_doses:
                    if np.random.uniform() < 0.05:
                        dose = np.random.choice([0, 50])
                    else:
                        dose = calc_dose(policy_array, features)
                else:
                    if np.random.uniform() < 0.05:
                        dose = np.random.normal(burden, 10)
                    else:
                        dose = np.random.normal(calc_dose(policy_array, features), 1)
            elif policy == 'expert':
                dose = calc_dose(policy_array, features)
            else:
                if binary_doses:
                    dose = int((1 / (1 + np.exp(-calc_dose(policy_array, features)))) >0.5)*50
                else:
                    dose = calc_dose(policy_array, features)

        dose, conc, burden = sim_step(dose, conc, gamma, alpha, beta, ed50, burden_noise)
        doses = np.append(doses, dose)
        concentrations = np.append(concentrations, conc)
        ea_burdens = np.append(ea_burdens, burden)

    if return_hist:
        if len(all_features) > 0:
            all_features = np.concatenate(all_features, axis=0)
        ea_outcome, conc_outcome, = calc_outcome(x, ea_burdens, concentrations, parts=True)
        return ea_outcome + conc_outcome, ea_burdens, doses, concentrations, all_features, \
            policy_array, ea_outcome, conc_outcome
    return calc_outcome(x, ea_burdens, concentrations)


def sim_step(dose, conc, gamma, alpha, beta, ed50, burden_noise):
    """One step of the PK/PD progression"""
    dose = np.round(np.clip(dose, 0, 100), decimals=2)
    conc = (np.exp(-gamma) * conc) + dose
    burden = np.clip(
        beta * (1 - (conc ** alpha) / ((conc ** alpha) + (ed50 ** alpha))) + np.random.normal(0, burden_noise),
        a_min=0, a_max=beta)
    return dose, conc, burden


def get_features(burdens, doses):
    """Gets feature values to use for informed policy and our method."""
    one_hr_burden = 0
    three_hr_burden = 0
    one_hr_drug = 0
    three_hr_drug = 0
    if len(burdens) >= 1:
        one_hr_burden = burdens[-1]
        if len(burdens) >= 3:
            three_hr_burden = burdens[-3:].mean()
    if len(doses) >= 1:
        one_hr_drug = doses[-1]
        if len(doses) >= 3:
            three_hr_drug = doses[-3:].mean()

    # calculate binary features
    features = np.array([
        one_hr_burden > 10,
        one_hr_burden > 20,
        one_hr_burden > 30,
        one_hr_burden > 40,
        one_hr_burden > 60,
        one_hr_burden > 80,
        one_hr_drug > 25,
        one_hr_drug > 50,
        (one_hr_burden > 40) & (three_hr_burden > 20),
        (one_hr_drug > 40) & (three_hr_drug > 20)
    ]).astype(int)
    return features


def calc_outcome(x, burdens, concentrations, parts=False):
    """Calculates the continuous outcome."""
    ea_cost = np.exp(x[:2].mean())*np.sum(np.exp(burdens[1:] / 50) - 1) / (len(burdens)-1)
    conc_cost = np.exp(x[2:4].mean()) * np.sum(np.exp(concentrations[1:] / 50) - 1) / (len(concentrations)-1)
    if parts:
        return ea_cost, conc_cost
    return ea_cost + conc_cost


def get_preset_policy(x, policy, discrete=False):
    """When generating observed data with an informed policy, assign a policy type to the patient."""
    if discrete:
        policies = discrete_policies
    else:
        policies = continuous_policies
    if np.mean(x[:2]) < 0 and np.mean(x[2:4]) < 0:
        if policy == 'semi-random':
            probs = [0.1, 0.8, 0.1]
        elif policy == 'expert':
            probs = [0.0, 1.0, 0.0]
    elif np.mean(x[:2]) > np.mean(x[2:4]):
        if policy == 'semi-random':
            probs = [0.8, 0.1, 0.1]
        elif policy == 'expert':
            probs = [1.0, 0.0, 0.0]
    else:
        if policy == 'semi-random':
            probs = [0.1, 0.1, 0.8]
        elif policy == 'expert':
            probs = [0.0, 0.0, 1.0]

    policy_prob = np.random.uniform()
    i = 0
    running_prob = probs[i]
    while policy_prob > running_prob:
        i += 1
        running_prob += probs[i]
    if policy == 'semi-random' and not discrete_policies:
        return np.array(policies[i]) + np.random.normal(0, 1, size=(10,))
    return np.array(policies[i])


def calc_dose(policy, features):
    """Calc the administered dose given the policy and feature values."""
    return np.sum(policy*features)


def parse_rf_data(df, reward_function=1):
    """Organize the data into a format that can be passed to d3rlpy deep RL methods."""
    x = df[[c for c in df.columns if 'X' in c]].values.tolist()
    burdens = df[[c for c in df.columns if 'Burden' in c]].values.tolist()
    burdens = [[c for c in b if str(c) != 'nan'] for b in burdens]
    doses = df[[c for c in df.columns if 'Dose' in c]].values.tolist()
    doses = [[c for c in d if str(c) != 'nan'] for d in doses]
    concentrations = df[[c for c in df.columns if 'Conc' in c]].values.tolist()
    concentrations = [[c for c in d if str(c) != 'nan'] for d in concentrations]

    obs = []
    actions = []
    rewards = []
    terminals = np.array([])
    for pat in range(len(x)):
        for d in range(len(doses[pat])):
            obs.append(np.append(x[pat], burdens[pat][d]))
            actions.append(doses[pat][d])
            if reward_function == 1:
                rewards.append((burdens[pat][d] - burdens[pat][d + 1]) + (0.25*(50 - doses[pat][d])))
            elif reward_function == 2:
                rewards.append(-((burdens[pat][d+1]*np.exp(x[pat][0])) + (doses[pat][d]*np.exp(x[pat][2]))))
            elif reward_function == 3:
                rewards.append(-(((np.exp(burdens[pat][d+1] / 50) - 1)*np.exp(np.mean(x[pat][:2]))) +
                                 ((np.exp(concentrations[pat][d+1] / 50) - 1)*np.exp(np.mean(x[pat][2:4])))))
        terminals = np.append(np.append(terminals, np.zeros(len(doses[pat]) - 1)), 1)
    obs = np.array(obs)
    actions = np.array(actions).reshape(-1, 1)
    rewards = np.array(rewards)
    return obs, actions, rewards, terminals
