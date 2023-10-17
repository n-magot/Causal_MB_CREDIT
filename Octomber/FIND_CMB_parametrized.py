"""Mia asxeti metavliti Z3 kai na ftiaxw ta functions na einai gia kathe eisodo"""


"""den exw valei na vriskei automata ton arithmo ton klaseon kai tin arithmisi sta data, dld [X,Z1,Z2..]-[0,1,2,]"""

import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive, log_likelihood
import operator
from jax.scipy.special import expit, logit
from itertools import combinations
import numpy
import pandas as pd

"""Ordinal Data"""

"""Values that we change in every run:"""
No = 1000
Ne = 1000

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

def Generate_Observational_Data(sample_size):

    #beta_X > Z1*X +Z2*X
    beta_X = jnp.array([1.5, 1.5])
    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([1.4, 1.2])

    e = 0 + 1 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    Z1 = 0 + 15 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z3 = dist.Bernoulli(probs=0.7).sample(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    data_Z1_Z2 = jnp.concatenate((Z1, Z2), axis=1)

    logits_X = jnp.sum(beta_X * data_Z1_Z2, axis=-1)
    X = dist.Bernoulli(logits=logits_X).sample(random.PRNGKey(numpy.random.randint(1000)))
    X = X.reshape(-1, 1)
    data_X_Z2 = jnp.concatenate((X, Z2), axis=1)
    logit_0 = -5 + jnp.sum(beta_Y * data_X_Z2 + e, axis=-1)
    logit_1 = 5 + jnp.sum(beta_Y * data_X_Z2 + e, axis=-1)
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)  # probability of class 1 or 0
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = jnp.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(1000)), sample_shape=(1,))[0]
    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((X, Z1, Z2, Z3), axis=1)

    return data, labels


def Generate_Experimental_Data(sample_size):

    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([1.4, 1.2])

    e = 0 + 1 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    Z1 = 0 + 15 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z3 = dist.Bernoulli(probs=0.7).sample(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    X = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    data_X_Z2 = jnp.concatenate((X, Z2), axis=1)
    logit_0 = -5 + jnp.sum(beta_Y * data_X_Z2 + e, axis=-1)
    logit_1 = 5 + jnp.sum(beta_Y * data_X_Z2 + e, axis=-1)
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)  # probability of class 1 or 0
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = jnp.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(1000)), sample_shape=(1,))[0]

    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((X, Z1, Z2, Z3), axis=1)

    return data, labels

def ordinal_logistic_regression(data, labels, D):
    N_classes = 3
    coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(D), 10*jnp.ones(D)))

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 10).expand([N_classes - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )
    logits = jnp.sum(coefs * data, axis=-1)

    # print('salala', sum((dist.OrderedLogistic(predictor=logits, cutpoints=cutpoints).log_prob(labels))))

    return numpyro.sample('obs', dist.OrderedLogistic(predictor=logits, cutpoints=cutpoints), obs=labels)


def log_likelihood_calc(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['obs']
    return obs_node['fn'].log_prob(obs_node['value'])


def log_predictive_density(rng_key, params, model, *args, **kwargs):
    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(lambda rng_key, params: log_likelihood_calc(rng_key, params, model, *args, **kwargs))
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return jnp.sum(logsumexp(log_lk_vals, 0) - jnp.log(n))


def sample_priors(data):
    prior_predictive = Predictive(ordinal_logistic_regression, num_samples=1000)
    prior_predictions_coefs = prior_predictive(rng_key_, data, labels, data.shape[1])["coefs"]
    prior_predictions_intercept = prior_predictive(rng_key_, data, labels, data.shape[1])["cutpoints"]
    prior_samples = {}
    prior_samples['coefs'] = prior_predictions_coefs
    prior_samples['cutpoints'] = prior_predictions_intercept

    return prior_samples

def sample_posterior(data):
    # Run NUTS.
    num_warmup, num_samples = 1000, 1000
    mcmc = MCMC(NUTS(model=ordinal_logistic_regression), num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(1244), data, labels, data.shape[1])
    posterior_samples = mcmc.get_samples()

    return posterior_samples


def var_combinations(data):

    df = pd.DataFrame(data, columns=[0, 1, 2, 3])
    sample_list = df.columns.values[0:]
    list_comb = []
    for l in range(df.shape[1]-1):
        list_combinations = list(combinations(sample_list, df.shape[1]-l))
        for x in list_combinations:
            if x[0] == 0:
                list_comb.append(x)
    return list_comb


correct_MB = 0
correct_IMB = 0
for i in range(10):

    data, labels = Generate_Observational_Data(No)
    # get unique values and counts of each value
    unique, counts = numpy.unique(labels, return_counts=True)

    # display unique values and counts side by side
    print(numpy.asarray((unique, counts)).T)

    exp_data, exp_labels = Generate_Experimental_Data(Ne)
    # get unique values and counts of each value
    unique, counts = numpy.unique(exp_labels, return_counts=True)

    # display unique values and counts side by side
    print(numpy.asarray((unique, counts)).T)

    MB_Scores = {}
    IMB_Scores_obs = {}
    IMB_Scores_exp = {}

    list_comb = var_combinations(data)
    print(list_comb)

    for comb in range(len(list_comb)):
        reg_variables = list_comb[comb]

        sub_data = data[:, reg_variables]

        prior_samples = sample_priors(sub_data)
        MB_Scores[reg_variables] = log_predictive_density(random.PRNGKey(2), prior_samples, ordinal_logistic_regression,
                                                          sub_data, labels, sub_data.shape[1])

    """Dictionary of marginals from prior sampling"""

    MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
    print(MB_Scores)
    print(MB_Do)
    if MB_Do == (0, 2):
        correct_MB = correct_MB + 1

    sample_list = list(MB_Do)

    """Searching for subsets of MB"""
    subset_list = []
    for s in range(len(sample_list)):
        list_combinations = list(combinations(sample_list, len(sample_list)-s))
        for x in list_combinations:
            if x[0] == 0:
                subset_list.append(x)
    print('The subsets of MB are {}'.format(subset_list))

    """For subsets of MB sample from experimental and observational data"""
    for j in range(len(subset_list)):

        reg_variables = subset_list[j]
        sub_data = data[:, reg_variables]
        exp_sub_data = exp_data[:, reg_variables]

        posterior_sample = sample_posterior(sub_data)

        prior_samples = sample_priors(exp_sub_data)

        marginal_prior = log_predictive_density(random.PRNGKey(2), prior_samples, ordinal_logistic_regression,
                                                exp_sub_data, exp_labels, exp_sub_data.shape[1])
        print('Log_density for {} with prior sampling from exp data:'.format(reg_variables), marginal_prior)
        IMB_Scores_exp[reg_variables] = marginal_prior

        marginal_posterior = log_likelihood(ordinal_logistic_regression, posterior_sample, exp_sub_data,
                                                 exp_labels, exp_sub_data.shape[1])

        marginal_posterior = jnp.sum(logsumexp(marginal_posterior['obs'], 0) - jnp.log(1000))
        print('Log_density for {} with posterior sampling from obs data:'.format(reg_variables), marginal_posterior)
        IMB_Scores_obs[reg_variables] = marginal_posterior

    if IMB_Scores_obs[(0, 2)] > IMB_Scores_exp[(0, 2)] and IMB_Scores_obs[(0,)] < IMB_Scores_exp[(0,)]:
        correct_IMB = correct_IMB + 1

print(correct_MB)
print(correct_IMB)
