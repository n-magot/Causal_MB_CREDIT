'''Only for (X,Z1,Z2) example """

import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
import operator
from jax.scipy.special import expit, logit
from itertools import combinations
import numpy

"""Ordinal Data"""

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

def Generate_Observational_Data(sample_size):

    #beta_X > Z1*X +Z2*X
    beta_X = jnp.array([0.9, 1.2])
    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([0.6, 1.7])

    Z1 = 0 + 1.5 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z2 = 0 + 1 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    data_Z1_Z2 = jnp.concatenate((Z1, Z2), axis=1)

    logits_X = jnp.sum(beta_X * data_Z1_Z2, axis=-1)
    X = dist.Bernoulli(logits=logits_X).sample(random.PRNGKey(numpy.random.randint(1000)))
    X = X.reshape(-1, 1)
    data_X_Z2 = jnp.concatenate((X, Z2), axis=1)
    logit_0 = -4 + jnp.sum(beta_Y * data_X_Z2, axis=-1)
    logit_1 = 4 + jnp.sum(beta_Y * data_X_Z2, axis=-1)
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)  # probability of class 1 or 0
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = jnp.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(1000)), sample_shape=(1,))[0]
    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((X, Z1, Z2), axis=1)

    return data, labels

def Generate_Experimental_Data(sample_size):

    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([0.6, 1.7])

    Z1 = 0 + 15 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    X = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    data_X_Z2 = jnp.concatenate((X, Z2), axis=1)
    logit_0 = -4 + jnp.sum(beta_Y * data_X_Z2, axis=-1)
    logit_1 = 4 + jnp.sum(beta_Y * data_X_Z2, axis=-1)
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)  # probability of class 1 or 0
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = jnp.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(1000)), sample_shape=(1,))[0]

    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((X, Z1, Z2), axis=1)

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

    # print('salala', sum(jnp.exp((dist.OrderedLogistic(predictor=logits, cutpoints=cutpoints).log_prob(labels)))))

    return numpyro.sample('obs', dist.OrderedLogistic(predictor=logits, cutpoints=cutpoints), obs=labels)


def log_likelihood(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['obs']
    return obs_node['fn'].log_prob(obs_node['value'])


data, labels = Generate_Observational_Data(100)

MB_Scores = {}
data_X_Z1 = data[:, [0, 1]]
data_X_Z2 = data[:, [0, 2]]
data_X = data[:, [0]]

"""For (X,Z1,Z2)"""

prior_predictive = Predictive(ordinal_logistic_regression, num_samples=1000)
prior_predictions_coefs = prior_predictive(rng_key_, data, labels, data.shape[1])["coefs"]
prior_predictions_intercept = prior_predictive(rng_key_, data, labels, data.shape[1])["cutpoints"]

"""Gia kathe ena sample apo ta num samples, upologizei to log_likelihood gia ola ta data.
 Posa samples pairnw to elegxo me "prior_predictions_intercept.shape[0]" """
# print(data)
L = 0
P = []

for draw in range(prior_predictions_intercept.shape[0]):
    beta = prior_predictions_coefs[draw]
    alpha = prior_predictions_intercept[draw]

    for i in range(len(data)):

        logit_0 = alpha[0] + jnp.sum(beta * data[i])
        logit_1 = alpha[1] + jnp.sum(beta * data[i])

        prob_0 = expit(logit_0)
        prob_1 = expit(logit_1) - prob_0
        prob_2 = 1 - expit(logit_1)

        if prob_0 == 0:
            prob_0 = 1.e-17
        if prob_1 == 0:
            prob_1 = 1.e-17
        if prob_2 == 0:
            prob_2 = 1.e-17

        # print(prob_0,prob_1, prob_2)
        if labels[i] == 0:
            P.append(numpy.log(prob_0))
        elif labels[i] == 1:
            P.append(numpy.log(prob_1))
        else:
            P.append(numpy.log(prob_2))
    # print(P)
    L = sum(numpy.exp(P))

print(L)

"""For (X,Z1,Z2)"""

"""Gia kathe ena sample apo ta num samples, upologizei to log_likelihood gia ola ta data.
 Posa samples pairnw to elegxo me "prior_predictions_intercept.shape[0]" """

prior_log_likelihood = 0
for sample in range(prior_predictions_intercept.shape[0]):
    prior_samples = {}
    prior_samples['coefs'] = prior_predictions_coefs[sample]
    prior_samples['cutpoints'] = prior_predictions_intercept[sample]
    log_lik_for_a_sample = log_likelihood(random.PRNGKey(12345), prior_samples, ordinal_logistic_regression,
                                          data, labels, data.shape[1])
    # print(sum(jnp.exp((log_lik_for_a_sample))))
    prior_log_likelihood = prior_log_likelihood + sum(jnp.exp((log_lik_for_a_sample)))

print('(X,Z1,Z2)', prior_log_likelihood)
# sample = jnp.array([0, 2, 4])
# draw = jnp.array([0.5, 2, 3])
# print(sample*draw)
