"""To eftiaxa prosarmosmeno sto tutorial tou NumPyro gia to Posterior predictive density kai
eftiaxa ena dictionary pou exei ta prior stamples, des
- url : https://num.pyro.ai/en/stable/tutorials/bayesian_regression.html?#Model-Predictive-Density
 Ta supersos url pou me voithisan na katalavw pws douleuei h katastasi htan
 - 1. url: https://forum.pyro.ai/t/how-to-define-a-likelihood-function-in-numpyro/2968/5
 - 2. url: https://forum.pyro.ai/t/sampling-from-model-prior-in-numpyro/2414
 - 3. url: https://forum.pyro.ai/t/prior-predictive-density/5082"""
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
import math

"""Ordinal Data"""

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

def Generate_Observational_Data(sample_size):

    #beta_X > Z1*X +Z2*X
    beta_X = jnp.array([0.9, 1.2])
    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([1.4, 0.7])

    Z1 = 0 + 15 * random.normal(random.PRNGKey(4543), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey(52234), (sample_size, 1))
    data_Z1_Z2 = jnp.concatenate((Z1, Z2), axis=1)

    logits_X = jnp.sum(beta_X * data_Z1_Z2, axis=-1)
    X = dist.Bernoulli(logits=logits_X).sample(random.PRNGKey(1744))
    X = X.reshape(-1, 1)
    data_X_Z2 = jnp.concatenate((X, Z2), axis=1)
    logit_0 = -4 - jnp.sum(beta_Y * data_X_Z2, axis=-1)
    logit_1 = 4 - jnp.sum(beta_Y * data_X_Z2, axis=-1)
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)  # probability of class 1 or 0
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = jnp.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(1345), sample_shape=(1,))[0]

    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((X, Z1, Z2), axis=1)

    return data, labels

data, labels = Generate_Observational_Data(500)

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
def log_likelihood(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['obs']
    return obs_node['fn'].log_prob(obs_node['value'])

def log_predictive_density(rng_key, params, model, *args, **kwargs):
    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(lambda rng_key, params: log_likelihood(rng_key, params, model, *args, **kwargs))
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return jnp.sum(logsumexp(log_lk_vals, 0) - jnp.log(n))


MB_Scores = {}
data_X_Z1 = data[:, [0, 1]]
data_X_Z2 = data[:, [0, 2]]
data_X = data[:, [0]]
"""For (X,Z1,Z2)"""

prior_predictive = Predictive(ordinal_logistic_regression, num_samples=10000)
prior_predictions_coefs = prior_predictive(rng_key_, data, labels, data.shape[1])["coefs"]
prior_predictions_intercept = prior_predictive(rng_key_, data, labels, data.shape[1])["cutpoints"]
dokimi_dict = {}
dokimi_dict['coefs'] = prior_predictions_coefs
dokimi_dict['cutpoints'] = prior_predictions_intercept

MB_Scores['(X,Z1,Z2)'] = log_predictive_density(random.PRNGKey(2), dokimi_dict,
      ordinal_logistic_regression, data, labels, data.shape[1])
print(log_predictive_density(random.PRNGKey(2), dokimi_dict,
      ordinal_logistic_regression, data, labels, data.shape[1]))

"""For (X,Z1)"""
coefs_X_Z1 = prior_predictions_coefs[:, [0, 1]]
cutpoints_X_Z1 = prior_predictions_intercept
dokimi_dict = {}
dokimi_dict['coefs'] = coefs_X_Z1
dokimi_dict['cutpoints'] = cutpoints_X_Z1

MB_Scores['(X,Z1)'] = log_predictive_density(random.PRNGKey(2), dokimi_dict,
      ordinal_logistic_regression, data_X_Z1, labels, data_X_Z1.shape[1])
print(log_predictive_density(random.PRNGKey(2), dokimi_dict,
      ordinal_logistic_regression, data_X_Z1, labels, data_X_Z1.shape[1]))

"""For (X,Z2)"""
coefs_X_Z2 = prior_predictions_coefs[:, [0, 2]]
cutpoints_X_Z2 = prior_predictions_intercept
dokimi_dict = {}
dokimi_dict['coefs'] = coefs_X_Z2
dokimi_dict['cutpoints'] = cutpoints_X_Z2

MB_Scores['(X,Z2)'] = log_predictive_density(random.PRNGKey(2), dokimi_dict,
      ordinal_logistic_regression, data_X_Z2, labels, data_X_Z2.shape[1])
print(log_predictive_density(random.PRNGKey(2), dokimi_dict,
      ordinal_logistic_regression, data_X_Z2, labels, data_X_Z2.shape[1]))

"""For (X)"""
coefs_X = prior_predictions_coefs[:, [0]]
cutpoints_X = prior_predictions_intercept
dokimi_dict = {}
dokimi_dict['coefs'] = coefs_X
dokimi_dict['cutpoints'] = cutpoints_X

MB_Scores['(X)'] = log_predictive_density(random.PRNGKey(2), dokimi_dict,
      ordinal_logistic_regression, data_X, labels, data_X.shape[1])
print(log_predictive_density(random.PRNGKey(2), dokimi_dict,
      ordinal_logistic_regression, data_X, labels, data_X.shape[1]))


MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
print(MB_Do)
