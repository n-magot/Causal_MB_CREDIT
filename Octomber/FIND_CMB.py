"""To eftiaxa prosarmosmeno sto tutorial tou NumPyro gia to Posterior predictive density kai
eftiaxa ena dictionary pou exei ta prior stamples, des
- url : https://num.pyro.ai/en/stable/tutorials/bayesian_regression.html?#Model-Predictive-Density
 Ta supersos url pou me voithisan na katalavw pws douleuei h katastasi htan
 - 1. url: https://forum.pyro.ai/t/how-to-define-a-likelihood-function-in-numpyro/2968/5
 - 2. url: https://forum.pyro.ai/t/sampling-from-model-prior-in-numpyro/2414
 - 3. url: https://forum.pyro.ai/t/prior-predictive-density/5082"""

"""Mexri stigmis auto nomizw pws einai to pio swsto 26/09/2023"""

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

"""Ordinal Data"""

"""Values that we change in every run:"""
No = 1000
Ne = 300

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

def Generate_Observational_Data(sample_size):

    #beta_X > Z1*X +Z2*X
    beta_X = jnp.array([1.3, 1.25])
    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([1.4, 1.15])

    Z1 = 0 + 15 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
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
    beta_Y = jnp.array([1.4, 1.15])

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


correct_MB = 0
correct_IMB = 0
for i in range(40):

    data, labels = Generate_Observational_Data(No)

    MB_Scores = {}
    data_X_Z1 = data[:, [0, 1]]
    data_X_Z2 = data[:, [0, 2]]
    data_X = data[:, [0]]

    """For (X,Z1,Z2)"""

    prior_samples = sample_priors(data)

    MB_Scores['(X,Z1,Z2)'] = log_predictive_density(random.PRNGKey(2), prior_samples, ordinal_logistic_regression,
                                                    data, labels, data.shape[1])

    """Posterior sampling for (X,Z1,Z2)"""

    posterior_sample_X_Z1_Z2 = sample_posterior(data)

    """For (X,Z1)"""

    prior_samples = sample_priors(data_X_Z1)

    MB_Scores['(X,Z1)'] = log_predictive_density(random.PRNGKey(2), prior_samples, ordinal_logistic_regression,
                                                 data_X_Z1, labels, data_X_Z1.shape[1])

    """For (X,Z2)"""

    prior_samples = sample_priors(data_X_Z2)

    MB_Scores['(X,Z2)'] = log_predictive_density(random.PRNGKey(2), prior_samples, ordinal_logistic_regression,
                                                 data_X_Z2, labels, data_X_Z2.shape[1])

    """Posterior sampling for (X,Z2)"""

    posterior_sample_X_Z2 = sample_posterior(data_X_Z2)

    """For (X)"""

    prior_samples = sample_priors(data_X)

    MB_Scores['(X)'] = log_predictive_density(random.PRNGKey(2), prior_samples, ordinal_logistic_regression,
                                              data_X, labels, data_X.shape[1])

    """Posterior sampling for (X)"""

    posterior_sample_X = sample_posterior(data_X)

    """Dictionairy with marginals with prior sampling"""

    MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
    print(MB_Scores)
    print(MB_Do)
    if MB_Do == '(X,Z2)':
        correct_MB = correct_MB + 1


    """Experimental Data"""

    exp_data, exp_labels = Generate_Experimental_Data(Ne)

    exp_data_X_Z1 = exp_data[:, [0, 1]]
    exp_data_X_Z2 = exp_data[:, [0, 2]]
    exp_data_X = exp_data[:, [0]]

    """For (X,Z1,Z2)"""

    # prior_samples = sample_priors(exp_data)
    #
    # print('Log_density for (X,Z1,Z2) with prior sampling from exp data:', log_predictive_density(random.PRNGKey(2),
    #       prior_samples, ordinal_logistic_regression, exp_data, exp_labels, exp_data.shape[1]))
    #
    # print('Log_density for (X,Z1,Z2) with posterior sampling from obs data:', log_predictive_density(random.PRNGKey(2),
    #       posterior_sample_X_Z1_Z2, ordinal_logistic_regression, exp_data, exp_labels, exp_data.shape[1]))


    """For (X,Z2)"""

    prior_samples = sample_priors(exp_data_X_Z2)

    marginal_prior_X_Z2 = log_predictive_density(random.PRNGKey(2), prior_samples, ordinal_logistic_regression,
                                                 exp_data_X_Z2, exp_labels, exp_data_X_Z2.shape[1])
    print('Log_density for (X,Z2) with prior sampling from exp data:', marginal_prior_X_Z2)

    # marginal_posterior_X_Z2 = log_predictive_density(random.PRNGKey(2), posterior_sample_X_Z2, ordinal_logistic_regression,
    #                                                  exp_data_X_Z2, exp_labels, exp_data_X_Z2.shape[1])
    marginal_posterior_X_Z2 = log_likelihood(ordinal_logistic_regression, posterior_sample_X_Z2, exp_data_X_Z2,
                                             exp_labels, exp_data_X_Z2.shape[1])
    # print('Log_density for (X,Z2) with posterior sampling from obs data:', sum(sum(marginal_posterior_X_Z2['obs']))/1000)
    """Des giati etsi stis simiwseis tetradio"""
    marginal_posterior_X_Z2 = jnp.sum(logsumexp(marginal_posterior_X_Z2['obs'], 0) - jnp.log(1000))
    print('Log_density for (X,Z2) with posterior sampling from obs data:', marginal_posterior_X_Z2)


    """For (X)"""

    prior_samples = sample_priors(exp_data_X)

    marginal_prior_X = log_predictive_density(random.PRNGKey(2), prior_samples, ordinal_logistic_regression,
                                              exp_data_X, exp_labels, exp_data_X.shape[1])
    print('Log_density for (X) with prior sampling from exp data:', marginal_prior_X)

    marginal_posterior_X = log_likelihood(ordinal_logistic_regression, posterior_sample_X, exp_data_X,
                                          exp_labels, exp_data_X.shape[1])
    # print('Log_density for (X) with posterior sampling from obs data:', sum(sum(marginal_posterior_X['obs']))/1000)
    marginal_posterior_X = jnp.sum(logsumexp(marginal_posterior_X['obs'], 0) - jnp.log(1000))
    print('Log_density for (X) with posterior sampling from obs data:', marginal_posterior_X)

    if marginal_posterior_X_Z2 > marginal_prior_X_Z2 and marginal_prior_X > marginal_posterior_X:
        correct_IMB = correct_IMB + 1

print(correct_MB)
print(correct_IMB)
