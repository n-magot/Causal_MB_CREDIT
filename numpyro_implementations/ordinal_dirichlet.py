import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpy
import arviz as az
import matplotlib.pyplot as plt
import operator
from jax.scipy.special import expit, logit
import time
from jax import numpy as np, random
import numpyro
from numpyro import sample, handlers
from numpyro.distributions import (
    Dirichlet,
    TransformedDistribution,
    transforms,
)
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import TransformReparam

numpyro.enable_x64()

st = time.time()

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

def Generate_Observational_Data(sample_size):

    #beta_X > Z1*X +Z2*X
    beta_X = jnp.array([1.3, 2.25])
    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([1.4, 1.15])

    Z1 = 0 + 15 * random.normal(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
    data_Z1_Z2 = jnp.concatenate((Z1, Z2), axis=1)

    logits_X = jnp.sum(beta_X * data_Z1_Z2, axis=-1)
    X = dist.Bernoulli(logits=logits_X).sample(random.PRNGKey(numpy.random.randint(100)))
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

    Z1 = 0 + 15 * random.normal(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))

    X = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
    data_X_Z2 = jnp.concatenate((X, Z2), axis=1)
    logit_0 = -4 + jnp.sum(beta_Y * data_X_Z2, axis=-1)
    logit_1 = 4 + jnp.sum(beta_Y * data_X_Z2, axis=-1)
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)  # probability of class 1 or 0
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = jnp.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(1,))[0]

    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((X, Z1, Z2), axis=1)

    return data, labels

# Define the logistic regression model
def logistic_regression_model(data, labels):
    D = data.shape[1]
    N_classes = 3
    concentration = np.ones((N_classes,)) * 10.0
    anchor_point = 0.0

    coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(D), 10 * jnp.ones(D)))

    with handlers.reparam(config={"cutpoints": TransformReparam()}):
        cutpoints = sample(
            "cutpoints",
            TransformedDistribution(
                Dirichlet(concentration),
                transforms.SimplexToOrderedTransform(anchor_point),
            ),
        )
    logits = jnp.sum(coefs * data, axis=-1)

    return numpyro.sample('obs', dist.OrderedLogistic(predictor=logits, cutpoints=cutpoints), obs=labels)


# Define a function to compute the log likelihood
def log_likelihood_calculation(cutpoints, coefs, data, obs):

    logits = jnp.dot(data, coefs)
    log_likelihood = dist.OrderedLogistic(predictor=logits, cutpoints=cutpoints).log_prob(obs)
    return log_likelihood.sum()

# Number of posterior samples to generate
def sample_prior(data, observed_data):

    prior_samples = {}
    D = data.shape[1]
    N_classes = 3
    concentration = np.ones((N_classes,)) * 10.0
    anchor_point = 0.0

    coefs = dist.Normal(jnp.zeros(D), 10*jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))

    cutpoints = (TransformedDistribution(Dirichlet(concentration), transforms.SimplexToOrderedTransform(anchor_point),)
                 .sample(random.PRNGKey(0), (num_samples,)))

    prior_samples['coefs'] = coefs
    prior_samples['cutpoints'] = cutpoints

    return prior_samples

# Perform MCMC with NUTS to sample from the posterior
def sample_posterior(data, observed_data):
    kernel = NUTS(logistic_regression_model, init_strategy=numpyro.infer.init_to_sample)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(jax.random.PRNGKey(42), data, observed_data)

    # Get the posterior samples
    posterior_samples = mcmc.get_samples()
    data_plot = az.from_numpyro(mcmc)
    az.plot_trace(data_plot, compact=True, figsize=(15, 25))
    plt.show()
    return posterior_samples

# Calculate log likelihood for each posterior sample
def calculate_log_marginal(num_samples, samples, data, observed_data):
    log_likelihoods = jnp.zeros(num_samples)

    for i in range(num_samples):
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_calculation(samples["cutpoints"][i],
                                                                               samples["coefs"][i], data, observed_data))

    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(num_samples)
    return log_marginal_likelihood

correct_IMB = 0
correct_MB = 0
for i in range(1):

    data, observed_data = Generate_Observational_Data(1000)
    num_samples = 2000
    MB_Scores = {}

    prior_samples = sample_prior(data, observed_data)
    marginal = calculate_log_marginal(num_samples, prior_samples, data, observed_data)
    MB_Scores['(X,Z1,Z2)'] = marginal

    data_X_Z1 = data[:, [0, 1]]
    data_X_Z2 = data[:, [0, 2]]
    data_X = data[:, [0]]

    prior_samples = sample_prior(data_X_Z1, observed_data)
    marginal = calculate_log_marginal(num_samples, prior_samples, data_X_Z1, observed_data)
    MB_Scores['(X,Z1)'] = marginal

    prior_samples = sample_prior(data_X_Z2, observed_data)
    marginal = calculate_log_marginal(num_samples, prior_samples, data_X_Z2, observed_data)
    MB_Scores['(X,Z2)'] = marginal

    prior_samples = sample_prior(data_X, observed_data)
    marginal = calculate_log_marginal(num_samples, prior_samples, data_X, observed_data)
    MB_Scores['(X)'] = marginal

    MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
    print(MB_Scores)
    if MB_Do == '(X,Z2)':
        correct_MB = correct_MB + 1

    exp_data, exp_observed_data = Generate_Experimental_Data(100)

    exp_data_X_Z1 = exp_data[:, [0, 1]]
    exp_data_X_Z2 = exp_data[:, [0, 2]]
    exp_data_X = exp_data[:, [0]]

    """For (X,Z2)"""
    prior_samples = sample_prior(exp_data_X_Z2, exp_observed_data)
    marginal_prior_X_Z2 = calculate_log_marginal(num_samples, prior_samples, exp_data_X_Z2, exp_observed_data)
    print('Marginal (X,Z2) from experimental sampling:', marginal_prior_X_Z2)

    #posterior sampling
    posterior_samples = sample_posterior(data_X_Z2, observed_data)
    marginal_posterior_X_Z2 = calculate_log_marginal(num_samples, posterior_samples, exp_data_X_Z2, exp_observed_data)
    print('Marginal (X,Z2) from observational sampling:', marginal_posterior_X_Z2)

    """For (X,)"""
    prior_samples = sample_prior(exp_data_X, exp_observed_data)
    marginal_prior_X = calculate_log_marginal(num_samples, prior_samples, exp_data_X, exp_observed_data)
    print('Marginal (X,) from experimental sampling:', marginal_prior_X)

    #posterior sampling
    posterior_samples = sample_posterior(data_X, observed_data)
    marginal_posterior_X = calculate_log_marginal(num_samples, posterior_samples, exp_data_X, exp_observed_data)
    print('Marginal (X,) from observational sampling:', marginal_posterior_X)

    if marginal_posterior_X_Z2 > marginal_prior_X_Z2 and marginal_prior_X > marginal_posterior_X:
        correct_IMB = correct_IMB + 1


print(correct_MB)
print(correct_IMB)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
