import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive, log_likelihood
from jax.scipy.special import logsumexp
from jax import random, vmap
import numpy
import operator


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
def Generate_Observational_Data(sample_size):

    #beta_X > Z1*X +Z2*X
    beta_X = jnp.array([1.15, 1.2])
    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([1.4, 1.25])

    Z1 = 0 + 15 * random.normal(random.PRNGKey((numpy.random.randint(1000))), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey((numpy.random.randint(1000))), (sample_size, 1))
    data_Z1_Z2 = jnp.concatenate((Z1, Z2), axis=1)

    logits_X = jnp.sum(beta_X * data_Z1_Z2, axis=-1)
    X = dist.Bernoulli(logits=logits_X).sample(random.PRNGKey((numpy.random.randint(1000))))
    X = X.reshape(-1, 1)
    data_X_Z2 = jnp.concatenate((X, Z2), axis=1)
    logits_Y = jnp.sum(beta_Y * data_X_Z2, axis=-1)
    Y = dist.Bernoulli(logits=logits_Y).sample(random.PRNGKey((numpy.random.randint(1000))))

    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((X, Z1, Z2), axis=1)

    return data, labels

# Define the logistic regression model
def logistic_regression_model(data, labels):
    D = data.shape[1]
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(D), 10 * jnp.ones(D)))
    logits = jnp.dot(data, beta)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)


# Define a function to compute the log likelihood
def log_likelihood_calculation(beta, data, obs):
    logits = jnp.dot(data, beta)
    log_likelihood = dist.Bernoulli(logits=logits).log_prob(obs)
    return log_likelihood.sum()

# Number of posterior samples to generate
def sample_prior(data, observed_data):

    prior_predictive = Predictive(logistic_regression_model, num_samples=num_samples)
    prior_samples = prior_predictive(rng_key_, data, observed_data)
    return prior_samples

# Perform MCMC with NUTS to sample from the posterior
def sample_posterior(data, observed_data):
    kernel = NUTS(logistic_regression_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=num_samples, num_chains=1)
    mcmc.run(jax.random.PRNGKey(0), data, observed_data)

    # Get the posterior samples
    posterior_samples = mcmc.get_samples()
    return posterior_samples

# Calculate log likelihood for each posterior sample
def calculate_log_marginal(num_samples, samples, data, observed_data):
    log_likelihoods = jnp.zeros(num_samples)

    for i in range(num_samples):
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_calculation(samples["beta"][i], data, observed_data))

    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(num_samples)
    return log_marginal_likelihood


data, observed_data = Generate_Observational_Data(100)
num_samples = 1000
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
print(MB_Do)
