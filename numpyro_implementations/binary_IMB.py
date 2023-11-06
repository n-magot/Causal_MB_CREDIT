import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive, log_likelihood
from jax.scipy.special import logsumexp
from jax import random, vmap
import numpy
from itertools import combinations
import operator
import arviz as az
import corner


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

numpyro.set_host_device_count(2)
def Generate_Observational_Data(sample_size):

    #beta_X > Z1*X +Z2*X
    beta_X = jnp.array([1.3, 1.25])
    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([1.4, 1.15])

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

def Generate_Experimental_Data(sample_size):

    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([1.4, 1.15])

    Z1 = 0 + 15 * random.normal(random.PRNGKey((numpy.random.randint(1000))), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey((numpy.random.randint(1000))), (sample_size, 1))

    X = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
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
    alpha = numpyro.sample("alpha", dist.Normal(0, 100))
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(D), 100 * jnp.ones(D)))
    logits = alpha + jnp.dot(data, beta)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)


# Define a function to compute the log likelihood
def log_likelihood_calculation(alpha, beta, data, obs):
    logits = alpha + jnp.dot(data, beta)
    log_likelihood = dist.Bernoulli(logits=logits).log_prob(obs)
    return log_likelihood.sum()

# Number of posterior samples to generate
def sample_prior(data, observed_data):

    # prior_predictive = Predictive(logistic_regression_model, num_samples=num_samples)
    # prior_samples = prior_predictive(rng_key_, data, observed_data)
    prior_samples = {}
    D = data.shape[1]

    prior_samples1 = dist.Normal(jnp.zeros(D), 100*jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))
    prior_samples2 = dist.Normal(jnp.zeros(1), 100*jnp.ones(1)).sample(random.PRNGKey(0), (num_samples,))

    prior_samples["beta"] = prior_samples1
    prior_samples["alpha"] = prior_samples2

    pdf_priors = (((dist.Normal(jnp.zeros(D), 100*jnp.ones(D)).log_prob(prior_samples1)).sum() +
                  dist.Normal(jnp.zeros(1), 100*jnp.ones(1)).log_prob(prior_samples2).sum()))/num_samples
    return prior_samples, pdf_priors

# Perform MCMC with NUTS to sample from the posterior
def sample_posterior(data, observed_data):

    D = data.shape[1]

    kernel = NUTS(logistic_regression_model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(jax.random.PRNGKey(1244), data, observed_data)

    # inf_data = az.from_numpyro(mcmc)
    # az.summary(inf_data)
    # corner.corner(inf_data, var_names=["alpha", "beta"])
    # plt.show()

    # Get the posterior samples
    posterior_samples = mcmc.get_samples()

    data_plot = az.from_numpyro(mcmc)
    az.plot_trace(data_plot, compact=True, figsize=(15, 25))
    plt.show()

    # pdf_priors = (((dist.Normal(jnp.zeros(D), 10 * jnp.ones(D)).log_prob(prior_samples1)).sum() +
    #                dist.Normal(jnp.zeros(1), 10 * jnp.ones(1)).log_prob(prior_samples2).sum())) / num_samples

    # print(posterior_samples['alpha'])
    fb_trace = 0
    for i in range(D):
        fb_trace = (fb_trace + dist.Normal(numpy.mean(posterior_samples['beta'][:, i]),
                    numpy.std(posterior_samples['beta'][:, i])).
                    log_prob(posterior_samples['beta'][:, i]).sum())


    fa_trace = (dist.Normal(numpy.mean(posterior_samples['alpha']), numpy.std(posterior_samples['alpha']))
                .log_prob(posterior_samples['alpha']).sum())

    pdf_posterior = (fa_trace + fb_trace)/num_samples

    return posterior_samples, pdf_posterior

# Calculate log likelihood for each posterior sample
def calculate_log_marginal(num_samples, samples, data, observed_data):
    log_likelihoods = jnp.zeros(num_samples)

    for i in range(num_samples):
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_calculation(samples["alpha"][i], samples["beta"][i],
                                                                               data, observed_data))

    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods)
    return log_marginal_likelihood

for i in range(1):

    data, observed_data = Generate_Observational_Data(1000)
    num_samples = 1000
    MB_Scores = {}

    prior_samples, pdf_priors = sample_prior(data, observed_data)
    marginal = calculate_log_marginal(num_samples, prior_samples, data, observed_data)
    MB_Scores['(X,Z1,Z2)'] = marginal + pdf_priors

    data_X_Z1 = data[:, [0, 1]]
    data_X_Z2 = data[:, [0, 2]]
    data_X = data[:, [0]]

    prior_samples, pdf_priors = sample_prior(data_X_Z1, observed_data)
    marginal = calculate_log_marginal(num_samples, prior_samples, data_X_Z1, observed_data)
    MB_Scores['(X,Z1)'] = marginal + pdf_priors

    prior_samples, pdf_priors = sample_prior(data_X_Z2, observed_data)
    marginal = calculate_log_marginal(num_samples, prior_samples, data_X_Z2, observed_data)
    MB_Scores['(X,Z2)'] = marginal + pdf_priors

    prior_samples, pdf_priors = sample_prior(data_X, observed_data)
    marginal = calculate_log_marginal(num_samples, prior_samples, data_X, observed_data)
    MB_Scores['(X)'] = marginal + pdf_priors

    MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
    print(MB_Scores)
    print(MB_Do)

    exp_data, exp_observed_data = Generate_Experimental_Data(300)

    exp_data_X_Z1 = exp_data[:, [0, 1]]
    exp_data_X_Z2 = exp_data[:, [0, 2]]
    exp_data_X = exp_data[:, [0]]

    """For (X,Z2)"""
    prior_samples, pdf_priors = sample_prior(exp_data_X_Z2, exp_observed_data)
    print(pdf_priors)
    marginal = calculate_log_marginal(num_samples, prior_samples, exp_data_X_Z2, exp_observed_data)
    print('Marginal (X,Z2) from experimental sampling:', marginal + pdf_priors)
    #posterior sampling
    posterior_samples, pdf_posterior = sample_posterior(data_X_Z2, observed_data)
    print(pdf_posterior)
    marginal = calculate_log_marginal(num_samples, posterior_samples, exp_data_X_Z2, exp_observed_data)
    print('Marginal (X,Z2) from observational sampling:', marginal + pdf_posterior)

    # marginal_posterior = log_likelihood(logistic_regression_model, posterior_samples, exp_data_X_Z2, exp_observed_data)
    #
    # print('X,Z2', jnp.sum(logsumexp(marginal_posterior['obs'], 0) - jnp.log(num_samples)))

    """For (X,)"""
    prior_samples, pdf_priors = sample_prior(exp_data_X, exp_observed_data)
    print(pdf_priors)
    marginal = calculate_log_marginal(num_samples, prior_samples, exp_data_X, exp_observed_data)
    print('Marginal (X,) from experimental sampling:', marginal + pdf_priors)
    #posterior sampling
    posterior_samples, pdf_posterior = sample_posterior(data_X, observed_data)
    print(pdf_posterior)
    marginal = calculate_log_marginal(num_samples, posterior_samples, exp_data_X, exp_observed_data)
    print('Marginal (X,) from observational sampling:', marginal + pdf_posterior)
