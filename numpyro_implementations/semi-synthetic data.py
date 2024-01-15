import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random, vmap
import numpy
from itertools import combinations
import operator
import pandas as pd
import arviz as az
import time


st = time.time()
def Generate_Experimental_Data(sample_size):

    #beta_Y > T*Y + A*Y + Z1*T
    beta_Y = jnp.array([1.4, 1.9, 1.7])
    e = 0 + 1 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    Z1 = 0 + 15 * random.normal(random.PRNGKey((numpy.random.randint(100))), (sample_size, 1))
    A = 0 + 10 * random.normal(random.PRNGKey((numpy.random.randint(100))), (sample_size, 1))

    T = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(100)), (sample_size, 1))
    data_T_A = jnp.concatenate((T, A, Z1), axis=1)
    logits_Y = jnp.sum(beta_Y * data_T_A + e, axis=-1)
    Y = dist.Bernoulli(logits=logits_Y).sample(random.PRNGKey((numpy.random.randint(100))))

    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((T, A, Z1), axis=1)

    return data, labels
Ne = 1000
data_exp, labels_exp = Generate_Experimental_Data(Ne)

"""standardizes data and scale them to have mean 0 and std 0.5:"""
# data_n_2 = 2 * (data - data.mean()) / (data.std())  # standardization


# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

def model(data, labels):

    D = data.shape[1]
    alpha = numpyro.sample("alpha", dist.Cauchy(0, 2.5))
    beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 10 * jnp.ones(D)))
    logits = alpha + jnp.dot(data, beta)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

# Run NUTS.
kernel = NUTS(model)
num_samples = 1000
mcmc = MCMC(kernel, num_warmup=1000, num_chains=1, num_samples=num_samples)
mcmc.run(
    rng_key_, data_exp, labels_exp
)
mcmc.print_summary()

trace = mcmc.get_samples()
intercept = trace['alpha'].mean()
slope = np.array([trace['beta'][:, 0].mean(), trace['beta'][:, 1].mean(), trace['beta'][:, 2].mean()])

A = data_exp[:, 1]
Z1 = data_exp[:, 2]
data_numpy = np.array(data_exp)
print(data_numpy)
T_new = []
for i in range(len(data_numpy[:, 0])):
    if data_numpy[i, 1] < 3:
        T_new.append(0)
    else:
        T_new.append(1)

obs_data = np.stack((T_new, A, Z1), axis=1)
print(obs_data)
print("mean intercept", intercept)
print("mean slope", slope)
e = 0 + 1 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (Ne, 1))
logits = jnp.sum(slope * obs_data + e, axis=-1)
obs_labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey((numpy.random.randint(100))))

def binary_logistic_regression(data, labels):

    D = data.shape[1]
    alpha = numpyro.sample("alpha", dist.Cauchy(0, 2.5))
    beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 10 * jnp.ones(D)))
    logits = alpha + jnp.dot(data, beta)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)


# Define a function to compute the log likelihood
def log_likelihood_calculation(alpha, beta, data, obs):
    logits = alpha + jnp.dot(data, beta)
    log_likelihood = dist.Bernoulli(logits=logits).log_prob(obs)
    return log_likelihood.sum()

def sample_prior(data):

    prior_samples = {}
    D = data.shape[1]
    #paizei kai i paralagi me Cauchy(0, 2.5) gia tous coefs kai Cauchy(0, 10 ) gia to intercept
    prior_samples1 = dist.Cauchy(jnp.zeros(D), 2.5*jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))
    prior_samples2 = dist.Cauchy(jnp.zeros(1), 10*jnp.ones(1)).sample(random.PRNGKey(0), (num_samples,))

    prior_samples["beta"] = prior_samples1
    prior_samples["alpha"] = prior_samples2

    return prior_samples

# Perform MCMC with NUTS to sample from the posterior
def sample_posterior(data, observed_data):

    D = data.shape[1]

    kernel = NUTS(binary_logistic_regression)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(jax.random.PRNGKey(1244), data, observed_data)


    # Get the posterior samples
    posterior_samples = mcmc.get_samples()

    data_plot = az.from_numpyro(mcmc)
    az.plot_trace(data_plot, compact=True, figsize=(15, 25))

    return posterior_samples

# Calculate log likelihood for each posterior sample
def calculate_log_marginal(num_samples, samples, data, observed_data):
    log_likelihoods = jnp.zeros(num_samples)

    for i in range(num_samples):
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_calculation(samples["alpha"][i], samples["beta"][i],
                                                                               data, observed_data))

    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(num_samples)
    return log_marginal_likelihood

def var_combinations(data):
    #how many variables we have
    num_variables = data.shape[1]
    column_list = list(map(lambda var: var, range(0, num_variables)))

    df = pd.DataFrame(data, columns=column_list)
    sample_list = df.columns.values[0:]
    list_comb = []
    for l in range(df.shape[1]):
        list_combinations = list(combinations(sample_list, df.shape[1]-l))
        for x in list_combinations:
            if x[0] == 0:
                list_comb.append(x)
    return list_comb


for i in range(1):

    data, observed_data = obs_data, obs_labels
    num_samples = 1000
    MB_Scores = {}
    IMB_Scores_obs = {}
    IMB_Scores_exp = {}

    list_comb = var_combinations(data)
    print(list_comb)

    for comb in range(len(list_comb)):
        reg_variables = list_comb[comb]

        sub_data = data[:, reg_variables]

        prior_samples = sample_prior(sub_data)

        marginal = calculate_log_marginal(num_samples, prior_samples, sub_data, observed_data)

        MB_Scores[reg_variables] = marginal


    MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
    print(MB_Scores)
    print(MB_Do)

    sample_list = list(MB_Do)

    """Searching for subsets of MB"""
    subset_list = []
    for s in range(len(sample_list)):
        list_combinations = list(combinations(sample_list, len(sample_list) - s))
        for x in list_combinations:
            if x[0] == 0:
                subset_list.append(x)
    print('The subsets of MB are {}'.format(subset_list))

    exp_data, exp_observed_data = data_exp, labels_exp

    """For subsets of MB sample from experimental and observational data"""
    for j in range(len(subset_list)):
        reg_variables = subset_list[j]
        sub_data = data[:, reg_variables]
        exp_sub_data = exp_data[:, reg_variables]

        posterior_samples = sample_posterior(sub_data, observed_data)

        prior_samples = sample_prior(exp_sub_data)

        marginal = calculate_log_marginal(num_samples, prior_samples, exp_sub_data, exp_observed_data)
        print('Marginal {} from experimental sampling:'.format(reg_variables), marginal)
        IMB_Scores_exp[reg_variables] = marginal

        marginal = calculate_log_marginal(num_samples, posterior_samples, exp_sub_data, exp_observed_data)
        print('Marginal {} from observational sampling:'.format(reg_variables), marginal)

        IMB_Scores_obs[reg_variables] = marginal


print(IMB_Scores_exp)
print(IMB_Scores_obs)


# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
