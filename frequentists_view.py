import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

import pandas as pd
data_diam = pd.read_csv('diamonds.csv')

from pandas.api.types import CategoricalDtype
cat_type = CategoricalDtype(categories=['Fair', 'Good', 'Ideal', 'Very Good', 'Premium'], ordered=True)
data_diam["cut"] = data_diam["cut"].astype(cat_type)

data_diam['volume'] = data_diam['x']*data_diam['y']*data_diam['z']
data_diam.drop(['x','y','z'],axis=1,inplace=True)

from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.special import expit
import arviz as az
import numpy
import numpyro.distributions as dist
from jax import numpy as np
import numpyro
from numpyro.distributions import (Normal)
from numpyro.infer import MCMC, NUTS, log_likelihood
from numpyro import sample
from itertools import combinations
from jax import random
import math
import operator
import pandas as pd
def Generate_Observational_Data(sample_size,rng):

    alpha = [-4, 4]
    beta = [0.4, 1.25, 1.4, 0.5]  # [Z1, Z2, X, Z2]
    # beta = [1.3, 1.25, 1.4, 1.15]

    e = dist.Normal(0, 1).sample(random.PRNGKey(numpy.random.randint(rng)), sample_shape=(sample_size,))
    Z1 = dist.Normal(0, 15).sample(random.PRNGKey(numpy.random.randint(rng)), sample_shape=(sample_size,))
    Z2 = dist.Normal(0, 10).sample(random.PRNGKey(numpy.random.randint(rng)), sample_shape=(sample_size,))

    μ_true = beta[0] * Z1 + beta[1] * Z2
    p_true = expit(μ_true)
    X = dist.Bernoulli(p_true).sample(random.PRNGKey(numpy.random.randint(rng)))

    logit_0 = alpha[0] + (beta[2] * X + beta[3]*Z2) + e
    logit_1 = alpha[1] + (beta[2] * X + beta[3]*Z2) + e
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = np.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(rng)), sample_shape=(1,))[0]
    data = pd.DataFrame({"Y": Y, 'X': X, "Z1": Z1, "Z2": Z2})

    return data

data = Generate_Observational_Data(1000, 42)
print(data)

mod_prob = OrderedModel(data['Y'],
                        data[['X', 'Z1', 'Z2']],
                        distr='logit')

res_log = mod_prob.fit(method='bfgs')
print(res_log.summary())


mod_prob = OrderedModel(data['Y'],
                        data[['X', 'Z1']],
                        distr='logit')

res_log = mod_prob.fit(method='bfgs')
print(res_log.summary())


mod_prob = OrderedModel(data['Y'],
                        data[['X', 'Z2']],
                        distr='logit')

res_log = mod_prob.fit(method='bfgs')
print(res_log.summary())


mod_prob = OrderedModel(data['Y'],
                        data[['X']],
                        distr='logit')

res_log = mod_prob.fit(method='bfgs')
print(res_log.summary())
