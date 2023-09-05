"""Swsth kai epiveveomeni ilopoihsh oti o By_hand upologismos tou Log_Likelihood vgazei to idio apotelesma me thn
etoimh synartisi. Prosoxi edw einai h swsth ilopoihsh tou log_likelihood"""
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


assert numpyro.__version__.startswith("0.11.0")
az.style.use("arviz-darkgrid")

def Generate_Observational_Data(sample_size):

    alpha = [-4, 4]
    beta = [0.9, 0.5, 1.4, 1.8]  # [Z1, Z2, X, Z2]
    # beta = [1.3, 1.25, 1.4, 1.15]

    e = dist.Normal(0, 1).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(sample_size,))
    Z1 = dist.Normal(0, 15).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(sample_size,))
    Z2 = dist.Normal(0, 10).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(sample_size,))

    μ_true = beta[0] * Z1 + beta[1] * Z2
    p_true = expit(μ_true)
    X = dist.Bernoulli(p_true).sample(random.PRNGKey(numpy.random.randint(100)))

    logit_0 = alpha[0] + (beta[2] * X + beta[3]*Z2) + e
    logit_1 = alpha[1] + (beta[2] * X + beta[3]*Z2) + e
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = np.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(1,))[0]
    data = pd.DataFrame({"Y": Y, 'X': X, "Z1": Z1, "Z2": Z2})

    return data, Y, X, Z1, Z2


def Regression_cases(Y, X, Z1, Z2, reg_variables):

    vars_dict = {'Y': Y, 'X': X, 'Z1': Z1, 'Z2': Z2}
    beta = {}
    for i in range(len(reg_variables)):
        beta['beta_{}'.format(reg_variables[i])] = (sample('beta_{}'.format(reg_variables[i]), Normal(0, 1)))

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal().expand([N_classes - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )
    prediction = 0
    for i in range(len(reg_variables)):
        prediction = prediction + beta['beta_{}'.format(reg_variables[i])] * vars_dict['{}'.format(reg_variables[i])]

    numpyro.sample(
        "Y",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),
        obs=Y
    )

def var_combinations(data):
    sample_list = data.columns.values[1:]
    list_comb = []
    for l in range(data.shape[1]-1):
        list_combinations = list(combinations(sample_list, data.shape[1]-l-1))
        for x in list_combinations:
            if x[0] == 'X':
                list_comb.append(x)

    return list_comb


correct_MB = 0
sample_size = 50
traces = []
for p in range(1):
    fb_trace = 0
    num_warmup, num_samples = 100, 100

    data, Y, X, Z1, Z2 = Generate_Observational_Data(sample_size)

    """Take all possible combinations for regression """
    list_comb = var_combinations(data)

    """Regression for (X,Z1,Z2), (X,Z1), (X,Z2), {X}"""
    MB_Scores = {}

    for i in range(len(list_comb)):
        reg_variables = list_comb[i]

        N_coef, N_classes = len(reg_variables), 3

        mcmc = MCMC(NUTS(model=Regression_cases), num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(random.PRNGKey(93), Y, X, Z1, Z2, reg_variables)
        # mcmc.print_summary()
        trace = mcmc.get_samples()
        traces.append(trace)

        Log_likelhood_dict = log_likelihood(Regression_cases, trace, Y, X, Z1, Z2, reg_variables)
        # print('Likelihood gia to b1', sum(Log_likelhood_dict['Y'][0]))
        Log_likelhood = sum(sum(Log_likelhood_dict['Y']))
        print(Log_likelhood)

        Marginal_Likelihood = Log_likelhood
        MB_Scores[reg_variables] = Marginal_Likelihood

    MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
    print(MB_Do)
    print(MB_Scores)
    if MB_Do == (('X', 'Z2')):
        correct_MB = correct_MB + 1
print(correct_MB / 1)

"""By Hand tropos"""
#Gia (X,Z1,Z2)
P = 0
trace = traces[0]
for dok in range(len(trace['beta_X'])):

    a_0 = trace['cutpoints'][:, 0][dok]
    a_1 = trace['cutpoints'][:, 1][dok]

    logit_0 = a_0 - (trace['beta_X'][dok] * X + trace['beta_Z1'][dok] * Z1 + trace['beta_Z2'][dok] * Z2)
    logit_1 = a_1 - (trace['beta_X'][dok] * X + trace['beta_Z1'][dok] * Z1 + trace['beta_Z2'][dok] * Z2)
    prob_0 = expit(logit_0)
    prob_1 = expit(logit_1) - prob_0
    prob_2 = 1 - expit(logit_1)
# Calculate Likelihood

    for i in range(len(prob_0)):
        if Y[i] == 0:
            P = P + math.log(prob_0[i])
        elif Y[i] == 1:
            P = P + math.log(prob_1[i])
        else:
            P = P + math.log(prob_2[i])
print(P)

#Gia (X,Z1)
P = 0
trace = traces[1]
#gia kathe bi apo to trace ypologise to log_likelihood gia ola ta data kai prosethese ola ta log_likelihood
# ayta gia to marginal
for dok in range(len(trace['beta_X'])):

    a_0 = trace['cutpoints'][:, 0][dok]
    a_1 = trace['cutpoints'][:, 1][dok]

    logit_0 = a_0 - (trace['beta_X'][dok] * X + trace['beta_Z1'][dok] * Z1)
    logit_1 = a_1 - (trace['beta_X'][dok] * X + trace['beta_Z1'][dok] * Z1)

    prob_0 = expit(logit_0)
    prob_1 = expit(logit_1) - prob_0
    prob_2 = 1 - expit(logit_1)
# Calculate Likelihood

    for i in range(len(prob_0)):
        if Y[i] == 0:
            P = P + math.log(prob_0[i])
        elif Y[i] == 1:
            P = P + math.log(prob_1[i])
        else:
            P = P + math.log(prob_2[i])
print(P)
#Gia (X,Z1)
P = 0
trace = traces[2]
#gia kathe bi apo to trace ypologise to log_likelihood gia ola ta data kai prosethese ola ta log_likelihood
# ayta gia to marginal
for dok in range(len(trace['beta_X'])):

    a_0 = trace['cutpoints'][:, 0][dok]
    a_1 = trace['cutpoints'][:, 1][dok]

    logit_0 = a_0 - (trace['beta_X'][dok] * X + trace['beta_Z2'][dok] * Z2)
    logit_1 = a_1 - (trace['beta_X'][dok] * X + trace['beta_Z2'][dok] * Z2)

    prob_0 = expit(logit_0)
    prob_1 = expit(logit_1) - prob_0
    prob_2 = 1 - expit(logit_1)
# Calculate Likelihood

    for i in range(len(prob_0)):
        if Y[i] == 0:
            P = P + math.log(prob_0[i])
        elif Y[i] == 1:
            P = P + math.log(prob_1[i])
        else:
            P = P + math.log(prob_2[i])
print(P)

#Gia {X}
P = 0
trace = traces[3]
#gia kathe bi apo to trace ypologise to log_likelihood gia ola ta data kai prosethese ola ta log_likelihood
# ayta gia to marginal
for dok in range(len(trace['beta_X'])):

    a_0 = trace['cutpoints'][:, 0][dok]
    a_1 = trace['cutpoints'][:, 1][dok]

    logit_0 = a_0 - (trace['beta_X'][dok] * X)
    logit_1 = a_1 - (trace['beta_X'][dok] * X)

    prob_0 = expit(logit_0)
    prob_1 = expit(logit_1) - prob_0
    prob_2 = 1 - expit(logit_1)
# Calculate Likelihood

    for i in range(len(prob_0)):
        if Y[i] == 0:
            P = P + math.log(prob_0[i])
        elif Y[i] == 1:
            P = P + math.log(prob_1[i])
        else:
            P = P + math.log(prob_2[i])
print(P)

"""Telos By hand tropou"""
