from scipy.special import expit
import arviz as az
import jax
import jax.numpy as jnp
from jax import random, vmap
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import numpy as np, random
import numpyro
from numpyro import sample, handlers
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS, log_likelihood
from scipy.special import expit
import arviz as az
import numpyro.distributions as dist
import numpy as np
import numpyro
from jax import random
import math
import numpy
import pandas as pd
from scipy.stats import norm
import operator
import arviz as az
import jax
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro import handlers
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
apply_to_index = {'0': 0, '1': 1, '2': 2}

def Generate_Observational_Data(sample_size,rng):
    alpha = [-4, 4]
    beta = [0.9, 0.5, 1.4, 1.8]  # [Z1, Z2, X, Z2]
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

df = Generate_Observational_Data(1000,42)

def Regression_cases(X,Z1,Z2,n_apply_levels, Y=None):

    bX = numpyro.sample(
        "pared",
        dist.Normal(0, 100),
    )
    bZ1 = numpyro.sample(
        "public",
        dist.Normal(0, 100),
    )
    bZ2 = numpyro.sample(
        "gpa",
        dist.Normal(0, 100),
    )

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 100).expand([n_apply_levels - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )

    prediction = (X * bX + Z1 * bZ1 + Z2 * bZ2)

    logits = cutpoints - prediction[:, jnp.newaxis]
    cumulative_probs = jnp.pad(
        jax.scipy.special.expit(logits),
        pad_width=((0, 0), (1, 1)),
        constant_values=(0, 1),
    )
    probs = numpyro.deterministic("probs", jnp.diff(cumulative_probs))
    # print(probs)
    numpyro.sample(
        "Y",
        dist.Categorical(probs=probs),
        obs=Y,
    )


prior_pred = numpyro.infer.Predictive(Regression_cases, num_samples=100)
prior_predictions = prior_pred(
    jax.random.PRNGKey(93),
    X=np.array([0, 1]),
    Z1=np.array([3, 3]),
    Z2=np.array([1, 1]),
    n_apply_levels=3,
    Y=df["Y"].map(apply_to_index).to_numpy()[:2],
)

print(df['Y'].value_counts())
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 6))
ax = ax.flatten()
rows = {

    0: "X = 0, Z1 = 3, Z2 = 1",
    1: "X = 1, Z1 = 3, Z2 = 1"
}

for row, description in rows.items():
    prior_df = pd.DataFrame(
        prior_predictions["probs"][:, row, :], columns=apply_to_index
    )
    sns.kdeplot(data=prior_df, ax=ax[row], cut=0)
    ax[row].set_title(description)
for row, description in rows.items():
    prior_df = pd.DataFrame(
        prior_predictions["probs"][:, row, :], columns=apply_to_index
    )
    print(prior_df)
    sns.kdeplot(data=prior_df, ax=ax[row], cut=0)
    ax[row].set_title(description)
    plt.show()
