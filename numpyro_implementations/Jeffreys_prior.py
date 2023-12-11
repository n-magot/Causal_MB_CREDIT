import matplotlib.pyplot as plt
import numpy as np
def generate_correlation_matrix(p, param):
    res = np.zeros(shape=(p, p))
    for s in range(p):
        for t in range(0, s+1):
            corr = param
            if s == t:
                corr = 1.0
            res[s, t] = corr
            res[t, s] = corr
    return res

def generate_design_matrix(n, K):
    mean = np.zeros(K.shape[0])
    return np.random.multivariate_normal(mean, K, size=n)

def generate_weights(p):
    return np.random.normal(size=p)

def generate_data_set(n, K):
    p = K.shape[0]
    X = generate_design_matrix(n, K)
    w = generate_weights(p)

    u = np.dot(X, w)

    p = 1 / (1 + np.exp(-u))

    y = []
    for i in range(n):
        y.append(np.random.binomial(1, p[i]))
    y = np.array(y)

    return X, y, w

np.random.seed(0)
n = 20
p = 1

K = generate_correlation_matrix(p, 0.5)
X, y, w_true = generate_data_set(n, K)

def compute_a_matrix(X, u):
    p_vector = 1 / (1 + np.exp(u))
    return np.diag(p_vector * (1 - p_vector))

def compute_fisher_information_matrix(X, u):
    A = compute_a_matrix(X, u)
    return np.dot(X.T, np.dot(A, X))

def compute_log_prior(X, u):
    FIM = compute_fisher_information_matrix(X, u)
    return 0.5 * np.linalg.slogdet(FIM)[1]

def evaluate_mesh(f, W1, W2):
    n, m = W1.shape
    res = np.zeros(W1.shape)
    for i in range(n):
        for j in range(m):
            res[i, j] = f(W1[i, j], W2[i, j])
    return res

def f_prior(w):
    u = np.dot(X, np.array([w]))
    return np.exp(compute_log_prior(X, u))

import scipy
from scipy.stats import cauchy, norm

Z, _ = scipy.integrate.quad(f_prior, -100, 100)
wx = np.arange(-20, 20, 0.1)
yx = [f_prior(w)/Z for w in wx]
plt.plot(wx, yx, label='jeffreys')

plt.plot(wx, cauchy.pdf(wx, 0, 2.5), label='cauchy')
plt.plot(wx, norm.pdf(wx, 0, 10), label='normal')

plt.xlabel('w')
plt.ylabel('Probability')
plt.title('Compare Jeffreys and Cauchy (0, 2.5) prior for logistic regression')
plt.legend()

plt.show()
