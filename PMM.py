from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import poisson

def generate_data(n,lambadas,cs):
    """

    :param n: number of samples to be generated
    :param lambadas: list of int>0 sized K s.t (list of lambda parameter for Poisson distribution)
    :param cs:list of int sized K s.t sum(ci for ci in cs) = 1
    :param K: PMM K parameter
    :return:
    generate data x1,...,xn of size n using parameters lambada,cs s.t
    1.sum(ci for ci in cs)=1
    2.forall lambda:lambadas - lambada >0
    3.zi ~ P where P(zi=z)=cs[z]
    4.xi ~ Pois(lambdas[zi])
    """
    xs = np.zeros(n)
    K = len(lambadas)
    for i in range(n):
        zi = np.random.choice(np.arange(K),p=cs)
        xs[i] = np.random.poisson(lambadas[zi])
    return xs


def update_term_em(j,xi,lambadas_t,cs):
    K=len(lambadas_t)
    a = cs[j]*poisson.pmf(xi,lambadas_t[j])
    b = sum([cs[k]*poisson.pmf(xi,lambadas_t[k]) for k in range(K)])
    return a/b

def em(xs,lambadas,cs,T):
    lambadas_t_plus_one = np.copy(lambadas)
    lambadas_t = None
    n = len(xs)
    for t in range(T):
        lambadas_t = np.copy(lambadas_t_plus_one)
        lambadas_t_plus_one = np.zeros(lambadas_t.shape[0])
        for j in range(len(lambadas_t)):
            temp = np.zeros(n)
            for i in range(n):
                temp[i] =update_term_em(j,xs[i],lambadas_t,cs)
            a = np.sum(np.multiply(xs,temp))
            b = np.sum(temp)
            lambadas_t_plus_one[j] = a/b
    return lambadas_t


#Question a
n=1000
lambdas = np.array([5,10,11])
cs = np.array([0.4,0.4,0.2])
xs = generate_data(n,lambdas,cs)

#Question b
"""T=1000
lambadas_hat = np.array([1,2,3])
cs_hat = np.array([1/3,1/3,1/3])
lambdas_em = em(xs,lambadas_hat,cs_hat,T)"""

#Question c
ts = [5,50,100,500,1000]
bins = np.linspace(0, 30, 30)
lambadas_hat = np.array([1, 2, 3])
cs_hat = np.array([1 / 3, 1 / 3, 1 / 3])
for t in ts:
    lambdas_em = em(xs, lambadas_hat, cs_hat, t)
    xs_estimate = generate_data(n,lambdas_em,cs_hat)
    plt.hist(xs,bins,alpha=0.5,label="estimated",density=True)
    plt.hist(xs_estimate, bins, alpha=0.5,label="true",density=True)
    plt.legend(loc='best')
    plt.title("t = {}".format(t))
    plt.ylabel("density")
    plt.xlabel("x")
    plt.show()