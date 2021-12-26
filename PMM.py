from typing import List

import numpy as np

def generate_data(n:int,lambadas:List[int],cs:List[int],K)->np.ndarray:
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
    for i in range(n):
        zi = np.random.choice(np.arange(K),p=cs)
        xs[i] = np.random.poisson(lambadas[zi])
    return xs


#Question a
n=1000
lambdas = [5,10,11]
cs = [0.4,0.4,0.2]
K=3
xs = generate_data(n,lambdas,cs,K)