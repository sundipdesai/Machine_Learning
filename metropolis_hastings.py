'''

Metropolis Hastings Algorithm
-----------------------------

When direct sampling cannot be made from a probability
distribution (i.e., Bayesian posteriors), a Markov Chain Monte Carlo (MCMC)
method may be used.


Use a target distribution proportional to the original distribution that
cannot be sampled from. Over many iterations, the target distribution
should become a stationary distribution that should yield samples from
the original distribution.

For the MH algorithm, we used a proposed distribution of a Gaussian. That is, we sample
from a Gaussian distribution centered at x at every iteration.

The key elements that make this algorithm work are:

(1) Markov Chain statistical model, i.e., current statistics are based only on the previous state's
    statistics. This allows us, over many iterations, to reach a stationary distribution, that is,
    a distribution that resembles the original distribution

(2) Target distribution is proportional to the distribution that cannot be sampled from

(3) Reasonable proposed distribution to generate candidate samples. Usually go with a Gaussian
    that is centered on the current x value with some fixed standard deviation, sigma.


Notes:

Add burn in and lag parameters for next rev.
Functionalize algorithm
'''

import numpy as np

# ----  Inputs ----- #

numSamples = 1000

# Standard deviation for Gaussian (proposed distribution)
sigma = 0.5

# --- End Inputs --- #

# Initialize parameters
x = np.zeros(numSamples)

xstar = np.zeros(numSamples)

accept = 0

# Initial value of x
x[0] = -1

def targetDistribution(z):
    '''

    Example distribution taken from:
    https://health.adelaide.edu.au/psychology/ccs/docs/ccs-class/technote_metropolishastings.pdf

    :param z: x or xstar value
    :return: Probability

    '''
    return np.exp(-z ** 2) * (2 + np.sin(5 * z) + np.sin(2 * z))


# -------- Metropolis-Hastings Algorithm -------- #
for i in range(numSamples - 1):

    # Sample from normal distribution
    xstar[i] = np.random.normal(x[i], sigma)

    # Compute the Acceptance probability
    pA = targetDistribution(xstar[i]) / targetDistribution(x[i])

    qA = np.abs(np.random.normal(xstar[i], sigma)) / np.abs(np.random.normal(x[i], sigma))

    # Acceptance check
    if np.abs(np.random.uniform()) < np.min([1, pA * qA]):

        x[i + 1] = xstar[i]

        accept += 1  # Keep counts of accepted values

    else:  # Retain previous value for next iteration

        x[i + 1] = x[i]

# -------- End Metropolis-Hastings Algorithm -------- #

print("Acceptance Rate: ", 100 * accept / numSamples)

