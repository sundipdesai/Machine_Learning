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

Rev 2:
 - Added burn in feature
 - Added plot functionality
 - Functionalized MH-Sampler

Notes:

TODO: Add lag parameters for next rev.

'''

import numpy as np
import matplotlib.pyplot as plt

# ----  Inputs ----- #

numSamples = 2000

# Standard deviation for Gaussian (proposed distribution)
sigma = 0.5

# --- End Inputs --- #

# Initialize parameters
x = 0
acceptSample = 0
burnIn = 200
xfinal = np.zeros(numSamples)

def targetDistribution(z):
    '''

    Example distribution taken from:
    https://health.adelaide.edu.au/psychology/ccs/docs/ccs-class/technote_metropolishastings.pdf

    :param z: x or xstar value
    :return: Probability

    '''
    return np.exp(-z ** 2) * (2 + np.sin(5 * z) + np.sin(2 * z))


# -------- Metropolis-Hastings Algorithm -------- #
def getMetropolisSample(x, sigma):

    # Default accept sample flag to false
    accept = False

    # Sample from normal distribution
    xstar = np.random.normal(x, sigma)

    # Compute the Acceptance probability
    pA = targetDistribution(xstar) / targetDistribution(x)

    qA = np.abs(np.random.normal(xstar, sigma)) / np.abs(np.random.normal(x, sigma))

    # Acceptance check
    if np.abs(np.random.uniform()) < np.min([1, pA * qA]):

        x = xstar

        accept = True  # Keep counts of accepted values

    return x, accept

# -------- Execution -------- #
# Let algorithm burn samples first then proceed with
# nominal sampling

# Burn In
for i in range(burnIn):
    x, accept = getMetropolisSample(x, sigma)

xfinal[0] = x

# MCMC Sample
for k in range(numSamples-1):
    xfinal[k+1], accept = getMetropolisSample(xfinal[k], sigma)
    if accept:
        acceptSample += 1

# -------- End Metropolis-Hastings Algorithm -------- #

print "Acceptance Rate: ", 100 * acceptSample / numSamples, "%"

# --- Plots --- #
iteration = np.arange(1, numSamples+1, 1)
plt.figure(1)
plt.subplot(211)
plt.hist(xfinal, 50)
plt.ylabel("Sample Frequency")
plt.xlabel("Bin")

plt.subplot(212)
plt.plot(xfinal, iteration)
plt.ylabel("Iteration")
plt.xlabel("Sample Value")
plt.show()