'''
Gibbs Sampling
--------------

Gibbs Sampling is an MCMC (Markov Chain Monte Carlo) method to
sample from a probability distribution that would ordinarily be difficult to
sample from. Bayesian posteriors contain integrals (i.e., denominator
of Bayes Rule where marginalization over nuisance variables must take place)
that can be intractable or cumbersome to compute. MCMC is an approach to sample from that pdf.

Take the pdf of interest and break it up into conditional pdfs wrt to each
variable. The algorithm is recursive and sequential, that is, a few iterations
of Gibbs sampling is

y0 = rand
xi ~ p(x | y=y0)
yi ~ p(y | x=xi)
.
.
.
xi+1 ~ p(x| y=yi-1)
yi+1 ~ p(y| xi+1)


keep looping recursively

'''
import numpy as np

'''
Take the distribution from Casella and George (1992)

p(x,y) = (n!/((n-x)!x!))* (y^(x+a-1))*(1-y)^(n-x+b-1)

Break this distribution in 2 parts:

p(x|y) ~ Binomial Distribution
p(y|x) ~ Beta Distribution

'''

# Define constants
n = 10
alpha = 1
beta = 2
y0 = 1/2
numIterations = 10000
burnIn = 200

def gibbs(y,num):
    x = np.random.binomial(n, y)
    y = np.random.beta(x+alpha, n-x+beta)
    return x, y

y=y0
x_gibbs=[]
y_gibbs=[]

# Burn in-samples first
for j in range(burnIn):
    x,y=gibbs(y, burnIn)

# Start gibbs sampling
for k in range(numIterations):
    x,y=gibbs(y,burnIn)
    x_gibbs.append(x)
    y_gibbs.append(y)

# Print the mean of the distribution
print(np.mean(x_gibbs))
print(np.mean(y_gibbs))

