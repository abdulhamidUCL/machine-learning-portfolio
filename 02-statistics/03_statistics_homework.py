import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# 1 Binomial E[x] and Var[x]
n = 20
p = 0.3
dist = stats.binom(n,p)

print(f'E[x] = {dist.mean()}, Var[x] = {dist.var()}, Sigma = {dist.std()}')

x = np.arange(1, 15)

# mu +- sigmani korsat deyilgan
mu = dist.mean()
sigma = dist.std()

plt.bar(x, dist.pmf(x), color='black')
plt.axvline(x=dist.mean(), color='red', linestyle='dashed')
plt.axvline(x=mu - sigma, color='green', linestyle='dashed')
plt.axvline(x=mu + sigma, color='yellow', linestyle='dashed')
plt.title('Binomial Pmf grafigi')
plt.show()


# 2 ikta normal taqqoslash

Savolga_javob = 'agar expected value bir hil lekin sigma har hil bosa variance ozgaradi'

# N (100, 10^2)  U N (100, 25^2)
mu1 = 100
sigma1 = 10
dist1 = stats.norm(mu1, sigma1)

mu2 = 100
sigma2 = 25
dist2 = stats.norm(mu2, sigma2)


x = np.linspace(50, 150, 300)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

axes[0].plot(x, dist1.pdf(x))
axes[0].set_title("N(100, 10²)")

axes[1].plot(x, dist2.pdf(x))
axes[1].set_title("N(100, 25²)") # bunisi wider boladi chunki dispersiyasi katta

plt.tight_layout()
plt.show()



# 3 Var(aX + b) = a^2 * Var(X) isbotlash

# X ~ Binomial(10, 0.4)
X = np.random.binomial(10, 0.4, 100000)

# Y = 3X + 7
Y = 3 * X + 7

# variance of x
var_X = np.var(X)

# 1: srazu Var(Y)
var_Y_1 = np.var(Y)

# 2: formula blan Var(aX + b) = a² Var(X)
var_Y_2 = 3**2 * var_X

print("Var(X):", var_X)
print("Var(Y) 1 usul:", var_Y_1)
print("Var(Y) 2 usul:", var_Y_2)