import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy import stats

# 1 Normal - Test natijalari

mu = 72
sigma = 12
#a
xa = stats.norm.cdf(84, mu ,sigma) - stats.norm.cdf(60, mu ,sigma)
print(xa)

#b
xb = 1-  stats.norm.cdf(90, mu ,sigma)
print(xb)

#c
samples = np.random.normal(mu, sigma, 1000) # teoriya ishlashni tekshirish uchun

x = np.linspace(20, 120, 300)
y = stats.norm.pdf(x, mu, sigma)
# for pdf chizigi uchun

plt.figure(figsize=(10,6))

plt.hist(samples, bins=30, density=True, alpha=0.5, label='Samples')
plt.plot(x, y, label='PDF')

plt.title('Histogram +  PDF')
plt.xlabel('Oquchilar bali')
plt.ylabel('Density')
plt.legend()
plt.show()


# 2 Binomial - Spam filtri
n = 100
p = 0.35
avg = 35

# aynan 30 ta spam ehtimoli
k = 30
pmf = stats.binom.pmf(k, n, p) # pmf == exactly one number k = 5 , [5]
print(f'aynan 30 ta spam ehtimoli -  {pmf}')



# 50 dan kop bolish ehtimoli
k = 50
cdf = 1 - stats.binom.cdf(k, n, p) # cdf <= till that number k = 5 , [1, 2, 3, 4, 5]
print(f'50 dan kop bolish ehtimoli - {cdf}')



# graph
k = np.arange(0, n + 1)
pmf = stats.binom.pmf(k, n, p)
cdf = stats.binom.cdf(k, n, p)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PMF
axes[0].bar(
    k,
    pmf,
    width=0.8,
    color='red',
    edgecolor='white'
)

axes[0].set_title('Binomial PMF')
axes[0].set_xlabel('Number of spam emails')
axes[0].set_ylabel('Probability')


# CDF
axes[1].bar(
    k,
    cdf,
    width=0.8,
    color='blue',
    edgecolor='white'
)

axes[1].set_title('Binomial CDF')
axes[1].set_xlabel('Number of spam emails')
axes[1].set_ylabel('Cumulative probability')

plt.tight_layout()
plt.show()


# 3 Poisson - Server so'rovlari

# aynan 10 bolish probability
lam = 8
k = 10

pmf = stats.poisson.pmf(k, lam)
print(f'aynana {k} ehtimoli -  {pmf}')

# 15 dan ko'p so'rov bo'lish ehtimoli
k = 15

cdf = 1 - stats.poisson.cdf(k, lam)
print(f'{k} dan koproq bolish ehtimoli -  {cdf}')

# A=3,8,15 solishtiruv grafigi

k = np.arange(0, 30)

pmf3 = stats.poisson.pmf(k, 3)
pmf8 = stats.poisson.pmf(k, 8)
pmf15 = stats.poisson.pmf(k, 15)

plt.figure(figsize=(10, 6))

plt.plot(k, pmf3, marker='o', label='λ = 3')
plt.plot(k, pmf8, marker='o', label='λ = 8')
plt.plot(k, pmf15, marker='o', label='λ = 15')

plt.title('Poisson taqsimoti: λ = 3, 8, 15')
plt.xlabel('Number of requests')
plt.ylabel('Probability')
plt.legend()

plt.show()

# 4 clean_laptopData — Price normalmi?

df = pd.read_csv('data/clean_laptopData.csv')

sns.histplot(
    df['Price'],
    bins=30,
    color = 'blue',
    fill = True,
)


# price juda katta loq1p ishlatish kere (Transformatsiya)
df['Price_log'] = np.log1p(df['Price'])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

sns.histplot(
    df['Price'],
    kde=True,
    color = 'blue',
    ax = axes[0]
)
axes[0].set_title('Before Log')

sns.histplot(
    df['Price_log'],
    kde=True,
    ax = axes[1],
    color= 'red',
)
axes[1].set_title('After Log')

plt.tight_layout()
plt.show()


# qq plot

stats.probplot(df["Price"], dist="norm", plot=plt)
plt.title("QQ Plot — Original Price")
plt.show()

stats.probplot(df["Price_log"], dist="norm", plot=plt)
plt.title("QQ Plot — log1p(Price)")
plt.show()