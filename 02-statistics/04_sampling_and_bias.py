import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# exercise 1

javob = 'masalan restoranga bir kunda kegan hammada sorovnoma olish ham qiyin, shunin uchun biz sample olamiz masalan soatiga 3 kishidan sorovnomaolinadi'

# exercise 2 μ, σ, x̄, s, N, n

μ = 'populationni ortachasi'
σ = 'populationni standard deviation'

x̄ = 'sample ni ortachasi'
s = 'sample ni standard deviation'

N = 'populationni hajmi, nechtaligi'
n = 'sampleniki'

# exercise 3

N = 3_000_000
n = 500

javob2 = 'populationni ortachasi va deviationi aniqmas'
javob3 = 'sampleni ortachasi va deviationi hisoblanadi'


# exercise 4

javob4 = 'baribur error boladida, errorni formulasida korsak boladi sigma / sqr(n), agar n qancha N yaqinlashursa oshanda error 0 yaqinlashadi'

# exercise 5

javobA = 'stratified'
javobB = 'systematic'
javobC = 'cluster'
javobD = 'simple random'

# exercise 6

population = np.random.randint(150, 200, 1000)

sample = np.random.choice(population, size=50, replace=False)

plt.hist(population, bins=50)
plt.show()

# exercise 7

df = pd.DataFrame({
    'kurs' : np.random.randint(1, 5, 200),
    'height' : np.random.randint(150, 200, 200),
})

sample7 = df.groupby('kurs').sample(n=10)

print(sample7)


# exercise 8

df = pd.DataFrame({
    'element' : range(1, 101),
    'cluster' : np.repeat(range(1, 21), 5),
})


clusters = np.random.choice(range(1, 21), 4, replace=False)

sample8 = []

for a in clusters:
    x = df[df['cluster'] == a]
    sample8.append(x)

sample8 = pd.concat(sample8)
print(sample8)


# exercise 9

N = 300
population = list(range(1, 301))

n = 15

k = 20

sample9 = population[::k]
print(sample9)

# exercise 10
# simple random vs systematic sampling

javob10A = 'agar bizda product qolimizda bosa yoki shu omborda turgan bolsa, biz randomni nechtadir olib sampling qisak boladi'
javob10B = 'agar product zavodda chiqayotgan bolsa , biz tohtatolmimizku u chiqaveradi chiqaveradi, oshanda systematic sampling qisak boladi'


# exercise 11 and exercise 12

lst = [12, 15, 14, 10, 13, 16, 11]

x = np.mean(lst)
s = np.std(lst, ddof=1)
s2 = np.var(lst, ddof=1)

print(x, s, s2)

# exercise 13

javob13 = 'shu biased bomasligi uchun ddof=1 ishlatardik samplega'

# exercise 14

def sample_stats(sample):

    n = len(sample)
    total = 0

    for x in sample:
        total += x

    mean = total / n # logika

    total_squared = 0

    for x in sample:
        total_squared += (x - mean) ** 2 # formula

    variance = total_squared / (n - 1)

    std = variance ** 0.5 # formula


    return {
        'mean': mean,
        'std': std,
        'variance': variance,
    }

# exercise 15

javob15 = 'shu ortachalarni taqsimote Sampling destribuition of the mean deyiladi, Parameters: E[x], standard error'


# exercise 16

population = np.random.normal(170, 10, 1000000)

means = []

for i in range(1000):
    sample = np.random.choice(population, size=50, replace=False)
    means.append(np.mean(sample))

plt.hist(means, bins=50)
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()



# exercise 17

population = np.random.randint(1, 7, 1000000)

means = []

for i in range(1000):
    sample = np.random.choice(population, size=50, replace=False)
    means.append(np.mean(sample))

plt.hist(means, bins=10)
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()


# exersice 18

sigma = 20
n = 25

std = 4 # 20 / 5


# exercise 19

sigma = 10

n = [1, 4, 25, 100, 400]

se = [10, 5, 2, 1, 0.5] # kallada hisobladim

# exersice 20
mu = 50
# Estimator A

xA = 50.1 # high var
biasA = xA - mu # 0.1 high bias

# Estimator B

xB = 55.0 # low var
biasB = xB - mu # 5.5 high bias