import numpy as np
from scipy import stats


# exercise 1

mu = 500
sigma = 40 # sigma given thats z-test

n = 64
x_mean = 490

# formula z = (x - mu) / (sigma / (n ** 0.5))

z = (x_mean - mu) / (sigma / (n ** 0.5))
# print(z)

p_value = 2 * (1 - stats.norm.cdf(abs(z)))# two tailed shunga 2 ga kopaytiramiz
# print(p_value)

# alpha = 0.05 and p = 0.04550026389635842
# p < alpha
# it is less than alpha that's why hypothesis is rejected
# the average lifetime is not 500 hours




# exercise 2

mu = 30

n = 20
x_mean = 32.5
s = 5 # sigma not given thats why t-test

#formula pochti bir hil t = (mu - x) / (s / (n ** 0.5))
t = (mu - x_mean) / (s / (n ** 0.5))
# print(t)

p_value = 2 * (1 - stats.t.cdf(abs(t), df=n-1)) # two tailed shunga 2 ga kopaytiramiz
# print(p_value)

# alpha = 0.05 and p = 0.03754054954852504
# p < alpha
# it is again less than alpha we reject it
# therefore, true average delivery time differs from 30 minutes.


# exercise 3

fertilizer_A = [4.2, 4.5, 4.0, 4.3, 4.6, 4.1, 4.4, 4.2] # t-test, its samples
fertilizer_B = [4.8, 5.1, 4.9, 5.0, 4.7, 5.2, 4.9, 5.0]

t_test , p_value = stats.ttest_rel(fertilizer_A, fertilizer_B) # pair method chunki ikta
# print(t_test) #-6.302636934111969
# print(p_value) #0.00040314103245355057


# alpha = 0.05 and p = 0.00040314103245355057
# p < alpha
# its less
# fertilizer_B increases (helps to crops), there is sufficient evidence