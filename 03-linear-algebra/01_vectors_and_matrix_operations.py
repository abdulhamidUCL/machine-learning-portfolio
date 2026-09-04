import numpy as np
from scipy.special.cython_special import cosm1

# 1 Vektor amallari

a1 = np.array([3, 4, 5, 10, 12, 13, 15])
b1 = np.array([1, 7, 5, 8, 12, 17, 19])


print(a1 + b1)
print(a1 - b1)
print(np.linalg.norm(a1))
print(np.linalg.norm(b1))
print(a1 @ b1)


# 2 Kosinus o'xshashligi, formula: a @ b / (|a| * |b|)

Customer_A = np.array([2, 3, 1, 0])
Customer_B = np.array([3, 2, 1, 1])
Customer_C = np.array([0,1,4,3])

# 1 para A va B
cos1 = Customer_A @ Customer_B / (np.linalg.norm(Customer_A) * np.linalg.norm(Customer_B))

# 2 para A va C
cos2 = Customer_A @ Customer_C / (np.linalg.norm(Customer_A) * np.linalg.norm(Customer_C))

# 3 para B va C
cos3 = Customer_B @ Customer_C / (np.linalg.norm(Customer_B) * np.linalg.norm(Customer_C))


# 3 Matritsa ko'paytmasi
# formula: (m x n) @ (n x p) = (m x p)

Savolga_javob = '(3 x 2) @ (2 x 3) = (3 x 3)'

a3 = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])


b3 = np.array([
    [5, 6, 7],
    [1, 2, 3],
               ])

print(f'Dot product of a3 and b3:\n {np.dot(a3, b3)}')



# 4 Mini loyiha
# [m2, xonalar soni]
X = np.array([
    [45, 1],
    [60, 2],
    [65, 3],
    [70, 4],
    [80, 5],
])

# 1 m2 = 15 mln, 1 xona = 30 mln

w = np.array([15, 30])
b = 10

# formula: y = X @ w + b
# bu bizga uylarni narhlari vectri beradi

y = X @ w + b
print(y)