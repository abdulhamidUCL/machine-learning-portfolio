import numpy as np

# task 1 Diagonal amaliyoti

dig = np.diag([1, 1])
v = np.array([2, 3])

# birlar bogani uchun ozgarmidi
print(dig @ v) # [2 3] v ni ozi chiqdi




# task 2 Identity isboti
# 3×3 matritsa uchun A @ I = A va I @ A = A ekanini kod bilan ko'rsating.
I = np.eye(3)
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


print(np.allclose(I @ A, A)) # True
print(np.allclose(A @ I, A)) # True

# Yani I ni birinchi A ga yoki A ni birinchi I ga dot product qisayam bir hil chiqadi



# task 3 Qo'lda va kodda
# 2×2 matritsa teskarisini AVVAL qo'lda (formula bilan), keyin np.linalg.inv bilan toping. Solishtiring.

A = np.array([[1, 2],
              [3, 4]])

A_inv = np.linalg.inv(A)

print(A_inv)

# [[-2.   1. ]
#  [ 1.5 -0.5]]
# qolda hisoblaganim blan bir hil chiqdi


# task 4 Qiyin: singular tajriba
# Ustunlari bog'liq matritsa yasang. det = 0 ekanini va inv xato berishini tajriba bilan ko'rsating.


A = np.array([[1, 2],
              [2, 4]])

def inverse(A):
    if A.shape != A.T.shape:
        return 'matritsya kvadrat emas'

    elif np.linalg.det(A) == 0:
        return 'det = 0'

    else:
        return np.linalg.inv(A)


print(inverse(A))
# output : det = 0
# shart bajarildi ustunlari bir biriga bogliq bogani uchun determinant == 0 , va output ham biz aytgan narsani return qildi