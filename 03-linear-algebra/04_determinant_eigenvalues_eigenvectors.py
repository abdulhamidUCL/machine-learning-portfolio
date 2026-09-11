import numpy as np

# task 1 — Determinant as area scaling

matrices = {
    "M1": np.array([[2, 0], [0, 5]]),
    "M2": np.array([[1, 0], [0, 1]]),
    "M3": np.array([[3, 6], [1, 2]]),
}

# 1a
# Qolda oldindan hisobladim:
# M1 -> 2*5 - 0*0 = 10
# M2 -> 1*1 - 0*0 = 1
# M3 -> 3*2 - 6*1 = 0
# det = 0 bolsa matritsa singular boladi

# 1b
for name, M in matrices.items():
    det = np.linalg.det(M)
    print(f"det({name}) = {det}")


# 1c
# M3 ning columnlari bir-biriga bogliq:
# col2 = 2 * col1
# Shuning uchun ular mustaqil yo'nalish bermaydi.
# Geometrik tomondan parallelogramm area = 0 ga collapse boladi.
# Shuning uchun det(M3) = 0 va inverse mavjud emas.


# task 2 — Is v an eigenvector of A?

def task2_check_eigenvector(A, v):
    # 2a
    # Av ni hisoblaymiz
    Av = A @ v

    # 2b
    # Agar Av = lambda * v bolsa, v eigenvector hisoblanadi.
    # Zero values bo'lishi mumkinligi uchun direct division qilmaymiz.
    # Instead, lambda ni non-zero element orqali topamiz.

    non_zero = np.abs(v) > 1e-12

    if not np.any(non_zero):
        return False, None

    eigenvalue = Av[non_zero][0] / v[non_zero][0]

    # Av ning hamma elementlari lambda * v ga tengligini tekshiramiz
    is_eigenvector = np.allclose(Av, eigenvalue * v)

    if not is_eigenvector:
        eigenvalue = None

    return is_eigenvector, eigenvalue


A = np.array([[4, 2], [1, 3]])
candidates = [
    np.array([2, 1]),
    np.array([1, 1]),
    np.array([1, -1])
]

for v in candidates:
    is_eig, val = task2_check_eigenvector(A, v)
    print(f"v={v} -> eigenvector? {is_eig}, eigenvalue = {val}")

# 2c
# Kandidatlar ichida bir xil direction dagi vector yo'q.
# Masalan, [2,1] va [1,1] bir-birining scalar multiple'i emas.
# Shuning uchun ular same direction emas.


# task 3 — Solve the characteristic equation by hand

A = np.array([[4, 2], [1, 3]])

# 3a
# det(A - lambda*I) = 0
#
# |4-lambda   2       |
# |1          3-lambda|
#
# (4-lambda)(3-lambda) - 2 = 0
# lambda^2 - 7lambda + 10 = 0
# (lambda - 5)(lambda - 2) = 0
#
# lambda = 5 yoki lambda = 2

# 3b
eigenvalues, eigenvectors = np.linalg.eig(A)

print("eigenvalues:", eigenvalues)
print("eigenvectors (columns!):\n", eigenvectors)

# 3c
# np.linalg.eig eigenvectorlarni COLUMNS sifatida qaytaradi.
# Shuning uchun birinchi eigenvector = eigenvectors[:, 0]

v0 = eigenvectors[:, 0]
check = np.allclose(A @ v0, eigenvalues[0] * v0)

print("Check eigenpair 0 is valid:", check)

# task 4 — Diagonal matrices are the easy case

D = np.diag([2, 3])

# 4a
# Diagonal matrixda eigenvalues diagonal elementlarning o'zi:
# lambda1 = 2
# lambda2 = 3
#
# Eigenvectors esa standard basis vectors:
# [1, 0] va [0, 1]

# 4b
vals, vecs = np.linalg.eig(D)

print("D eigenvalues:", vals)
print("D eigenvectors:\n", vecs)

# 4c
# D = diag([7,7]) bo'lsa, ikkala direction ham lambda = 7 bilan
# eigenvector bo'ladi.
# 2x2 matrix bo'lgani uchun 2 ta independent eigenvector direction
# olishimiz mumkin.
# Bu holatda eigenvectors unique emas, istalgan independent vectorlar
# shu eigenspace ichida eigenvector bo'lishi mumkin.


# task 5 — det & trace as shortcuts for eigenvalues

N = np.array([[3, 6], [1, 2]])

# 5a
eigenvalues, _ = np.linalg.eig(N)

# 5b
# Eigenvalues product = determinant
product_check = np.allclose(
    np.prod(eigenvalues),
    np.linalg.det(N)
)

# 5c
# Eigenvalues sum = trace
sum_check = np.allclose(
    np.sum(eigenvalues),
    np.trace(N)
)

print("eigenvalues:", eigenvalues)
print("product of eigenvalues == det(N)?", product_check)
print("sum of eigenvalues == trace(N)?", sum_check)

# 5d
# N singular matrix va det(N) = 0.
# Eigenvalues producti det ga teng.
# Product 0 bo'lishi uchun kamida bitta eigenvalue 0 bo'lishi kerak.
# Shuning uchun bu yerda bitta eigenvalue = 0.


# task 6 Mini bridge to PCA

# Small synthetic dataset
rng = np.random.default_rng(42)
x = rng.normal(0, 3, 200)
y = x * 0.5 + rng.normal(0, 0.5, 200)

data = np.column_stack([x, y])

# 6a
# Har bir columnning meanini topib, datadan ayiramiz.
# Bu centering deyiladi.
centered = data - np.mean(data, axis=0)

# 6b
# np.cov variablesni rows sifatida kutadi,
# shuning uchun centered.T ishlatamiz.
cov = np.cov(centered.T)

print("Covariance matrix:\n", cov)

# 6c
# Covariance matrixning eigenvalues va eigenvectorsini topamiz.
eigenvalues, eigenvectors = np.linalg.eig(cov)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# 6d
# Eng katta eigenvalue maximum variance directionni bildiradi.
# argmax orqali eng katta eigenvalue indexini topamiz.
max_index = np.argmax(eigenvalues)

principal_direction = eigenvectors[:, max_index]

print("Principal direction of the data:", principal_direction)
