import numpy as np

# 1 task
data = np.array([
    [1,  2,  3,  5,  10,  7],
    [2,  4,  6, 10,  20, 14],
    [3,  6,  9, 15,  30, 21],
    [4,  8, 12, 20, 40, 28],
    [5, 10, 15, 25, 50, 35]
])

# 2 task
x = np.linalg.matrix_rank(data) # rank 1 chiqdi, yani bogliq ustunlar bor
print(x)

# 3 task

# columnlani ajratib olamiz datasetdan
v1 = data[:, 0]
v2 = data[:, 1]
v3 = data[:, 2]
v4 = data[:, 3]
v5 = data[:, 4]
v6 = data[:, 5]


# har bir columni birinchi columnga bogliqligini tekshiramiz
c2, *_ = np.linalg.lstsq(v1.reshape(-1, 1), v2, rcond=None)
c3, *_ = np.linalg.lstsq(v1.reshape(-1, 1), v3, rcond=None)
c4, *_ = np.linalg.lstsq(v1.reshape(-1, 1), v4, rcond=None)
c5, *_ = np.linalg.lstsq(v1.reshape(-1, 1), v5, rcond=None)
c6, *_ = np.linalg.lstsq(v1.reshape(-1, 1), v6, rcond=None)



# qanday boqligli , yani koeffitsiyenti chiqardim
print("col2 =", c2[0], "* col1") # col2 = 2.0 * col1
print("col3 =", c3[0], "* col1") # col3 = 3.0 * col1
print("col4 =", c4[0], "* col1") # col4 = 5.0 * col1
print("col5 =", c5[0], "* col1") # col5 = 10.0 * col1
print("col6 =", c6[0], "* col1") # col6 = 7.0 * col1




# 4 task
# hamma columnla faqat birinchi columnga bogliq ekan
clean_data = data[:, [0]]

print("Clean dataset:")
print(clean_data)

print("Old shape:", data.shape) # (5, 6)
print("New shape:", clean_data.shape) # (5, 1)
# shu yani bitta ustun qoldi