import numpy as np
import matplotlib.pyplot as plt

# task 1
A = np.array([[1, 1],
              [0, 1]])

v = np.array([2, 1])

w = A @ v

plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='gray')

plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='green')

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True)
plt.show()

# task 2

theta = np.radians(45)

R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

print(R @ [1, 0])  # [0.70710678 0.70710678]

# task 3

kv = np.array([[0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0]])

tlar = {
    # 'Scaling': [[2, 0], [0, 0.5]],
    # 'Rotation': [[0, -1], [1, 0]],
    # 'Reflection': [[1, 0], [0, -1]],
    # 'Shear': [[1, 1], [0, 1]],
    'b': [[0, 1], [-1, 0]],
    'c': [[-1, 0], [0, 1]],
    'd': [[0, 0], [0, 0]]
}

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
for ax, (nom, T) in zip(axes, tlar.items()):
    yangi = np.array(T) @ kv
    ax.plot(kv[0], kv[1], 'gray')
    ax.plot(yangi[0], yangi[1], 'green')
    ax.set_title(nom)
    ax.grid(True)
    ax.axis('equal')

plt.show()

tlar2 = {
    'Scaling': [[2, 0], [0, 0.5]],
    'Shear': [[1, 1], [0, 1]],
}

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

yangi = kv

for ax, (nom, T) in zip(axes, tlar2.items()):
    yangi = np.array(T) @ yangi

    ax.plot(kv[0], kv[1], 'gray')
    ax.plot(yangi[0], yangi[1], 'green')
    ax.set_title(nom)
    ax.grid(True)
    ax.axis('equal')

plt.show()

# task 4

# a . avval 90 gradusga burib 3X chozish, find matrix and [1, 0]
tlar3 = {
    'Rotation': [[0, -1], [1, 0]],
    'Scaling': [[3, 0], [0, 1]],
}

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

yangi = kv

for ax, (nom, T) in zip(axes, tlar3.items()):
    yangi = np.array(T) @ yangi

    ax.plot(kv[0], kv[1], 'gray')
    ax.plot(yangi[0], yangi[1], 'green')
    ax.set_title(nom)
    ax.grid(True)
    ax.axis('equal')
    ax.set_xlim(-4, 2)
    ax.set_ylim(-1, 2)

plt.show()


# now only with code

S = np.array([[2, 0], [0, 1]]) # chozish -> longer
R = np.array([[0, -1], [1, 0]])# 90 degree rotation
v = np.array([1, 0])

print(R @ S)
print(S @ R)


print(R @ S @ v)
print(S @ R @ v)
print(np.allclose(S @ R , R @ S))


# R45 = np.array([[0, -1], [1, 0]])
# R90 = np.radians([[0, -0,707], [0.707, 0]])
# print(np.allclose(R45 @ R90 , R45 @ R90))


# task 5
# constructing letter N
img = np.zeros((8,8))

img [1:7, 1] = 1
img [1:7, 6] = 1

for i in range (1, 7):
    img[i, i] = 1

plt.imshow(img, cmap='gray')
plt.show()


img_rotated = np.rot90(img, -1)

plt.imshow(img_rotated, cmap='gray')
plt.show()



# manually rotating
img = np.zeros((8,8))

img [1, 1:7] = 1
img [6, 1:7] = 1

for i in range (1, 7):
    img[i, 7 - i] = 1

plt.imshow(img, cmap='gray')
plt.show()