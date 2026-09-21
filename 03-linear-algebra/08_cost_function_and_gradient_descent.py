import numpy as np
import matplotlib.pyplot as plt

# task 1 cost hisoblash

x = np.array([-3, -2, -1, 0, 1, 2])
y = np.array([-3, -1, 0, 1, 2, 3])

def cost(w):
    return np.mean((w * x - y) ** 2)

for w in [-2, -1, 0, 1, 2, 3, 4, 5]:
    c = cost(w)

    print("w: ", w," cost: ", c)

# w:  -1  cost:  0.027777777777777776 eng kami shu chiqdi, yani weight -1 paytida cost eng kam boladi



# task 2 gradient descent + tohtash sharti

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])


def gradient(w):
    return np.mean(2 * x * (w * x - y))


w = 0
lr = 0.01
count = 1
tarix = [w]

while True:
    old_w = w
    w = w - lr * gradient(w) # yangi weightlani formula blan hisoblash
    count += 1
    tarix.append(w) # listga solish

    if abs(old_w - w) < 1e-5:
        break

print("last w: ", w)
print(count)


plt.figure(figsize=(8, 8))

plt.plot(tarix)
plt.xlabel("Iteration")
plt.ylabel("Weight")
plt.grid()
plt.show()



# task 3 bias blan

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

def gradient(w, b):
    prediction = w * x + b
    error = prediction - y

    # Partial derivatives of the cost function
    dw = np.mean(2 * x * error)
    db = np.mean(2 * error)

    return dw, db


w = 0
b = 0
lr = 0.01
count = 1
tarix_w = [w]
tarix_b = [b]

while True:
    old_w = w
    old_b = b


    dw, db =gradient(w, b)

    w = w - lr * dw
    b = b - lr * db

    tarix_w.append(w)
    tarix_b.append(b)
    count += 1

    if abs(old_w - w) < 1e-5 and abs(old_b - b) < 1e-5:
        break



plt.figure(figsize=(8, 8))
plt.plot(tarix_w, label="w")
plt.plot(tarix_b, label="b")

plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Gradient Descent Convergence")

plt.legend()
plt.show()
