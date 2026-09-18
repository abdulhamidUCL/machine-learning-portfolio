import sympy as sp
import matplotlib.pyplot as plt
import numpy as np


# mini project
# 1 bosqich func yozish
def pastga_tush(f_hosila, x0, lr, qadamlar):

    x = x0
    trajectory = [x0]

    for i in range(qadamlar):
        hosila = f_hosila.subs('x', x)
        x = x - lr * float(hosila)
        trajectory.append(x)


    return trajectory


def f_hosila(func, symbol):
    x = sp.Symbol(str(symbol))
    new = sp.diff(func, x)

    return new



# 2 bosqich test
x = sp.Symbol('x')
func = (x - 5)**2

# Find derivative
derivative = f_hosila(func, x)

# Run gradient descent
trajectory = pastga_tush(
    derivative,
    x0=0,
    lr=0.1,
    qadamlar=20
)

print("Derivative:", derivative)
print("Trajectory:", trajectory)
print("Landed at:", trajectory[-1])




# 3 bosqich - grafik

x_vals = np.linspace(-2, 12, 400)
y_vals = (x_vals - 5) ** 2

plt.figure(figsize=(8, 6))

plt.plot(
    x_vals,
    y_vals,
    label="f(x) = (x - 5)²",
    color="black"
)

# Gradient descent trajectory
trajectory = pastga_tush(
    derivative,
    x0=0,
    lr=0.1,
    qadamlar=20
)

trajectory_y = [(x - 5) ** 2 for x in trajectory]

plt.scatter(
    trajectory,
    trajectory_y,
    color="red",
    zorder=5,
    label="Gradient Descent steps"
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent (lr=0.1)")
plt.legend()
plt.grid(True)
plt.show()


# 4 bosqich - different learning rates

learning_rates = [0.01, 0.5, 1.05]

for lr in learning_rates:

    trajectory = pastga_tush(
        derivative,
        x0=0,
        lr=lr,
        qadamlar=20
    )

    print(
        f"lr={lr}: "
        f"oxirgi x = {trajectory[-1]:.4f}, "
        f"oxirgi f(x) = {(trajectory[-1] - 5) ** 2:.4f}"
    )