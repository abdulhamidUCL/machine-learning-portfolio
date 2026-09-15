import numpy as np
import matplotlib.pyplot as plt


# task 1

# 1)
def f(x):
    return 3000 * x + 5000


# 2)
#  x uchun 4 ga chapdan yaqinlashuvchi ro'yxat tuzing: [3.9, 3.99, 3.999].
print(" 4 ga chapdan: ")
for x in [3.9, 3.99, 3.999]:
    print(x, " -> ", f(x))

# 3)
# x uchun 4 ga o'ngdan yaqinlashuvchi ro'yxat tuzing: [4.001, 4.01, 4.1].
print("4 ga ongdan: ")
for x in [4.001, 4.01, 4.1]:
    print(x, " -> ", f(x))

# 4) natijalarni solishtirish

#  4 ga chapdan:
# 3.9  ->  16700.0
# 3.99  ->  16970.0
# 3.999  ->  16997.0
# 4 ga ongdan:
# 4.001  ->  17003.0
# 4.01  ->  17030.0
# 4.1  ->  17300.0

# 5)
# 17000 ga yaqinlashayapti , lekin aynan shu bolmaydi, chapdan ham ongdan ham shu songa intilayapti


print("------------------------------------------------------------------------------------------------")


# task 2

# 1)
def c(x):
    return (x ** 2 - 9) / (x - 3)


# 2)
try:
    c(3)
except ZeroDivisionError:
    print("Can't divide by zero")

# 3)
for x in [2.9, 2.99, 2.999, 3.001, 3.01, 3.1]:
    print(x, " -> ", c(x))

# 4)
# 2.9  ->  5.899999999999993
# 2.99  ->  5.990000000000023
# 2.999  ->  5.99899999999986
# 3.001  ->  6.00100000000014
# 3.01  ->  6.009999999999977
# 3.1  ->  6.100000000000007

# x 3 ga intilayotganda c(x) 6 ga intilaypti


# 5)
fig, axes = plt.subplots(1, 1, figsize=(14, 6))

x = np.linspace(2, 4, 600)
x = x[x != 3]

axes.scatter(x, c(x), s=2)
axes.scatter(3, 6, s=80, facecolors='none', edgecolors='red')

axes.grid(True)
plt.show()


print("------------------------------------------------------------------------------------------------")
# task 3

# 1)
def p(x):
    if x < 500000:
        return x  # chegirma yoq
    else:
        return 0.9 * x  # 10% chegirma blan


# 2)
print('chap taraf')
for x in [499000, 499900, 499990, 499999]:
    print(x, " -> ", p(x))


# 3)
print('ong taraf')
for x in [500000, 500001, 500010, 500100, 501000]:
    print(x, " -> ", p(x))


# 4)
# chap taraf
# 499000  ->  499000
# 499900  ->  499900
# 499990  ->  499990
# 499999  ->  499999
# ong taraf
# 500000  ->  450000.0
# 500001  ->  450000.9
# 500010  ->  450009.0
# 500100  ->  450090.0
# 501000  ->  450900.0

# Bu yerda korsak boladi bizda jump(sakrash) ketmoqda, 499000 ketayaptida keyin 450000 bop ketayapti


# 5)
# x = 300000 dan 700000 gacha scatter bilan chizing

x = np.linspace(300000, 700000, 1000)
y = [p(val) for val in x]

plt.figure(figsize=(14, 6))
plt.scatter(x, y, s=2)

plt.grid(True)
plt.show()

# uzulish borligi korindi :)


# 6)

# Bu grafik step activation function ga o'xshaydi, chunki 500000 nuqtada keskin sakrash mavjud.
# Bunday sakrash gradient descent uchun muammo boladi, chunki gradient bu nuqtada aniqlanmagan va model qaysi tomonga harakat qilishni bilishi qiyin.