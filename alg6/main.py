import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

VX = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
VY = [-1.86, -1.95, -2.12, -2.06, -2.15, -2.00, -2.12, -2.31, -2.29, -2.57, -2.56, -2.86, -2.85, -3.03, -3.25, -3.08, -3.29, -3.67, -3.70, -3.85]
#аппроксимация
def approximation(m):
    A = np.zeros((m + 1, m + 1))
    B = np.zeros(m + 1)
    n = 20
    for i in range(0, m + 1):
        for j in range(i, i + m + 1):
            sum1 = 0
            for t in range(0, n):
                sum1 = sum1 + VX[t] ** j
            A[i][j-i] = sum1
        sum2 = 0
        for j in range(0, n):
            sum2 = sum2 + VX[j] ** i * VY[j]
        B[i] = sum2
    return np.linalg.solve(A, B)

# метод для расчета СКО
def reg(t, s):
    return np.sqrt(np.sum(np.power((t - s), 2)) / len(t))

 # Линейная регрессия
def chart1():
    x = np.arange(0, 2, 0.01)
    pf = np.polyfit(VX, VY, 1)
    pv = np.polyval(pf, VX)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(VX, VY, "ro", markersize=5)
    ax.plot(VX, pv, "b-")
    plt.title("Линейная регрессия")
    plt.legend(["Исходные точки", "Регрессия"])
    ax.grid()
    plt.show()
#Метод для линейной регрессии
def F(x, a=1, b=2, c=1):
    np.seterr(divide="ignore", invalid="ignore")
    return a / x + b * x * x + c * np.e ** x

# Линейная регрессия общего вида
def chart2():
    x = np.arange(0, 2.01, 0.05)
    poly = curve_fit(F, VX, VY)[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(VX, VY, "ro", markersize=5)
    ax.plot(x, np.array([F(i, poly[0], poly[1], poly[2]) for i in x]), "b-")
    plt.title("Линейная регрессия общего вида")
    plt.legend(["Исходные точки", "Регрессия"])
    ax.grid()
    plt.show()
#Метод для аппроксимации по МНК
def fun(x, m):
    p = approximation(m)
    value = 0
    for i in range(m+1):
        value += p[i] * np.power(x, i)
    return value


# Аппроксимация по МНК при m = 1, 2, 3
def chart3():
    x = np.arange(0, 2.01, 0.01)
    plt.plot(VX, VY, "ro", markersize=5)
    plt.plot(x, fun(x, 1), "-b")
    plt.plot(x, fun(x, 2), "-g")
    plt.plot(x, fun(x, 3), "-y")
    plt.legend(["Исходные точки", "m = 1", "m = 2", "m = 3"])
    plt.title("Аппроксимация по МНК")
    plt.grid()
    plt.show()

# Аппроксимация по МНК при m = 4, 5, 6
def chart4():
    x = np.arange(0, 2.01, 0.01)
    plt.plot(VX, VY, "ro", markersize=5)
    plt.plot(x, fun(x, 4), "-b")
    plt.plot(x, fun(x, 5), "-g")
    plt.plot(x, fun(x, 6), "-y")
    plt.legend(["Исходные точки","m = 4", "m = 5", "m = 6"])
    plt.title("Аппроксимация по МНК")
    plt.grid()
    plt.show()

# Аппроксимация по МНК при m = 7, 8
def chart5():
    x = np.arange(0, 2.01, 0.01)
    plt.plot(VX, VY, "ro", markersize=5)
    plt.plot(x, fun(x, 7), "-b")
    plt.plot(x, fun(x, 8), "-g")
    plt.legend(["Исходные точки","m = 7", "m = 8"])
    plt.title("Аппроксимация по МНК")
    plt.grid()
    plt.show()

# Аппроксимация по МНК при m = 31
def chart6():
    x = np.arange(0, 2.01, 0.05)
    plt.plot(VX, VY, "ro", markersize=5)
    plt.plot(x, fun(x, 18), "-b")
    plt.legend(["Исходные точки", "m = 18"])
    plt.title("Аппроксимация по МНК")

    plt.grid()
    plt.show()

chart1()
chart2()
chart3()
chart4()
chart5()
chart6()

print("\nСреднеквадратичное отклонение для линейной регрессии общего вида:")
poly = curve_fit(F, VX, VY)[0]
st1 = np.array([F(i, poly[0], poly[1], poly[2]) for i in VX])
print("Своим методом:", reg(st1, VY))
print("Встроенным методом:", np.std(st1 - VY))

print("\nСреднеквадратичное отклонение для метода наименьших квадратов:")
st2 = [reg(fun(VX, i + 1), VY) for i in range(20)]
st3 = [np.std(fun(VX, i + 1) - VY) for i in range(20)]
for i in range(len(st2)):
    print(i + 1, "степень   Свой метод:", st2[i], "\tВстроенный метод:", st3[i])


