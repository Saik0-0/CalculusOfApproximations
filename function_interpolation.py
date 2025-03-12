import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify


def f(x) -> float:
    """Задали дробно-рациональную функцию"""
    return 1 / (x ** 2 - 4)


def diff_x(function, arg) -> float:
    """Вычисляем значение производной переданной функции по x"""
    x = symbols('x')
    result_function = diff(function, x)
    return lambdify(x, result_function)(arg)


# Задаем интервал, на котором определена ф-ция
interval = [5, 10]

# Разбиваем отрезок на 3 части (4 точки)
x_net = np.linspace(interval[0], interval[1], 4)

# Вычисляем значения на концах отрезка и в точках разбиения
y_net = f(x_net)


def lagrangian_interpolation(x_net, y_net):
    n = len(x_net)
    x = symbols('x')
    w_n = 1
    for i in range(0, n):
        w_n *= (x - x_net[i])

    g = 0
    for x_i, y_i in zip(x_net, y_net):
        diff_w_n_calculated = diff_x(w_n, x_i)
        g += y_i * (w_n / ((x - x_i) * diff_w_n_calculated))

    g_result = lambdify(x, g)
    return g_result


x_plot = np.linspace(5, 10, 300)
y_lagrange = lagrangian_interpolation(x_net, y_net)(x_plot)

plt.plot(x_plot, f(x_plot), 'k-', label='Исходная функция f(x)')
plt.plot(x_plot, y_lagrange, 'r--', label='Интерполяция (Лагранж через разделённые разности)')
plt.scatter(x_net, y_net, color='r', zorder=5, label='Узлы Лагранжа (4 шт.)')

plt.show()


