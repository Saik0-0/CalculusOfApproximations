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


def lagrangian_interpolation(x_net, y_net, arg):
    """Интерполяция по Лагранжу через базисный полином"""
    n = len(x_net)
    x = symbols('x')
    lagrange_function = 0

    for i in range(n):
        base_func_i = 1
        for j in range(n):
            if i != j:
                base_func_i *= (x - x_net[j]) / (x_net[i] - x_net[j])
        lagrange_function += y_net[i] * base_func_i

    lagrange_function_result = lambdify(x, lagrange_function)(arg)
    return lagrange_function_result


def plot_maker(x_net, y_net, function, interpolation_function):
    x_values = np.linspace(x_net[0], x_net[-1], 50)

    interpolation_function_values = interpolation_function(x_net, y_net, x_values)

    plt.plot(x_values, function(x_values), 'k-', label='Исходная функция f(x)')
    plt.plot(x_values, interpolation_function_values, 'r--', label='Интерполяция')
    plt.scatter(x_net, y_net, color='r', zorder=5, label='Узлы')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_maker(x_net, y_net, f, lagrangian_interpolation)

