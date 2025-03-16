import math

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


def make_grid(interval: list, amount_of_nodes:int):
    """Разбивает интервал на отрезки по количеству необходимых узлов"""
    return np.linspace(interval[0], interval[1], amount_of_nodes)


def lagrangian_interpolation(x_grid, y_grid, arg):
    """Интерполяция по Лагранжу через базисный полином"""
    n = len(x_grid)
    x = symbols('x')
    lagrange_function = 0

    for i in range(n):
        base_func_i = 1
        for j in range(n):
            if i != j:
                base_func_i *= (x - x_grid[j]) / (x_grid[i] - x_grid[j])
        lagrange_function += y_grid[i] * base_func_i

    lagrange_func = lambdify(x, lagrange_function, modules=["numpy"])
    lagrange_function_result = lagrange_func(arg)

    return lagrange_function_result


def newton_interpolation(x_grid, y_grid, arg):
    """Интерполяция по Ньютону через разделенные разности"""
    n = len(x_grid)
    x = symbols('x')

    differences = np.zeros((n, n))
    differences[:, 0] = y_grid
    for i in range(1, n):
        for j in range(n - i):
            differences[j, i] = (differences[j + 1, i - 1] - differences[j, i - 1]) / (x_grid[j + i] - x_grid[j])

    coefficients_for_newton = differences[0, :]
    newton_function = coefficients_for_newton[0]
    polynom = 1
    for i in range(1, len(coefficients_for_newton)):
        polynom *= (x - x_grid[i - 1])
        newton_function += coefficients_for_newton[i] * polynom

    newton_func = lambdify(x, newton_function, modules=["numpy"])
    newton_function_result = newton_func(arg)

    return newton_function_result


def make_chebyshev_grid(interval, deg):
    """Вычисление Чебышевской сетки"""
    a = interval[0]
    b = interval[1]
    grid = []
    for n in range(deg):
        node = 0.5 * (a + b) + 0.5 * (b - a) * math.cos((2 * n + 1) * math.pi / (2 * deg))
        grid.append(node)
    return np.array(grid)


def plot_maker(x_grid, y_grid, function, interpolation_function, massage):
    x_values = np.linspace(x_grid[0], x_grid[-1], 30)
    # x_values = x_grid.copy()
    interpolation_function_values = interpolation_function(x_grid, y_grid, x_values)

    plt.plot(x_values, function(x_values), 'k-', label='Исходная функция f(x)')
    plt.plot(x_values, interpolation_function_values, 'r--', label='Интерполяция')
    plt.scatter(x_grid, y_grid, color='r', zorder=5, label='Узлы')
    plt.title(massage)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Задаем интервал, на котором определена ф-ция
my_interval = [5, 10]


# Разбиваем отрезок на 3 части (4 точки) для интерполяции по Лагранжу (deg = 3)
x_grid_lagrangian = make_grid(my_interval, 4)
# Вычисляем значения на концах отрезка и в точках разбиения
y_grid_lagrangian = f(x_grid_lagrangian)


# Разбиваем отрезок на 7 частей(6 узлов) для интерполяции по Ньютону (deg = 5)
x_grid_newton = make_grid(my_interval, 6)
y_grid_newton = f(x_grid_newton)


# Разбиваем отрезок на 6 узла для интерполяции по Ньютону (deg = 5) по Чебышевской сетке
x_chebyshev_grid = make_chebyshev_grid(my_interval, 6)
y_chebyshev_grid = f(x_chebyshev_grid)


# Разбиваем отрезок на 3 узла для построения инт. кубического сплайна
x_splain_grid = make_grid(my_interval, 6)
y_splain_grid = f(x_splain_grid)


# plot_maker(x_grid_lagrangian, y_grid_lagrangian, f, newton_interpolation, 'Интерполяция по Ньютону')
# plot_maker(x_grid_lagrangian, y_grid_lagrangian, f, lagrangian_interpolation, 'Интерполяция по Лагранжу')
# plot_maker(x_chebyshev_grid, y_chebyshev_grid, f, lagrangian_interpolation, 'Интерполяция по Ньютону по Чебышевской сетке')