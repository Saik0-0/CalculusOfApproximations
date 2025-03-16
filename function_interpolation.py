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


def cubic_spline(x_grid, y_grid, arg):
    """Построение интерполяционного кубического сплайна"""
    n = len(x_grid) - 1
    h = np.diff(x_grid)

    beta = np.zeros(n - 1)
    for i in range(1, n):
        beta[i - 1] = 3 * ((y_grid[i + 1] - y_grid[i]) / h[i] - (y_grid[i] - y_grid[i - 1]) / h[i - 1])

    coef_matrix = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        if i > 0:
            coef_matrix[i, i - 1] = h[i]  # Нижняя диагональ
        if i < n - 2:
            coef_matrix[i, i + 1] = h[i + 1]  # Верхняя диагональ
        coef_matrix[i, i] = 2 * (h[i] + h[i + 1])  # Главная диагональ

    # Матрица для вторых производных в узлах (коэффициент a_2)
    matrix = np.zeros(n + 1)
    matrix[1:n] = np.linalg.solve(coef_matrix, beta)

    a_0 = y_grid[:-1]
    a_1 = np.zeros(n)
    a_2 = matrix[:-1]
    a_3 = np.zeros(n)

    for i in range(n):
        a_3[i] = (matrix[i + 1] - matrix[i]) / (3 * h[i])
        a_1[i] = (y_grid[i + 1] - y_grid[i]) / h[i] - a_2[i] * h[i] - a_3[i] * h[i] * h[i]

    y_result = np.zeros_like(arg)
    for j, x in enumerate(arg):
        # определение отрезка интерполяции (где лежит x)
        for i in range(n):
            if x_grid[i] <= x <= x_grid[i + 1]:
                dx = x - x_grid[i]
                y_result[j] = a_0[i] + a_1[i] * dx + a_2[i] * (dx ** 2) + a_3[i] * (dx ** 3)
                break

    return y_result


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


plot_maker(x_grid_lagrangian, y_grid_lagrangian, f, newton_interpolation, 'Интерполяция по Ньютону')
plot_maker(x_grid_lagrangian, y_grid_lagrangian, f, lagrangian_interpolation, 'Интерполяция по Лагранжу')
plot_maker(x_chebyshev_grid, y_chebyshev_grid, f, lagrangian_interpolation, 'Интерполяция по Ньютону по Чебышевской сетке')
plot_maker(x_splain_grid, y_splain_grid, f, cubic_spline, 'Построение интерполяционного кубического сплайна')