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


def make_net(interval: list, amount_of_nodes:int):
    """Разбивает интервал на отрезки по количеству необходимых узлов"""
    return np.linspace(interval[0], interval[1], amount_of_nodes)


# Задаем интервал, на котором определена ф-ция
my_interval = [5, 10]

# Разбиваем отрезок на 3 части (4 точки) для интерполяции по Лагранжу
x_net_lagrangian = make_net(my_interval, 4)

# Вычисляем значения на концах отрезка и в точках разбиения
y_net_lagrangian = f(x_net_lagrangian)


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


# Разбиваем отрезок на 7 частей(6 узлов) для интерполяции по Ньютону (deg = 5)
x_net_newton = make_net(my_interval, 6)
y_net_newton = f(x_net_newton)


def newton_interpolation(x_net, y_net, arg):
    """Интерполяция по Ньютону через разделенные разности"""
    n = len(x_net)
    x = symbols('x')

    differences = np.zeros((n, n))
    differences[:, 0] = y_net
    for i in range(1, n):
        for j in range(n - i):
            differences[j, i] = (differences[j + 1, i - 1] - differences[j, i - 1]) / (x_net[j + i] - x_net[j])

    coefficients_for_newton = differences[0, :]
    newton_function = coefficients_for_newton[0]
    polynom = 1
    for i in range(1, len(coefficients_for_newton)):
        polynom *= (x - x_net[i - 1])
        newton_function += coefficients_for_newton[i] * polynom

    newton_function_result = lambdify(x, newton_function)(arg)

    return newton_function_result


def plot_maker(x_net, y_net, function, interpolation_function, massage):
    x_values = np.linspace(x_net[0], x_net[-1], 50)

    interpolation_function_values = interpolation_function(x_net, y_net, x_values)

    plt.plot(x_values, function(x_values), 'k-', label='Исходная функция f(x)')
    plt.plot(x_values, interpolation_function_values, 'r--', label='Интерполяция')
    plt.scatter(x_net, y_net, color='r', zorder=5, label='Узлы')
    plt.title(massage)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_maker(x_net_lagrangian, y_net_lagrangian, f, newton_interpolation, 'Интерполяция по Ньютону')
plot_maker(x_net_lagrangian, y_net_lagrangian, f, lagrangian_interpolation, 'Интерполяция по Лагранжу')

