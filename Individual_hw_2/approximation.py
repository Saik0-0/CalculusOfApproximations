import numpy as np
import sympy as sp


x = sp.symbols('x')

my_polynom = 4 * (x ** 5) + 3 * (x ** 4) + 1 * (x ** 3) + 6 * (x ** 2) + 3 * x + 9
my_polynom_degree = 5
my_diapason = [1, 3]


def replace_x(diapason, x_1):
    return (2 * x_1 - (diapason[1] + diapason[0])) / (diapason[1] - diapason[0])


x_2 = replace_x(my_diapason, x)

basis_0 = 1
basis_1 = x
basis_2 = x ** 2
basis_3 = x ** 3
basis_4 = x ** 4

basis = [basis_0, basis_1, basis_2, basis_3, basis_4]

legendre_0 = 1
legendre_1 = x_2
legendre_2 = sp.Rational(1, 2) * (3 * (x_2 ** 2) - 1)
legendre_3 = sp.Rational(1, 2) * (5 * (x_2 ** 3) - 3 * x_2)
legendre_4 = sp.Rational(1, 8) * (35 * (x_2 ** 4) - 30 * (x_2 ** 2) + 3)

legendre_basis = [legendre_0, legendre_1, legendre_2, legendre_3, legendre_4]
# print(legendre_basis)


def approximate(polynom, diapason, polynom_degree, basis_list):
    x = sp.symbols('x')

    gram_matrix = np.zeros((polynom_degree, polynom_degree))
    for line_ind, line in enumerate(gram_matrix):
        for row_ind, _ in enumerate(line):
            gram_matrix[line_ind, row_ind] = sp.N(sp.integrate(basis_list[line_ind] * basis_list[row_ind],
                                                          (x, diapason[0], diapason[1])))

    right_vector = np.zeros((polynom_degree, 1))
    for line_ind, _ in enumerate(right_vector):
        right_vector[line_ind, 0] = sp.N(sp.integrate(polynom * basis_list[line_ind], (x, diapason[0], diapason[1])))

    coefficients_vector = np.linalg.solve(gram_matrix, right_vector)

    result_polynom = 0
    for ind, c in enumerate(coefficients_vector):
        result_polynom += c[0] * (x ** ind)

    perturbed_gram_matrix = gram_matrix + 0.001
    perturbed_right_vector = right_vector + 0.001

    perturbed_coefficients_vector = np.linalg.solve(perturbed_gram_matrix, perturbed_right_vector)
    delta = np.linalg.norm(perturbed_coefficients_vector - coefficients_vector) / np.linalg.norm(coefficients_vector)
    return result_polynom, float(delta)


approximating_polynom_1, delta_1 = approximate(my_polynom, my_diapason, my_polynom_degree, basis)
print(f'Приближающий полином по базисам: {approximating_polynom_1}, абсолютная погрешность: {delta_1: .17f}')

approximating_polynom_2, delta_2 = approximate(my_polynom, my_diapason, my_polynom_degree, legendre_basis)
print(f'Приближающий полином по Лежандру: {approximating_polynom_2}, абсолютная погрешность: {delta_2: .17f}')
