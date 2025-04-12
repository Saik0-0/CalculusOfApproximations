import numpy as np

'''Взяли невырожденную систему уравнений:

1 * x_1 + 2 * x_2 + 3 * x_3 + 4 * x_4 + 5 * x_5 = 55
2 * x_1 + 1 * x_2 + 0 * x_3 + 1 * x_4 + 3 * x_5 = 23
0 * x_1 + 4 * x_2 + 1 * x_3 + -1 * x_4 + 3 * x_5 = 22
1 * x_1 + 0 * x_2 + 2 * x_3 + 1 * x_4 + 1 * x_5 = 16
3 * x_1 + 1 * x_2 + 0 * x_3 + 2 * x_4 + -1 * x_5 = 8

Её решение: x_1 = 1, x_2 = 2, x_3 = 3, x_4 = 4, x_5 = 5
'''

first_left_matrix = np.array([
    [1, 2, 3, 4, 5],
    [2, 1, 0, 1, 3],
    [0, 4, 1, -1, 3],
    [1, 0, 2, 1, 1],
    [3, 1, 0, 2, -1]
])

first_right_vector = np.array([
    [55],
    [23],
    [22],
    [16],
    [8]
])

# print(np.linalg.solve(first_left_matrix, first_right_vector))

extra_left_matrix = np.array([
    [1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2]
])

extra_right_vector = np.array([
    [100],
    [50]
])

A = np.vstack((first_left_matrix, extra_left_matrix))
b = np.vstack((first_right_vector, extra_right_vector))


def pseudosolution_func(left_part, right_vector):
    left_part_transposed = np.transpose(left_part)
    new_left_part = np.dot(left_part_transposed, left_part)
    new_right_part = np.dot(left_part_transposed, right_vector)

    pseudosolution = np.linalg.solve(new_left_part, new_right_part)

    return pseudosolution


print(pseudosolution_func(A, b))