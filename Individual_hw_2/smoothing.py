import math
import numpy as np
from matplotlib import pyplot as plt


def func(t):
    return math.sin(4 * (t ** 3))


diapason = [0, 1]

parted_diapason = np.linspace(diapason[0], diapason[1], 500 + 1)
signals = list(map(func, parted_diapason))

# Генерируем шум с нормальным распределением
noise_std = 0.2  # стандартное отклонение шума
noise = noise_std * np.random.randn(len(signals))

# Добавляем шум к сигналу
noisy_signals = signals + noise


def moving_average(signal, k):
    alpha = 1 / k
    smoothed_signal = np.zeros(len(signal))
    smoothed_signal[:k - 1] = signal[:k - 1].copy()
    for s in range(k, len(signal)):
        for i in range(k - 1):
            smoothed_signal[s] += alpha * signal[s - i]
    return smoothed_signal


def moving_median(signal, k):
    smoothed_signal = np.zeros(len(signal))
    smoothed_signal[:k - 1] = signal[:k - 1].copy()
    for i in range(len(signal) - 3):
        smoothed_signal[i + 3] = np.median(signal[i:i+4])
    return smoothed_signal


def exponential_moving_average(signal, alpha):
    smoothed_signal = [signal[0].copy()]
    for i in range(1, len(signal)):
        smoothed_signal_i = alpha * signal[i] + (1 - alpha) * smoothed_signal[i - 1]
        smoothed_signal.append(smoothed_signal_i)
    return smoothed_signal


def plot_maker(parted_diapason, noisy_signal, smoothed_signal, label):
    plt.figure(figsize=(10, 6))
    plt.plot(parted_diapason, smoothed_signal, label=label)
    plt.plot(parted_diapason, noisy_signal, label='Сигнал с шумом', alpha=0.7)
    plt.title('Сглаживание')
    plt.legend()
    plt.show()


moving_average_signal_1 = moving_average(noisy_signals, 10)
moving_average_signal_2 = moving_average(noisy_signals, 20)
moving_average_signal_3 = moving_average(noisy_signals, 40)
label_for_plot_1 = 'Сглаживание скользящим средним'

moving_median_signal_1 = moving_median(noisy_signals, 10)
moving_median_signal_2 = moving_median(noisy_signals, 20)
moving_median_signal_3 = moving_median(noisy_signals, 43)
label_for_plot_2 = 'Сглаживание скользящей медианой'

exponential_signal_1 = exponential_moving_average(noisy_signals, 0.001)
exponential_signal_2 = exponential_moving_average(noisy_signals, 0.15)
exponential_signal_3 = exponential_moving_average(noisy_signals, 0.75)
label_for_plot_3 = 'Экспоненциальное сглаживание'

plot_maker(parted_diapason, noisy_signals, exponential_signal_2, label_for_plot_3)