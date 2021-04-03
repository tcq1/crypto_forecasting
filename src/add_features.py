import numpy as np


def calc_moving_average(data, n):
    """ Calculate the n-period moving average.

    :param data: array of prices
    :param n: period of moving window
    :return: moving average values of prices
    """
    data = np.asarray(data)
    weights = np.ones(n)
    weights /= weights.sum()

    # calculate ma values
    ma_values = np.convolve(data, weights, mode='full')[:len(data)]
    ma_values[:n] = ma_values[n]

    return ma_values
