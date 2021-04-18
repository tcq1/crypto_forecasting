import numpy as np
from datetime import datetime


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


def add_ma(df):
    """ Add moving averages to df

    :param df: candle data
    :return: df
    """
    try:
        df['ma_7'] = calc_moving_average(df['close'], 7)
        df['ma_30'] = calc_moving_average(df['close'], 30)
        df['ma_100'] = calc_moving_average(df['close'], 100)
    except IndexError:
        print(f'Couldn\'t add moving averages since there were not enough candles!')

    return df


def prepare_features(df):
    """ Prepare features for training.

    Convert open and close time to respective minute of the day.

    :param df: dataframe
    :return: dataframe
    """
    df['open_time'] = df['open_time'].apply(convert_timestamp_to_minutes)
    df['close_time'] = df['close_time'].apply(convert_timestamp_to_minutes)
    df = add_ma(df)

    return df


def convert_timestamp_to_minutes(timestamp):
    """ Convert a unix timestamp to the minute of a year.
    Example: 1616257800000 = 2021-03-20 16:30 -> minute 990

    :param timestamp: unix timestamp
    :return: int
    """
    # binance timestamp is in ms not in s --> divide by 1000
    timestamp /= 1000

    # get year of timestamp and subtract timestamp of year/1/1 to get timestamp without the year
    year = datetime.utcfromtimestamp(timestamp).year
    timestamp = timestamp - datetime(year, 1, 1).timestamp()

    # divide by 60 to get minutes
    return timestamp / 60
