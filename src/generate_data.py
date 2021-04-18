import pandas as pd
import numpy as np

from src.add_features import prepare_features


def calculate_change_label(last_close, close):
    """ Calculate the label for a given input. The label is the change of the closing price from the last closing price.

    :param last_close: closing price of last candle
    :param close: closing price of current candle
    :return: float
    """
    return (close - last_close) / last_close


def calculate_up_down_label(last_close, close):
    """ Calculate the label for a given input. The label is a boolean with True if the current closing time is higher
    than the last closing time and False if equal or lower.

    :param last_close: closing price of last candle
    :param close: closing price of current candle
    :return: boolean
    """
    return close > last_close


def calculate_next_closing_price(last_close, close):
    """ Return closing price

    :param last_close: closing price of last candle (not necessary)
    :param close: closing price of candle to predict
    :return: float
    """
    return close


def get_samples(time_series, feature_length=50, output_length=5, label_function=calculate_change_label):
    """ Divide time series into samples with feature_length inputs and the closing prices of the next output_length time steps being the output.

    The samples are of this form: [[features_0 features_1 ... features_feature_length-1], [close_feature_length+1 ... close_feature_length+output_length]]
    :param time_series: the time series as dataframe
    :param feature_length: number of time steps to use for one sample
    :param output_length: number of output values
    :param label_function: the function that calculates the label
    :return: features, labels
    """
    features = None
    labels = None
    current_index = 0

    # get samples
    while current_index + feature_length + 1 < len(time_series):
        feature = [time_series[current_index:current_index + feature_length].to_numpy()]
        last_close = time_series.iloc[current_index + feature_length - 1]['close']
        close = time_series[current_index + feature_length:current_index + feature_length + output_length]['close'].to_numpy()
        label = np.array([label_function(last_close, close)])

        if features is None:
            features = np.array(feature)
            labels = np.array(label)
        else:
            features = np.vstack((features, feature))
            labels = np.vstack((labels, label))
        current_index += feature_length

    return features, labels


def main():
    src_path = '../output/dataframes/test.csv'
    df = pd.read_csv(src_path, dtype=float)
    df = prepare_features(df)

    features, labels = get_samples(df, feature_length=3, output_length=1)
    print(features)
    print(labels)


if __name__ == '__main__':
    main()
