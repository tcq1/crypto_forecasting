import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from src.generate_data import get_samples
from keras.layers import Input, Flatten, Dense, Dropout
from keras import Sequential


def get_datasets(time_series_path, feature_length, output_length, train_size):
    """ Get training and validation feature and label sets.

    :param time_series_path: path to time series
    :param feature_length: number of past values to use for prediction
    :param output_length: number of future values to predict
    :param train_size: size of training set
    :return: x_train, x_test, y_train, y_test
    """
    # load time series
    time_series = pd.read_csv(time_series_path)

    # extract samples
    features, labels = get_samples(time_series, feature_length, output_length)

    # split samples
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=train_size)

    return x_train, x_test, y_train, y_test


def normalize_data(data, mean=None, std=None):
    """ Normalize data by calculating the Z-score.

    :param data: data set not flattened
    :param mean: array of mean of all columns of training data, if None then find the mean in the data set
    :param std: array of std of all columns of training data, if None then find the std in the data set
    :return: data, mean, std
    """
    if mean is None or std is None:
        # stack samples
        stacked = np.vstack((sample for sample in data))
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)

    # normalize samples
    for sample in range(len(data)):
        data[sample] = (data[sample] - mean) / std

    return data, mean, std


def create_model(input_shape, hidden_layers, width, dropout,
                 activation_function, output_activation, optimizer, loss, metrics):
    """ Create the model. This is just a plain feed forward neural network.

    :param input_shape: shape of input
    :param hidden_layers: number of hidden layers
    :param width: width of hidden layers
    :param dropout: dropout rate
    :param activation_function: activation function in hidden layers
    :param output_activation: activation function of output layer
    :param optimizer: model optimizer
    :param loss: loss function
    :param metrics: metrics
    :return: model
    """
    model = Sequential()
    # input layer
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    # hidden layers with dropout
    for i in range(hidden_layers):
        model.add(Dropout(dropout))
        model.add(Dense(width, activation=activation_function))

    # output layer
    model.add(Dense(1, output_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def main():
    # variables

    # path to time series
    time_series_path = '../output/dataframes/BTCUSDT/BTCUSDT_m_05.csv'
    # number of past values to use for prediction
    feature_length = 50
    # number of future values to predict
    output_length = 1
    # size of training set
    train_size = 0.7

    # model params
    epochs = 256
    batch_size = 32
    dropout = 0.2

    # load data sets
    print('Loading data sets...')
    x_train, x_test, y_train, y_test = get_datasets(time_series_path, feature_length, output_length, train_size)

    # normalize data
    x_train, mean, std = normalize_data(x_train)
    x_test = normalize_data(x_test, mean, std)[0]

    model = create_model(input_shape=x_train[0].shape,
                         hidden_layers=4,
                         width=128,
                         dropout=dropout,
                         activation_function='relu',
                         output_activation='relu',
                         optimizer='sgd',
                         loss='mean_absolute_error',
                         metrics=['mean_absolute_error'])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()
    # plt.savefig('history.png')


if __name__ == '__main__':
    main()
