import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras

from keras.preprocessing import timeseries_dataset_from_array
from keras.layers import Input, LSTM, Dense
from keras import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


def visualize_loss(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Testing loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()


def show_plot(plot_data, delta, title, counter, mean, std):
    """ Show prediction plot. Draw a plot of 'past' values and show the prediction and the expected outcome.

    :param plot_data: history and expected outcome
    :param delta: steps ahead that are to be forecasted
    :param title: title of plot
    :param counter: current index of prediction
    :param mean: mean of training data of closing_price
    :param std: std of training data of closing_price
    :return: None
    """
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i] * std + mean, marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, (plot_data[i] * std + mean).flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.savefig(f'../output/predictions/prediction_{counter}.png')
    plt.close()
    return


def normalize(data, train_split):
    """ Get the standard score of the data.

    :param data: data set
    :param train_split: number of training samples
    :return: normalized data, mean, std
    """
    mean = data[:train_split].mean(axis=0)
    std = data[:train_split].std(axis=0)

    return (data - mean) / std, mean, std


def test_on_current_data(model, mean, std, path):
    """ Test model on current data

    :param model: Keras model
    :param mean: mean of training data
    :param std: std of training data
    :param path: path to csv file with current 100 values
    :return: expected, predicted
    """
    df = pd.read_csv(path, dtype=float)
    data = df[[df.columns[i] for i in range(1, len(df.columns))]]
    data.index = df['open_time']

    x = data.to_numpy()
    x = (x - mean) / std
    x = np.array([x])

    prediction = model.predict(x)

    show_plot([x[0][:, 3], np.array([(59292.16 - mean[3]) / std[3]]), prediction[0]], 1, "Single step prediction", 'now', mean[3], std[3])

    prediction = prediction * std[3] + mean[3]
    print(prediction)


def main():
    file_path = '../output/dataframes/BTCUSDT_m_15.csv'
    df = pd.read_csv(file_path, sep=',', dtype=float)

    # select features (currently all features)
    selected_features = [df.columns[i] for i in range(1, len(df.columns))]

    data = df[selected_features]
    data.index = df['open_time']

    # get index of value that has to be predicted. subtract 1 since open time column doesn't count
    predict_label_index = df.columns.get_loc('close') - 1

    # get number of samples for training set
    split_ratio = 0.7
    train_split = int(split_ratio * df.shape[0])

    # normalize data
    data, mean, std = normalize(data.values, train_split)
    # get DataFrame from np ndarray
    data = pd.DataFrame(data)

    # data sets
    training_data = data.loc[0:train_split - 1]
    test_data = data.loc[train_split:]

    # use 'past' timestamps to predict 'future' timestamps
    past = 100
    future = 10

    # set sampling rate
    sampling_rate = 1

    # set batch size and epochs
    batch_size = 128
    epochs = 50

    # get labels for training datasets
    start = past + future
    end = start + train_split

    x_train = training_data.values
    y_train = data.iloc[start:end][[predict_label_index]]

    # get subsets for training of model
    training_dataset = timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=past,
        sampling_rate=sampling_rate,
        batch_size=batch_size
    )

    # prepare validation data set
    x_end = len(test_data) - start
    label_start = train_split + start

    x_test = test_data.iloc[:x_end].values
    y_test = data.iloc[label_start:][[predict_label_index]]

    testing_dataset = timeseries_dataset_from_array(
        x_test,
        y_test,
        sequence_length=past,
        sampling_rate=sampling_rate,
        batch_size=batch_size
    )

    for batch in training_dataset.take(1):
        inputs, targets = batch

    # build the model
    input_layer = Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm = LSTM(32)(input_layer)
    output = Dense(1)(lstm)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss="mse")

    # set checkpoint and callback
    path_checkpoint = '../output/models/model_checkpoint.h5'
    callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    model_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        training_dataset,
        epochs=epochs,
        validation_data=testing_dataset,
        callbacks=[callback, model_checkpoint_callback]
    )
    
    model.save('../output/models/model.h5')

    visualize_loss(history, "Training and Validation loss")

    model = load_model('../output/models/model.h5')

    counter = 0
    for x, y in testing_dataset.take(10):
        show_plot(
            [x[0][:, 3].numpy(), y[0].numpy(), model.predict(x)[0]],
            1,
            "Single Step Prediction",
            counter,
            mean[3],
            std[3]
        )

        print(f'Prediction: {model.predict(x)[0] * std[3] + mean[3]}')
        counter += 1

    test_on_current_data(model, mean, std, '../output/dataframes/test.csv')


if __name__ == '__main__':
    main()
