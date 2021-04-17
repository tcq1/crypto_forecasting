import numpy as np

from binance.client import Client
from keras.models import load_model

from src.generate_data import calculate_next_closing_price, prepare_features, get_samples
from src.load_candles import get_api_information, get_klines_df, add_ma
from src.regressor import normalize_data


def main():
    # model
    model_dir = '../output/models/model2'
    # load the model and normalization variables
    print('Loading model...')
    model = load_model(f'{model_dir}/model.h5')
    mean = np.load(f'{model_dir}/mean.npy')[1:]
    std = np.load(f'{model_dir}/std.npy')[1:]

    # load data to evaluate
    # number of candles to use as features
    feature_length = 50
    # number of candles to predict
    output_length = 1
    # interval
    interval = Client.KLINE_INTERVAL_15MINUTE
    # label function
    label = calculate_next_closing_price

    # binance client
    key, secret = get_api_information('api_key')
    client = Client(key, secret)
    df = get_klines_df(client, symbol='BTCUSDT', interval=interval, limit=feature_length + 100).astype(float)
    # calculate moving averages
    df = add_ma(df)
    # only take feature_length values
    df = df[-50:]
    # prepare features
    df = prepare_features(df)

    # get feature data
    features = df.to_numpy()

    # normalize data
    features = (features - mean) / std
    # flatten
    features = features.flatten()

    print(f'Prediction for the next candle: ${model.predict(np.array([features]))[0][0]}')


if __name__ == '__main__':
    main()
