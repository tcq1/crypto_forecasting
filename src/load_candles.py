import pandas as pd
import datetime

from binance.client import Client
from timeit import default_timer as timer


def get_api_information(file_path):
    """ Get binance api key and api secret

    :param file_path: path to file with api information
    :return: api_key, api_secret
    """
    with open(file_path, 'r') as file:
        content = file.read().splitlines()
        key = content[0]
        secret = content[1]

    return key, secret


def get_klines_df(client, symbol, interval, start_date=None, end_date=None, limit=1000):
    """ Get klines as a pandas dataframe.

    :param client: Binance client
    :param symbol: Exchange symbol
    :param interval: Candle interval
    :param start_date: Start time as unix timestamp
    :param end_date: End time as unix timestamp
    :param limit: Maximum number of candles returned
    :return: dataframe
    """
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_trades', 'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore']
    klines = client.get_klines(symbol=symbol, interval=interval, startTime=start_date, endTime=end_date, limit=limit)
    df = pd.DataFrame(data=klines, columns=columns)
    del df['ignore']

    return df


def last_close_time_before_now(df):
    """ Check if the last close time is before current time.

    :param df: Dataframe with candles
    :return: boolean
    """
    now = datetime.datetime.now().timestamp() * 1000

    return now > df.iloc[-1]['close_time']


def kline_intervals():
    """ Dictionary containing all intervals and their respective minute values.

    :return: dictionary
    """
    return {Client.KLINE_INTERVAL_1MINUTE: 1,
            Client.KLINE_INTERVAL_3MINUTE: 3,
            Client.KLINE_INTERVAL_5MINUTE: 5,
            Client.KLINE_INTERVAL_15MINUTE: 15,
            Client.KLINE_INTERVAL_30MINUTE: 30,
            Client.KLINE_INTERVAL_1HOUR: 60,
            Client.KLINE_INTERVAL_2HOUR: 120,
            Client.KLINE_INTERVAL_4HOUR: 240,
            Client.KLINE_INTERVAL_6HOUR: 360,
            Client.KLINE_INTERVAL_8HOUR: 480,
            Client.KLINE_INTERVAL_12HOUR: 720,
            Client.KLINE_INTERVAL_1DAY: 1440,
            Client.KLINE_INTERVAL_3DAY: 4320,
            Client.KLINE_INTERVAL_1WEEK: 10080,
            Client.KLINE_INTERVAL_1MONTH: 40320}


def calculate_next_timestamp(last_stamp, interval):
    """ Calculate the next time stamp.

    :param last_stamp: last time stamp
    :param interval: Binance interval
    :return: timestamp
    """
    return last_stamp + kline_intervals()[interval] * 60000


def get_all_candles(client, symbol, interval, start_date):
    """ Get all candles from the given start date until now.

    :param client: Binance client
    :param symbol: Currency symbol
    :param interval: Candle interval
    :param start_date: Unix timestamp * 1000
    :return: Dataframe
    """
    # get first 1000 candles
    df = get_klines_df(client=client, symbol=symbol, interval=interval,
                       start_date=start_date,
                       limit=1000)

    # get all candles until now
    while last_close_time_before_now(df):
        df = df.append(get_klines_df(client=client,
                                     symbol=symbol,
                                     interval=interval,
                                     start_date=calculate_next_timestamp(df.iloc[-1]['close_time'], interval),
                                     limit=1000))

    return df


def main():
    # get api information
    api_file_path = 'api_key'
    api_key, api_secret = get_api_information(api_file_path)

    # initialize client
    client = Client(api_key, api_secret)

    # define symbol
    symbol = 'BTCUSDT'

    # iterate over all intervals
    for interval in kline_intervals().keys():
        start_time = timer()

        # get data
        df = get_all_candles(client, symbol, interval,
                             client._get_earliest_valid_timestamp(symbol, interval))

        # store data
        df.to_csv(f'../output/dataframes/{symbol}_{interval}.csv')
        print(f'Done with interval {interval} after {timer() - start_time}s.')


if __name__ == '__main__':
    main()
