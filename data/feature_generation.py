"""
В модуле описаны функции для генерации данных о поставщике для спецификаций.
Предполагается, что датафреймы прошли обработку в preprocessing.
"""

import numpy as np
import pandas as pd


def spec_preporarion(df, zpp4):
    """Кодирует и удаляет некоторые переменные таблицы ЦК
    """

    # кодирование
    def status_func(x):
        if x == 'СХТП':
            return 1
        if x == 'Трейдер':
            return 0
        return 0.5
    df['supplier_status'] = df['supplier_status'].map(status_func)

    # продолжительность спецификации
    df['delivery_length'] = (df['delivery_period_end'] - df['spec_date']).map(lambda x: x.days)

    # неинформативные колонки
    trash_cols = ['item', 'basis', 'payment_terms', 'logistics']
    df = df.drop(trash_cols, axis=1)

    # Дополнительные целевые переменные

    # опоздал или нет
    df['is_late'] = df.apply(lambda row: int(deliveries['date'].max() > row['delivery_period_end']) if len(
        (deliveries := zpp4[zpp4['id'] == row['id']])) > 0 else np.nan, axis=1)

    # был ли недовес
    df['is_underweight'] = df.apply(
        lambda row: int((deliveries['quantity'].sum() / row['volume_contracted']) < 1) if len(
            (deliveries := zpp4[zpp4['id'] == row['id']])) > 0 else np.nan, axis=1)

    # окозалось ли качество хуже более чем на 5%
    df['is_poorquality'] = df.apply(lambda row: int(((deliveries['quantity'] / row['volume_contracted']) * deliveries[
        'price_change']).sum() < -5) if len((deliveries := zpp4[zpp4['id'] == row['id']])) > 0 else np.nan, axis=1)
    return df


def zpp4_preporarion(df, spec):
    """Генерирует НАКОПИТЕЛЬНЫЕ переменные для поставок
    и удаляет некоторые артефакты"""

    # для вычисления переменных необходимы данные о спецификации,
    # такие как дата окончания спецификации и объём поставок
    df = pd.merge(df, spec[['id', 'delivery_period_end', 'volume_contracted']], how='inner', on=['id', 'id'],
                  suffixes=('', '_DROP'))

    def lateness_percentage(row):
        if (row['delivery_period_end'] - row['spec_date']).days > 0:
            # сколько прошло дней с начала спецификации
            days_passed = max((row['date'] - row['spec_date']).days, 0)

            # сколько дней длится спецификация
            delivery_lenght = (row['delivery_period_end'] - row['spec_date']).days

        # Если спецификация началась и закончилась в один день,
        # то считаем длительность поставки равной одному дню,
        # а к количеству прошедших дней прибавляем 1.
        # Таким образом мы считаем, что спецификация начинается в начале указаного дня,
        # а доставка и конец спецификации в конце указанного дня.
        else:
            days_passed = max((row['date'] - row['spec_date']).days + 1, 0)
            delivery_lenght = 1

        return days_passed / delivery_lenght

    # На сколько поставщик близок к концу спецификации
    df['lateness_percentage'] = df.apply(lambda row: lateness_percentage(row), axis=1)

    # Доля уже доставленного товара от всего законтрактованого
    df['weight_percentage'] = df.groupby('id')['quantity'].cumsum() / df['volume_contracted']

    # процент количества на процент качества (сумма по поставке даст средний процент изменения качества)
    df['relative_price_change'] = (df['quantity'] / df['volume_contracted']) * df['price_change']

    # удалить записи с нулевым volume_contracted
    df = df.drop(df.loc[df['volume_contracted'] == 0].index)

    # дата последней поставки
    df['last_date'] = df.groupby(['id'])['date'].transform('max')
    return df


# def zpp4_preporarion_v2(df, spec):
#     """Генерирует переменные для поставок
#     и удаляет некоторые артефакты"""
#
#     # для вычисления переменных необходимы данные о спецификации,
#     # такие как дата окончания спецификации и объём поставок
#     df = pd.merge(df, spec[['id', 'delivery_period_end', 'volume_contracted']], how='inner', on=['id', 'id'],
#                   suffixes=('', '_DROP'))
#
#     def lateness_percentage(row):
#         if (row['delivery_period_end'] - row['spec_date']).days > 0:
#             # сколько прошло дней с начала спецификации
#             days_passed = max((row['date'] - row['spec_date']).days, 0)
#
#             # сколько дней длится спецификация
#             delivery_lenght = (row['delivery_period_end'] - row['spec_date']).days
#
#         # Если спецификация началась и закончилась в один день,
#         # то считаем длительность поставки равной одному дню,
#         # а к количеству прошедших дней прибавляем 1.
#         # Таким образом мы считаем, что спецификация начинается в начале указаного дня,
#         # а доставка и конец спецификации в конце указанного дня.
#         else:
#             days_passed = max((row['date'] - row['spec_date']).days + 1, 0)
#             delivery_lenght = 1
#
#         return days_passed / delivery_lenght
#
#     # На сколько поставщик близок к концу спецификации
#     df['lateness_percentage'] = df.apply(lambda row: lateness_percentage(row), axis=1)
#
#     # Доля уже доставленного товара от всего законтрактованого
#     df['weight_percentage'] = df['quantity'] / df['volume_contracted']
#
#     # удалить записи с нулевым volume_contracted
#     df = df.drop(df.loc[df['volume_contracted'] == 0].index)
#
#     # # дата последней поставки
#     # df['last_date'] = df.groupby(['id'])['date'].transform('max')
#     return df


def past_agg(df, select_cols, id_col, time_col, select_rows=True, agg_func=np.mean):
    """Возвращает агрегацию для объектов из id_col,
    из записей которые были раньше по времени

    Args:
        df (pd.DataFrame): Датафрейм.
        select_cols (str or list of str): Колонки для которых нужно сделать агрегацию.
        id_col (str): Колонка идентификатор, по которому нужно группировать записи.
        time_col (str): Колонка с датой.
        select_rows (pd.Series or bool or list of bool): Строки из которых нужно выбирать.
        agg_func (callable): Аггригирующая функция.

    Returns:
        pd.DataFrame or pd.Series: Сагрегированные данные
    """

    func = lambda row: agg_func(df.loc[
                                    (df[id_col] == row[id_col]) &
                                    (df[time_col] < row[time_col]) &
                                    select_rows
                                    ][select_cols])
    return df.apply(func, axis=1)


def spec_agg_features(df, zpp4):
    """Функция создаёт новые переменные для записей в spec (ЦК)
    путём агрегации данных из прошлых спецификаций поставщика.
    """

    df = spec_preporarion(df, zpp4)

    # средняя продолжительность предыдущих законтрактованных спецификаций поставщика
    df['mean_delivery_length'] = past_agg(df, 'delivery_length', 'supplier', 'spec_date',
                                          (df['bids_contracted'] == 1))

    # разница продолжительности спецификации и средней продолжительности
    # предыдущих законтрактованных спецификаций поставщика
    df['delivery_length_diff'] = (df['delivery_length'] - df['mean_delivery_length']).abs()

    # среднее количество товара поставщика по предыдущим законтрактованным спецификациям
    df['mean_volume'] = past_agg(df, 'volume_requested', 'supplier', 'spec_date',
                                 (df['bids_contracted'] == 1))

    # разница количетва запрашиваемого в спецификации товара и среднего количества
    # товара по предыдущим законтрактованным спецификациям поставщика
    df['volume_diff'] = (df['volume_requested'] - df['mean_volume']).abs()

    # конверсия поставщика на момент спецификации
    df['conversion'] = past_agg(df, 'bids_contracted', 'supplier', 'spec_date')
    return df


def zpp4_agg_features(df, deliveries):
    """Функция создаёт в spec (ЦК) следующие переменные:
    1) Как часто в среднем опаздывает поставщик;
    2) Как часто у поставщика недовес;
    3) На сколько в среднем качество хуже заявленного.
    """

    deliveries = zpp4_preporarion(deliveries, df)

    # агрегируем данные по посылкам
    deliveries_agg = deliveries.groupby('id').agg({
        'supplier': lambda x: x.iloc[0],
        'delivery_period_end': lambda x: x.iloc[0],
        'lateness_percentage': lambda x: x.max(),
        'weight_percentage': lambda x: 1 - x.max(),
        'relative_price_change': 'sum'
    }).rename(columns={
        'lateness_percentage': 'lateness',
        'weight_percentage': 'underweight',
        'relative_price_change': 'price_change'
    })

    # агрегируем данные по предыдущим поставкам поставщика
    df[['supplier_lateness', 'supplier_underweight', 'supplier_price_change']] = df.apply(lambda row: deliveries_agg.loc[
        (deliveries_agg['supplier'] == row['supplier']) & (deliveries_agg['delivery_period_end'] < row['spec_date'])] \
        [['lateness', 'underweight', 'price_change']].mean(), axis=1)
    return df


def zpp4_embed_agg(df):
    """Функция агрегирует эмбеддинги поставок полученные через LSTM энкодер
    """

    # здесь используется заранее сделанная в LSTM.ipynb таблица с эмбеддингами
    embed_df = pd.read_csv('../data/processed_data/embed_df.csv')
    embed_df['date'] = pd.to_datetime(embed_df['date'], format='%Y-%m-%d')

    df[embed_df.select_dtypes(include=[np.number]).columns] = df.apply(
        lambda row: embed_df.loc[(embed_df['supplier'] == row['supplier']) & (embed_df['date'] < row['spec_date'])][
            embed_df.select_dtypes(include=[np.number]).columns].mean(), axis=1)
    return df