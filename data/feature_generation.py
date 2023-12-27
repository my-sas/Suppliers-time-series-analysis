"""
В модуле описаны функции для генерации данных о поставщике для спецификаций.

Перечень колонок, содержащихся в выходной таблице:

Служебные:
 - 'supplier' — наименование поставщика
 - 'id' — идентификатор спецификации
 - 'spec_date' — дата спецификации
 - 'delivery_period_end' — дата конца поставок по спецификации

Целевые переменные:
 - 'bids_contracted' — законтрактованность (главная целевая переменная)
 - 'is_late' — поставщик опоздал
 - 'is_underweight' — поставщик доставил меньше обещенного
 - 'is_poorquality' — качество оказалось ниже заявленного

Параметры спецификации
 - 'supplier_status' — тип поставщика (Трейдер/СХТП/другое)
 - 'option' — ...
 - 'delivery_length' — продолжительность спецификации
 - 'volume_requested' — запрашиваемый объём товара

 Параметры предыдущих спецификаций поставщика
 - 'mean_delivery_length' — средняя продолжительность
 - 'mean_volume' — средний объём
 - 'delivery_length_diff' — разность 'delivery_length' и 'mean_delivery_length'
 - 'volume_diff' — разность 'volume_contracted' и 'mean_volume'
 - 'conversion' — конверсия (доля заключённых спецификаций)

Параметры предыдущих поставок поставщика:
 - 'supplier_lateness' — среднее опоздание
 - 'supplier_underweight' — средний недовес
 - 'supplier_price_change' — среднее изменение

Эмбеддинг поставщика:
 - колонки от '0' до '16'
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('..'))
from data.preprocessing import Preprocessor


RAW_SPEC_PATH = '../data/raw_data/specs.csv'
RAW_ZPP4_PATH = '../data/raw_data/zpp4.csv'

PROCESSED_SPEC_PATH = '../data/processed_data/specs.csv'
PROCESSED_ZPP4_PATH = '../data/processed_data/zpp4.csv'

FINAL_DATA_PATH = '../data/final_data/spec.csv'

EMBED_PATH = '../data/processed_data/embed_df.csv'


def spec_preporarion(spec, zpp4):
    """Кодирует и удаляет некоторые переменные таблицы ЦК,
    также генерируются дополнительные целевые переменные
    """

    # кодирование
    def status_func(x):
        if x == 'СХТП':
            return 1
        if x == 'Трейдер':
            return 0
        return 0.5
    spec['supplier_status'] = spec['supplier_status'].map(status_func)

    # продолжительность спецификации
    spec['delivery_length'] = (spec['delivery_period_end'] - spec['spec_date']).map(lambda x: x.days)

    # неинформативные колонки
    trash_cols = ['item', 'basis', 'payment_terms', 'logistics']
    spec = spec.drop(trash_cols, axis=1)

    # Дополнительные целевые переменные

    # опоздал или нет
    spec['is_late'] = spec.apply(lambda row: int(deliveries['date'].max() > row['delivery_period_end']) if len(
        (deliveries := zpp4[zpp4['id'] == row['id']])) > 0 else np.nan, axis=1)

    # был ли недовес
    spec['is_underweight'] = spec.apply(
        lambda row: int((deliveries['quantity'].sum() / row['volume_contracted']) < 1) if len(
            (deliveries := zpp4[zpp4['id'] == row['id']])) > 0 else np.nan, axis=1)

    # окозалось ли качество хуже более чем на 5%
    spec['is_poorquality'] = spec.apply(lambda row: int(((deliveries['quantity'] / row['volume_contracted']) * deliveries[
        'price_change']).sum() < -5) if len((deliveries := zpp4[zpp4['id'] == row['id']])) > 0 else np.nan, axis=1)
    return spec


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


def zpp4_preporarion2(zpp4, spec):
    """Генерирует НЕ НАКОПИТЕЛЬНЫЕ переменные для поставок
    и удаляет некоторые артефакты"""

    # для вычисления переменных необходимы данные о спецификации,
    # такие как дата окончания спецификации и объём поставок
    zpp4 = pd.merge(zpp4, spec[['id', 'delivery_period_end', 'volume_contracted']], how='inner', on=['id', 'id'],
                    suffixes=('', '_DROP'))

    def days_diff(series):
        dates = series.unique().astype('datetime64')
        dates.sort()

        days_diff = (dates[1:] - dates[:-1]).astype('timedelta64[D]').astype(int)
        days_diff = np.insert(days_diff, 0, -1)

        dates = dates.astype('datetime64[us]')
        dates_dict = {k: v for (k, v) in zip(dates, days_diff)}

        return series.map(lambda x: dates_dict[np.datetime64(x)])

    # продолжительность поставки
    zpp4['delivery_length'] = (zpp4['delivery_period_end'] - zpp4['spec_date']).map(lambda x: x.days)

    # 0 заменяем на 1
    zpp4['delivery_length'] = zpp4['delivery_length'].map(lambda x: (1 if x == 0 else x))

    # сколько дней прошло с прошлых поставок
    zpp4['days_diff'] = zpp4.groupby('id')['date'].transform(lambda x: days_diff(x))

    # если первая поставка, считаем сколько дней прошло с начала спецификации
    zpp4['days_diff'] = zpp4.apply(
        lambda row: (row['days_diff'] if row['days_diff'] != -1 else (row['date'] - row['spec_date']).days), axis=1)

    # переводим в проценты
    zpp4['time_persentage'] = zpp4['days_diff'] / zpp4['delivery_length']

    # процент доставленного товара
    zpp4['weight_percentage'] = zpp4['quantity'] / zpp4['volume_contracted']

    # удалить записи с нулевым volume_contracted
    zpp4 = zpp4.drop(zpp4.loc[zpp4['volume_contracted'] == 0].index)

    # дата последней поставки
    zpp4['last_date'] = zpp4.groupby(['id'])['date'].transform('max')
    return zpp4


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


def spec_features(spec, zpp4):
    """Функция создаёт новые переменные для записей в spec (ЦК)
    путём агрегации данных из прошлых спецификаций поставщика.
    """

    # средняя продолжительность предыдущих законтрактованных спецификаций поставщика
    spec['mean_delivery_length'] = past_agg(spec, 'delivery_length', 'supplier', 'spec_date',
                                            (spec['bids_contracted'] == 1))

    # разница продолжительности спецификации и средней продолжительности
    # предыдущих законтрактованных спецификаций поставщика
    spec['delivery_length_diff'] = (spec['delivery_length'] - spec['mean_delivery_length']).abs()

    # среднее количество товара поставщика по предыдущим законтрактованным спецификациям
    spec['mean_volume'] = past_agg(spec, 'volume_requested', 'supplier', 'spec_date',
                                   (spec['bids_contracted'] == 1))

    # разница количетва запрашиваемого в спецификации товара и среднего количества
    # товара по предыдущим законтрактованным спецификациям поставщика
    spec['volume_diff'] = (spec['volume_requested'] - spec['mean_volume']).abs()

    # конверсия поставщика на момент спецификации
    spec['conversion'] = past_agg(spec, 'bids_contracted', 'supplier', 'spec_date')
    return spec


def zpp4_features(spec, zpp4):
    """Функция создаёт в spec (ЦК) следующие переменные:
    1) Как часто в среднем опаздывает поставщик;
    2) Как часто у поставщика недовес;
    3) На сколько в среднем качество хуже заявленного.
    """

    # агрегируем данные по посылкам
    deliveries_agg = zpp4.groupby('id').agg({
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
    spec[['supplier_lateness', 'supplier_underweight', 'supplier_price_change']] = spec.apply(lambda row: deliveries_agg.loc[
        (deliveries_agg['supplier'] == row['supplier']) & (deliveries_agg['delivery_period_end'] < row['spec_date'])] \
        [['lateness', 'underweight', 'price_change']].mean(), axis=1)
    return spec


def embed_features(df):
    """Функция агрегирует эмбеддинги поставок полученные через LSTM энкодер
    """

    # здесь используется заранее сделанная в train_LSTM.ipynb таблица с эмбеддингами
    embed_df = pd.read_csv(EMBED_PATH)
    embed_df['date'] = pd.to_datetime(embed_df['date'], format='%Y-%m-%d')

    df[embed_df.select_dtypes(include=[np.number]).columns] = df.apply(
        lambda row: embed_df.loc[(embed_df['supplier'] == row['supplier']) & (embed_df['date'] < row['spec_date'])][
            embed_df.select_dtypes(include=[np.number]).columns].mean(), axis=1)
    return df


def pipeline():
    # если предобработанных таблиц нет, то создаём
    if (os.path.isfile(PROCESSED_SPEC_PATH) and os.path.isfile(PROCESSED_ZPP4_PATH)) is None:
        preprocessor = Preprocessor()
        zpp4 = preprocessor.zpp4_preprocessing()
        spec = preprocessor.spec_preprocessing()
    # если находим готовый файл с предобработанными таблицами, то загружаем
    else:
        spec = pd.read_csv('../data/processed_data/specs.csv')
        spec['spec_date'] = pd.to_datetime(spec['spec_date'], format='%Y-%m-%d')
        spec['delivery_period_end'] = pd.to_datetime(spec['delivery_period_end'], format='%Y-%m-%d')

        zpp4 = pd.read_csv('../data/processed_data/zpp4.csv')
        zpp4['date'] = pd.to_datetime(zpp4['date'], format='%Y-%m-%d')
        zpp4['spec_date'] = pd.to_datetime(zpp4['spec_date'], format='%Y-%m-%d')

    # Генерация фичей

    spec = spec_preporarion(spec, zpp4)
    zpp4 = zpp4_preporarion(zpp4, spec)

    # фичи аггрегации параметров прошлых поставок
    spec = zpp4_features(spec, zpp4)

    # фичи эмбеддинги
    spec = embed_features(spec)

    # агрегация из прошлых спецификаций
    spec = spec_features(spec, zpp4)

    return spec[
        ['supplier', 'id', 'spec_date', 'delivery_period_end'] +  # служебные
        ['bids_contracted', 'is_late', 'is_underweight', 'is_poorquality'] +  # целевые переменные
        ['supplier_status', 'option', 'delivery_length', 'volume_requested'] +  # параметры спецификации
        ['supplier_lateness', 'supplier_underweight', 'supplier_price_change'] +  # агрегации прошлых поставок
        ['mean_delivery_length', 'delivery_length_diff',  # агрегации параметров прошлых спецификаций
         'mean_volume', 'volume_diff', 'conversion'] +
        [str(i) for i in range(16)]  # эмбеддинг поставщика
    ]


if __name__ == "__main__":
    spec = pipeline()

    # сохраняем предобработанные файлы
    spec.to_csv(FINAL_DATA_PATH, index=False)
