import numpy as np


def spec_preporarion(df):

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
    return df.drop(trash_cols, 1)



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


def spec_agg_features(df):
    """Функция создаёт новые переменные для записей в spec (ЦК)
    путём агрегации данных из прошлых спецификаций поставщика,
    а также удаляет неинформативные колонки
    """

    df = spec_preporarion(df)

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
