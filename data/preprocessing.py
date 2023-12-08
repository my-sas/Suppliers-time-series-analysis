import pandas as pd
from tqdm import tqdm
import datetime


def past_agg(df, col, id_col, time_col, select_rows=True, agg='mean'):
    """Возвращает агрегацию для объектов из id_col,
    из записей, которые были раньше по времени

    Args:
        df (pd.DataFrame): Датафрейм.
        col (str or list of str): Колонки для которых нужно сделать агрегацию.
        id_col (str): Колонка идентификатор, по которому нужно группировать записи.
        time_col (str): Колонка с датой.
        select_rows (pd.Series or bool or list of bool): Строки из которых нужно выбирать.
        agg (callable): Аггригирующая функция.

    Returns:
        pd.DataFrame or pd.Series: Сагрегированные данные
    """

    if agg == 'mean':
        func = lambda row: df.loc[
            (df[id_col] == row[id_col]) &
            (df[time_col] < row[time_col]) &
            select_rows
            ][col].mean()

    elif agg == 'count':
        func = lambda row: len(df.loc[
                                   (df[id_col] == row[id_col]) &
                                   (df[time_col] < row[time_col]) &
                                   select_rows
                                   ][col])

    return df.apply(func, axis=1)


class Preprocessor:
    """Клас обрабатывает данные о спецификациях и поставках, а именно:
    - удаляет пропущенные значения;
    - заменяет неверно заполненные значения;
    - удаляет выбросы;
    - исправляет опечатки;
    - и т.д."""
    def __init__(self, DATA_DIR=None, SPEC_PATH=None, ZPP4_PATH=None, EXTRA_DATA_DIR=None):
        self.DATA_DIR = DATA_DIR if DATA_DIR is not None \
            else '../data/raw_data/'
        self.EXTRA_DATA_DIR = EXTRA_DATA_DIR if EXTRA_DATA_DIR is not None \
            else '../data/extra_data/'
        self.SPEC_PATH = self.DATA_DIR + SPEC_PATH if SPEC_PATH is not None \
            else self.DATA_DIR + 'ЦК.xlsx'
        self.ZPP4_PATH = self.DATA_DIR + ZPP4_PATH if ZPP4_PATH is not None \
            else self.DATA_DIR + 'ZPP4.xlsx'

    def load_spec_data(self):
        """Загрузка данных о спецификациях"""
        df = pd.read_excel(self.SPEC_PATH, skiprows=6, skipfooter=1, header=None)
        return df

    def load_zpp4_data(self):
        """Загрузка данных о поставках"""
        df = pd.read_excel(self.ZPP4_PATH, skiprows=3)
        return df

    def func1(self, df):
        df = df.drop(df.isna().all()[df.isna().all() == True].index.tolist(), axis=1)
        df.columns = ['bid_date', 'item', 'supplier', 'supplier_status', 'spec_date', 'delivery_period_end',
                      'payment_terms', 'option', 'comment',
                      'logistics', 'declared_price', 'consent_price',
                      'spec_price', 'basis',
                      'volume_requested', 'volume_contracted', 'bids_submitted', 'bids_contracted']

        df = df[~df['supplier'].isna()]
        df = df[~df['spec_date'].isna()]
        df = df[~df['delivery_period_end'].isna()]
        df = df[~df['volume_requested'].isna()]

        df['option'] = df['option'].fillna(0)

        df = df[df['logistics'] != 100]
        df = df[df['consent_price'] != 1].reset_index(drop=True)
        df = df[df['consent_price'] != 3].reset_index(drop=True)
        df = df[df['consent_price'] != 999]

        df['consent_price'] = df['consent_price'].apply(lambda x: x / 1000 if x > 1000 else x)
        df['consent_price'] = df['consent_price'].apply(lambda x: x / 10 if x >= 100 else x)
        df['declared_price'] = df['declared_price'].apply(lambda x: x / 1000 if x > 1000 else x)
        df['declared_price'] = df['declared_price'].apply(lambda x: x / 10 if x >= 100 else x)
        return df

    def func2(self, df):
        temp = df[(df['item'] == 'подсолнечник') & (df['consent_price'] > 33)].index
        df = df.drop(temp, axis=0).reset_index(drop=True)
        del temp

        try:
            df.loc[9822, 'consent_price'] = df.iloc[9822]['spec_price']
            df.loc[9822, 'declared_price'] = df.iloc[9822]['spec_price']
        except:
            pass

        temp = df.loc[~df['logistics'].isna()]
        log_min = temp['logistics'].describe()['min']
        log_max = temp['logistics'].describe()['max']
        temp_ = temp.loc[temp['declared_price'].between(log_min, log_max), ['consent_price', 'logistics']]
        temp_['declared_price'] = temp_['consent_price'] + temp_['logistics']
        df.loc[temp_.index] = temp_
        del temp, temp_
        return df

    def func3(self, df):

        df['bid_date'] = pd.to_datetime(df['bid_date'], format='%d.%m.%Y')
        df['spec_date'] = pd.to_datetime(df['spec_date'], format='%d.%m.%Y', errors='coerce')
        df['delivery_period_end'] = pd.to_datetime(df['delivery_period_end'], format='%d.%m.%Y', errors='coerce')

        df = df.sort_values(['spec_date', 'delivery_period_end', 'supplier', 'basis']).reset_index(drop=True)
        df = df[df['spec_date'] > pd.to_datetime(datetime.date(2022, 6, 30))].reset_index(drop=True)

        df.replace({'payment_terms': {'по факту': 'По факту', 'По фактк': 'По факту'}}, inplace=True)
        df['comment'] = df['comment'].fillna('')
        df['logistics'].fillna(0, inplace=True)
        df['declared_price'].fillna(0, inplace=True)
        df['spec_price'].fillna(0, inplace=True)

        df = df[['supplier', 'supplier_status', 'item', 'basis',
                 'spec_date', 'delivery_period_end',
                 'payment_terms', 'option',
                 'logistics', 'declared_price', 'consent_price', 'spec_price',
                 'volume_requested', 'volume_contracted',
                 'bids_submitted', 'bids_contracted']].reset_index(drop=True)
        return df

    def func4(self, df):
        cols = ['option', 'logistics', 'declared_price', 'consent_price', 'spec_price',
                'volume_requested', 'volume_contracted', 'bids_submitted', 'bids_contracted']

        duplicated_rows = df[df.duplicated(subset=['supplier', 'item', 'basis', 'spec_date'], keep=False)]
        duplicated_rows = duplicated_rows.sort_values(['spec_date', 'supplier', 'item', 'basis'])
        unique_rows = duplicated_rows.groupby(['supplier', 'item', 'basis', 'spec_date']) \
            [['bids_submitted', 'bids_contracted']].sum().reset_index()

        res = pd.DataFrame()
        for i, r in tqdm(unique_rows.iterrows(), desc='Duplicate rows cleaning'):
            sup = r['supplier']
            item = r['item']
            basis = r['basis']
            date = r['spec_date']

            if r['bids_contracted'] == 1:
                res_row = duplicated_rows[(duplicated_rows['supplier'] == sup)
                                          & (duplicated_rows['item'] == item)
                                          & (duplicated_rows['basis'] == basis)
                                          & (duplicated_rows['spec_date'] == date)
                                          & (duplicated_rows['bids_contracted'] == 1)]
                res = res.append(res_row, ignore_index=True)

            elif r['bids_contracted'] == 0:
                res_rows = duplicated_rows[(duplicated_rows['supplier'] == sup)
                                           & (duplicated_rows['item'] == item)
                                           & (duplicated_rows['basis'] == basis)
                                           & (duplicated_rows['spec_date'] == date)]
                mean_rows = res_rows[cols].mean()
                res_rows = res_rows.drop_duplicates(['supplier', 'item', 'basis', 'spec_date'])
                res_rows[cols] = mean_rows
                res = res.append(res_rows, ignore_index=True)

            elif r['bids_contracted'] == r['bids_submitted']:
                res_rows = duplicated_rows[(duplicated_rows['supplier'] == sup)
                                           & (duplicated_rows['item'] == item)
                                           & (duplicated_rows['basis'] == basis)
                                           & (duplicated_rows['spec_date'] == date)]
                mean_rows = res_rows[cols].mean()
                res_rows = res_rows.drop_duplicates(['supplier', 'item', 'basis', 'spec_date', 'spec_price'])

                if res_rows.shape[0] == 1:
                    res_rows[cols] = mean_rows
                    res = res.append(res_rows, ignore_index=True)
                else:
                    res = res.append(res_rows, ignore_index=True)

            elif r['bids_contracted'] < r['bids_submitted']:
                res_rows = duplicated_rows[(duplicated_rows['supplier'] == sup)
                                           & (duplicated_rows['item'] == item)
                                           & (duplicated_rows['basis'] == basis)
                                           & (duplicated_rows['spec_date'] == date)]
                res_rows = res_rows[res_rows['bids_contracted'] != 0]
                mean_rows = res_rows[cols].mean()
                res_rows = res_rows.drop_duplicates(['supplier', 'item', 'basis', 'spec_date'])
                res_rows[cols] = mean_rows
                res = res.append(res_rows, ignore_index=True)

            else:
                continue

        # этот файл потом просматривается вручную, чтобы удалить дубликаты
        # # res[res.duplicated(subset=['supplier', 'item', 'basis', 'spec_date'], keep=False)].to_csv('temp.csv')
        res = res[~res.duplicated(subset=['supplier', 'item', 'basis', 'spec_date'], keep=False)]
        temp = pd.read_csv(self.EXTRA_DATA_DIR + 'duplicated_rows.csv', index_col=0)
        temp['spec_date'] = pd.to_datetime(temp['spec_date'], format='%Y-%m-%d', errors='coerce')
        temp['delivery_period_end'] = pd.to_datetime(temp['delivery_period_end'], format='%Y-%m-%d', errors='coerce')

        res = pd.concat([res, temp]).reset_index(drop=True)

        df = df[~df.duplicated(subset=['supplier', 'item', 'basis', 'spec_date'], keep=False)]
        df = pd.concat([df, res]).reset_index(drop=True)

        return df

    def func5(self, df):

        # на один базис может быть только одна заявка
        # заявки пшеница как одна номенклатура
        temp = df.loc[df.duplicated(subset=['supplier', 'spec_date', 'basis'], keep=False), :].copy()
        temp['item'] = temp['item'].apply(lambda x: 'пшеница' if 'пшеница' in x else x)

        # отдельно законтрактованные и не законтрактованные заявки
        no_contract = temp[temp['bids_contracted'] == 0]
        contract = temp[temp['bids_contracted'] == 1]

        # Удаление дублиактов из не законтрактованных
        no_contract_wo_dupl = no_contract.drop_duplicates(['supplier', 'item', 'basis', 'spec_date']) \
            .sort_values(['supplier', 'item', 'basis', 'spec_date']).reset_index(drop=True)
        no_contract = no_contract.groupby(['supplier', 'spec_date', 'basis', 'item']).mean().reset_index() \
            .sort_values(['supplier', 'item', 'basis', 'spec_date']).reset_index(drop=True)
        no_contract_wo_dupl[['logistics', 'declared_price',
                             'consent_price', 'volume_requested']] = no_contract[['logistics', 'declared_price',
                                                                                  'consent_price', 'volume_requested']]

        # Удаление дублиактов из законтрактованных
        contract = contract[contract['consent_price'] == contract['spec_price']]
        contract = contract.drop_duplicates(contract.columns.tolist()) \
            .sort_values(['supplier', 'item', 'basis', 'spec_date']).reset_index(drop=True)
        contract_wo_dupl = contract.drop_duplicates(['supplier', 'item', 'basis', 'spec_date']) \
            .sort_values(['supplier', 'item', 'basis', 'spec_date']).reset_index(drop=True)
        contract = contract.groupby(['supplier', 'spec_date', 'basis', 'item']).mean().reset_index() \
            .sort_values(['supplier', 'item', 'basis', 'spec_date']).reset_index(drop=True)
        contract_wo_dupl[['logistics', 'declared_price', 'consent_price',
                          'spec_price', 'volume_requested', 'volume_contracted']] = \
            contract[['logistics', 'declared_price', 'consent_price',
                      'spec_price', 'volume_requested', 'volume_contracted']]

        # удаление всех заявок из временного датафрейма temp
        df = df.drop(temp.index, axis=0)

        # возвращение заявок из временного датафрейма без дубликатов
        temp = pd.concat([no_contract_wo_dupl, contract_wo_dupl])
        df = pd.concat([df, temp])

        return df

    def func6(self, df):
        df = df.sort_values(['supplier', 'spec_date', 'item', 'basis']).reset_index(drop=True)
        df = df[~df['delivery_period_end'].isna()].reset_index(drop=True)

        # только поставщики с более чем одной законтрактованной спецификацией
        temp = df.groupby('supplier')['bids_contracted'].sum().reset_index()
        df = df[df['supplier'].isin(temp[temp['bids_contracted'] > 1]['supplier'])].reset_index(drop=True)

        # уникальный идентификатор спецификации для мерджа с поставками, отслеживается по поставщику, дате и базису
        df['id'] = df['supplier'].astype(str) + '_' + df['basis'].astype(str) + '_' + df['spec_date'].astype(str)

        df = df.fillna(0)
        return df

    def feature_generation(self, df):
        # неинформативные колонки
        trash_cols = ['item', 'basis', 'payment_terms', 'logistics']

        # продолжительность спецификации
        df['delivery_length'] = (df['delivery_period_end'] - df['spec_date']).map(lambda x: x.days)

        # средняя продолжительность предыдущих законтрактованных спецификаций поставщика
        df['mean_delivery_length'] = past_agg(df, 'delivery_length', 'supplier', 'spec_date',
                                                (df['bids_contracted'] == 1))

        # разница продолжительности спецификации и средней продолжительности предыдущих законтрактованных спецификаций поставщика
        df['delivery_length_diff'] = (df['delivery_length'] - df['mean_delivery_length']).abs()

        # среднее количество товара поставщика по предыдущим законтрактованным спецификациям
        df['mean_volume'] = past_agg(df, 'volume_requested', 'supplier', 'spec_date',
                                       (df['bids_contracted'] == 1))

        # разница количетва запрашиваемого в спецификации товара и среднего количества
        # товара по предыдущим законтрактованным спецификациям поставщика
        df['volume_diff'] = (df['volume_requested'] - df['mean_volume']).abs()

        # конверсия поставщика на момент спецификации
        df['conversion'] = past_agg(df, 'bids_contracted', 'supplier', 'spec_date')
        return df.drop(trash_cols, 1)

    def spec_preprocessing(self):
        """Подготовка данных о спецификациях"""
        df = self.load_spec_data()
        df = self.func1(df)
        df = self.func2(df)
        df = self.func3(df)
        df = self.func4(df)
        df = self.func5(df)
        df = self.func6(df)
        return df

    def func7(self, df):
        df.drop(df.isna().sum()[df.isna().sum() == df.shape[0]].index.tolist(), axis=1, inplace=True)
        df['spec_date'] = pd.to_datetime(df['Спецификация'].fillna('') \
                                         .apply(lambda x: x.split('от ')[-1]), format='%d.%m.%Y', errors='coerce')
        df['Дата'] = pd.to_datetime(df['Дата'].apply(lambda x: x[:10]), format='%d.%m.%Y')

        try:
            dogovor = df[df['Контрагент'].isna()]['НомерДоговора'].unique().tolist()
            temp = df[df['НомерДоговора'].isin(dogovor)].groupby('НомерДоговора')['Контрагент'].agg(
                pd.Series.mode).reset_index()
            d = {x: y for x, y in zip(temp['НомерДоговора'].tolist(), temp['Контрагент'].tolist()) if type(y) == str}
            df['Контрагент'] = df['Контрагент'].fillna(df['НомерДоговора'].map(d))
            del dogovor, temp, d
        except:
            df = df[~df['Контрагент'].isna()]


        df = df[~df['Спецификация'].isna()]
        df = df[~df['Контрагент'].isna()]

        try:
            df = df.drop(['Договор', 'НомерДоговора', 'СреднееКачество', 'Ссылка', 'НомерСтроки',
                        'ТТН', 'КачествоРасчет', 'Водитель', 'НомерНакладной', 'НомерАвтомашины',
                        'Брутто', 'Тара', 'Спецификация'], axis=1)
        except:
             df = df.drop(['Договор', 
                             'Ссылка', 'НомерСтроки',
                          'КачествоРасчет', 'Водитель', 'НомерНакладной', 'НомерАвтомашины',
                          'Спецификация'], axis=1)
             

        df = df.rename(columns={
            'Количество': 'quantity',
            'ЦенаПоДоговору': 'contract_price', 
            'ЦенаРасчетная': 'estimated_price',
            'Сумма': 'sum',
            'ИтогИзмененияЦены': 'price_change',
            'Контрагент': 'supplier',
            'Элеватор': 'basis',
            'Дата': 'date',
            'ЦенаПоДоговору1': 'contract_price1',
            'ЦенаРасчетная1': 'estimated_price1'
        })

        # df.columns = ['date', 'supplier', 'basis', 'contract_price', 'estimated_price', 'quantity',
                    #   'contract_price1', 'estimated_price1', 'sum', 'price_change', 'spec_date']
        df = df[['date', 'supplier', 'basis', 'spec_date', 'contract_price', 'estimated_price',
                 'contract_price1', 'estimated_price1', 'quantity', 'sum', 'price_change']]

        return df

    def func8(self, df):
        df['contract_price'] = df['contract_price'].apply(lambda x: x / 1000 if x > 1000 else x)
        df['contract_price1'] = df['contract_price1'].apply(lambda x: x / 1000 if x > 1000 else x)
        df['contract_price'] = df['contract_price'].fillna(df['contract_price1'])
        df['contract_price1'] = df['contract_price1'].fillna(df['contract_price'])
        df = df[~df['contract_price'].isna()]
        df = df[~df['contract_price1'].isna()]
        df = df.reset_index(drop=True)

        df['estimated_price'] = df['estimated_price'].apply(lambda x: x / 1000 if x > 1000 else x)
        df['estimated_price1'] = df['estimated_price1'].apply(lambda x: x / 1000 if x > 1000 else x)
        df['estimated_price1'] = df['estimated_price1'].fillna(df['estimated_price'])
        df['estimated_price'] = df['estimated_price'].fillna(df['estimated_price1'])
        df = df[~df['estimated_price'].isna()]
        df = df[~df['estimated_price1'].isna()]
        df = df.reset_index(drop=True)

        df['quantity'] = df['quantity'].apply(lambda x: x / 1000 if x > 1000 else x)

        df['sum_'] = (df['estimated_price1'] * df['quantity'] * 1000).apply(lambda x: round(x, 1))
        df['sum_diff'] = df['sum'] - df['sum_']
        df['sum'] = df['sum'].fillna(df['sum_'])
        df = df.drop(['sum_', 'sum_diff'], axis=1)
        df['price_change'] = round((df['estimated_price1'] - df['contract_price1']) / df['contract_price1'] * 100, 2)
        df['id'] = df['supplier'].astype(str) + '_' + df['basis'].astype(str) + '_' + df['spec_date'].astype(str)

        return df

    def zpp4_preprocessing(self):
        """Подготовка данных о поставках"""
        df = self.load_zpp4_data()
        df = self.func7(df)
        df = self.func8(df)
        return df
