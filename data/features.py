import json
import joblib
from tqdm import tqdm
import pandas as pd
import datetime
from sklearn.preprocessing import QuantileTransformer


class FeaturesGenerator:
    """"Класс генерирует признаки для спецификаций и поставок, а также агрегирует эти данные"""
    def __init__(self, SPECS_DF, ZPP4_DF, EXTRA_DATA_DIR=None, FEATURES_PATH=None, TRANSFORMERS_DIR=None, PROCESSED_DATA_DIR=None, NEW_DATA=False):
        self.SPECS_DF = SPECS_DF
        self.ZPP4_DF = ZPP4_DF
        self.CONTRACTED_SPECS = pd.DataFrame()
        self.EXTRA_DATA_DIR = EXTRA_DATA_DIR if EXTRA_DATA_DIR is not None \
            else '../data/extra_data/'
        self.FEATURES_PATH = self.EXTRA_DATA_DIR + FEATURES_PATH if FEATURES_PATH is not None \
            else self.EXTRA_DATA_DIR + 'features_to_use.json'
        self.TRANSFORMERS_DIR = TRANSFORMERS_DIR if TRANSFORMERS_DIR is not None \
            else '../data/processed_data/transformers/'
        self.PROCESSED_DATA_DIR = PROCESSED_DATA_DIR if PROCESSED_DATA_DIR is not None \
            else '../data/processed_data/'
        with open(self.FEATURES_PATH) as f:
            self.features_to_use = json.load(f)
        self.DATE_LIST = []
        self.NEW_DATA = NEW_DATA

    def preprocessing(self):
        """Подготовка данных из файлов спецификаций, поставок, а также создание файла с законченными спецификациями"""
        self.SPECS_DF['spec_date'] = pd.to_datetime(self.SPECS_DF['spec_date'], format='%Y-%m-%d', errors='coerce')
        self.SPECS_DF['delivery_period_end'] = pd.to_datetime(self.SPECS_DF['delivery_period_end'], format='%Y-%m-%d',
                                                              errors='coerce')

        self.ZPP4_DF['date'] = pd.to_datetime(self.ZPP4_DF['date'], format='%Y-%m-%d', errors='coerce')
        self.ZPP4_DF['spec_date'] = pd.to_datetime(self.ZPP4_DF['spec_date'], format='%Y-%m-%d', errors='coerce')

    
        if self.NEW_DATA:
            specs_old = pd.read_csv(self.PROCESSED_DATA_DIR + 'specs.csv', index_col=0)
            specs_old['spec_date'] = pd.to_datetime(specs_old['spec_date'], format='%Y-%m-%d', errors='coerce')
            specs_old['delivery_period_end'] = pd.to_datetime(specs_old['spec_date'], format='%Y-%m-%d', errors='coerce')

            zpp4_old = pd.read_csv(self.PROCESSED_DATA_DIR + 'zpp4.csv', index_col=0)
            zpp4_old['date'] = pd.to_datetime(zpp4_old['date'], format='%Y-%m-%d', errors='coerce')
            zpp4_old['spec_date'] = pd.to_datetime(zpp4_old['spec_date'], format='%Y-%m-%d', errors='coerce')

            self.SPECS_DF = self.SPECS_DF[self.SPECS_DF['spec_date']>specs_old['spec_date'].max()]
            self.ZPP4_DF = self.ZPP4_DF[self.ZPP4_DF['date']>zpp4_old['date'].max()]

            # даты для агрегации признаков
            d = self.SPECS_DF['spec_date'].min()
            while d <= self.SPECS_DF['spec_date'].max():
                self.DATE_LIST.append(d)
                d = d + datetime.timedelta(days=1)

            self.SPECS_DF = pd.concat([self.SPECS_DF, specs_old], ignore_index=True).sort_values('spec_date').reset_index(drop=True)
            self.ZPP4_DF = pd.concat([self.ZPP4_DF, zpp4_old], ignore_index=True).sort_values('spec_date').reset_index(drop=True)
            self.SPECS_DF.to_csv(self.PROCESSED_DATA_DIR + 'specs.csv', mode='a')
            self.ZPP4_DF.to_csv(self.PROCESSED_DATA_DIR + 'zpp4.csv', mode='a')
        
        else:
            # даты для агрегации признаков
            d = self.SPECS_DF['spec_date'].min()
            while d <= self.SPECS_DF['spec_date'].max():
                self.DATE_LIST.append(d)
                d = d + datetime.timedelta(days=1)

        self.CONTRACTED_SPECS = self.SPECS_DF[self.SPECS_DF['bids_contracted'] > 0].reset_index(drop=True)
        self.CONTRACTED_SPECS.loc[self.CONTRACTED_SPECS['declared_price'] == 0, 'declared_price'] = self.CONTRACTED_SPECS[
            'spec_price']

        # self.CONTRACTED_SPECS['declared_price'][self.CONTRACTED_SPECS['declared_price'] == 0] = self.CONTRACTED_SPECS[
        #     'spec_price']
        self.CONTRACTED_SPECS = self.CONTRACTED_SPECS[self.CONTRACTED_SPECS['id'].isin(self.ZPP4_DF['id'])]

    def features_generation(self):
        """Генерация признаков"""

        # сколько нужно поставлять в день
        self.SPECS_DF['delivery_period_days'] = \
            (self.SPECS_DF['delivery_period_end'] - self.SPECS_DF['spec_date']).dt.days + 1
        self.SPECS_DF['contracted_volume_per_day'] = \
            self.SPECS_DF['volume_contracted'] / self.SPECS_DF['delivery_period_days']

        # минимальная, максимальная дата поставки в спецификации
        temp1 = self.ZPP4_DF.groupby('id')['date'].min().reset_index().rename(columns={'date': 'date_min'})
        temp2 = self.ZPP4_DF.groupby('id')['date'].max().reset_index().rename(columns={'date': 'date_max'})
        temp = temp1.merge(temp2)
        self.CONTRACTED_SPECS = self.CONTRACTED_SPECS.merge(temp, on='id', how='left')
        del temp, temp1, temp2

        # нарушение сроков поставки - сколько дней прошло между последенй поставкой и концом периода договора
        self.CONTRACTED_SPECS['delivery_period_violation_days'] = \
            (self.CONTRACTED_SPECS['date_max'] - self.CONTRACTED_SPECS['delivery_period_end']).dt.days
        self.CONTRACTED_SPECS['delivery_on_time_bin'] = \
            self.CONTRACTED_SPECS['delivery_period_violation_days'].apply(lambda x: 1 if x <= 0 else 0)

        # поставленный объем за одну спецификацию
        # разница поставленного объема и заявленного
        # разница поставленного объема и заявленного в процентах
        temp = self.ZPP4_DF.groupby('id')['quantity'].sum().reset_index()
        self.CONTRACTED_SPECS = self.CONTRACTED_SPECS.merge(temp, on='id', how='left')

        self.CONTRACTED_SPECS['vol_diff'] = \
            round((self.CONTRACTED_SPECS['volume_contracted'] - self.CONTRACTED_SPECS['quantity']) * -1, 2).fillna(0)
        self.CONTRACTED_SPECS['vol_diff_%'] = \
            round(self.CONTRACTED_SPECS['vol_diff'] / self.CONTRACTED_SPECS['volume_contracted'] * 100, 2).fillna(0)
        del temp

        # объем поставок по спецификаии выше/ниже границ опциона
        self.CONTRACTED_SPECS['option_bin'] = \
            (self.CONTRACTED_SPECS['vol_diff_%'] >= self.CONTRACTED_SPECS['option'] * -1).apply(int)

        # среднее, макисмальное и минимальное изменение цены (изменение цены в ПРОЦЕНТАХ) по всем поставкам в спецификации
        self.CONTRACTED_SPECS = self.CONTRACTED_SPECS.merge(self.ZPP4_DF.groupby('id')['price_change'] \
                                                            .agg(['mean', 'max', 'min']).reset_index() \
                                                            .rename(columns={'mean': 'spec_price_change_mean',
                                                                             'max': 'spec_price_change_max',
                                                                             'min': 'spec_price_change_min'}))

        # качество поставки, качество поставок спецификации
        # количество хороших/плохих поставок в спецификации
        # процент хороших поставок в спецфикации
        self.ZPP4_DF['price_change_bin'] = self.ZPP4_DF['price_change'].apply(lambda x: 0 if x < 0 else 1)
        self.CONTRACTED_SPECS['price_change_bin'] = \
            self.CONTRACTED_SPECS['spec_price_change_mean'].apply(lambda x: 0 if x < 0 else 1)

        temp = self.ZPP4_DF.groupby('id')['price_change_bin'].agg(['sum', 'count']).reset_index()
        temp.columns = ['id', 'good_quality_count', 'total_deliveries']
        temp['bad_quality_count'] = temp['total_deliveries'] - temp['good_quality_count']
        temp['quality_%'] = temp['good_quality_count'] / temp['total_deliveries']
        self.CONTRACTED_SPECS = self.CONTRACTED_SPECS.merge(temp.drop('total_deliveries', axis=1), on='id', how='left')

        # сколько в среднем поставляет в день по конкретной спецификации
        self.CONTRACTED_SPECS = \
            self.CONTRACTED_SPECS.merge(self.ZPP4_DF.groupby(['id', 'date'])['quantity'].sum() \
                                        .groupby(level=0).agg(['mean', 'max', 'min']).reset_index() \
                                        .rename(columns={'mean': 'mean_delivered_volume_per_day',
                                                         'max': 'max_delivered_volume_per_day',
                                                         'min': 'min_delivered_volume_per_day'}))

        self.CONTRACTED_SPECS = \
            self.CONTRACTED_SPECS.merge(self.ZPP4_DF.groupby(['id'])['quantity'] \
                                        .agg(['mean', 'max', 'min']).reset_index() \
                                        .rename(columns={'mean': 'mean_delivered_volume_per_delivery',
                                                         'max': 'max_delivered_volume_per_delivery',
                                                         'min': 'min_delivered_volume_per_delivery'}))

        # Разница заявленной цены и цены спецификации:
        # количество спецификаций, где цена спецификации меньше заявленной
        # количество спецификаций, где цена спецификации больше заявленной
        self.CONTRACTED_SPECS['sd_price_diff'] = \
            round((self.CONTRACTED_SPECS['spec_price'] - self.CONTRACTED_SPECS['declared_price'])
                  / self.CONTRACTED_SPECS['declared_price'], 2)
        self.CONTRACTED_SPECS['sd_price_diff_bin'] = \
            self.CONTRACTED_SPECS['sd_price_diff'].apply(lambda x: -1 if x > 0 else (0 if x == 0 else 1))
        

    def make_features_df(self):
        """Класс агрегирует данные и собирает все один датафрейм признаков"""

        # препроцессинг и создание признаков
        self.preprocessing()
        self.features_generation()

        # цикл по датам, сохранение в датафрейм results
        results = pd.DataFrame()
        for d in tqdm(self.DATE_LIST):
            res_df = pd.DataFrame()

            temp_df = self.SPECS_DF.loc[self.SPECS_DF['spec_date'] < d]
            temp_zpp4 = self.ZPP4_DF.loc[self.ZPP4_DF['date'] < d]
            temp_contracted_specs = self.CONTRACTED_SPECS.loc[self.CONTRACTED_SPECS['date_max'] < d]

            '''
                Сколько раз поставщик не был законтрактован 
                общее количество заявок
                количество законтрактованных заявок
                % законтрактованных от всех заявок поставщика
                общий законтрактованный объем
            '''
            temp_bids = temp_df.groupby('supplier')[['bids_submitted', 'bids_contracted', 'volume_contracted']].sum() \
                .reset_index().rename(columns={'volume_contracted': 'total_volume_contracted'})
            temp_bids['bids_nocontracted'] = temp_bids['bids_submitted'] - temp_bids['bids_contracted']
            temp_bids['conv'] = round(temp_bids['bids_contracted'] / temp_bids['bids_submitted'], 2)

            res_df = pd.concat([res_df, temp_bids])
            res_df['date'] = d
            del temp_bids

            '''
                Разница заявленной цены и цены спецификации:
                количество спецификаций, где цена спецификации меньше заявленной
                количество спецификаций, где цена спецификации больше заявленной
            '''
            temp_price = temp_contracted_specs.groupby(['supplier'])['sd_price_diff'].agg(['mean', 'max', 'min']) \
                .reset_index().rename(columns={'mean': 'sd_price_diff_mean',
                                               'max': 'sd_price_diff_max',
                                               'min': 'sd_price_diff_min'})
            temp_price_bin = self.CONTRACTED_SPECS.groupby(['supplier', 'sd_price_diff_bin'])['spec_date'].count() \
                .reset_index().pivot_table(
                index='supplier',
                columns='sd_price_diff_bin',
                values='spec_date',
                fill_value=0).reset_index().rename(columns={-1: 'price_higher_count',
                                                            0: 'price_not_changed_count',
                                                            1: 'price_lower_count'})

            res_df = res_df.merge(temp_price, how='left', on='supplier').fillna(0)
            res_df = res_df.merge(temp_price_bin, how='left', on='supplier').fillna(0)

            '''
                Как долго сотрудничает с ЮГ Руси
            '''
            temp_date = temp_df.groupby('supplier')['spec_date'].agg(['min', 'max']).reset_index() \
                .rename(columns={'min': 'first_spec_date',
                                 'max': 'last_spec_date'})
            temp_spec_date = \
                temp_contracted_specs[temp_contracted_specs['bids_contracted'] > 0].groupby('supplier')['spec_date'] \
                    .agg(['min', 'max']).reset_index().rename(columns={'min': 'first_contracted_date',
                                                                       'max': 'last_last_contracted_date'})
            temp_zpp4_date = temp_zpp4.groupby('supplier')['date'].agg(['min', 'max']).reset_index() \
                .rename(columns={'min': 'first_delivery_date',
                                 'max': 'last_delivery_date'})

            for c in temp_zpp4_date:
                if c != 'supplier':
                    temp_zpp4_date[c] = (d - temp_zpp4_date[c]).dt.days + 1

            for c in temp_date:
                if c != 'supplier':
                    temp_date[c] = (d - temp_date[c]).dt.days + 1

            for c in temp_spec_date:
                if c != 'supplier':
                    temp_spec_date[c] = (d - temp_spec_date[c]).dt.days + 1

            res_df = res_df.merge(temp_date, how='left', on='supplier').fillna(0)
            res_df = res_df.merge(temp_spec_date, how='left', on='supplier').fillna(0)
            res_df = res_df.merge(temp_zpp4_date, how='left', on='supplier').fillna(0)
            del temp_date, temp_spec_date

            '''
                общие объемы поставленного за весь срок
                количество поставок в день
            '''
            temp_quanity = temp_zpp4.groupby('supplier')['quantity'].agg(['sum', 'mean', 'max', 'count']).reset_index() \
                .rename(columns={'sum': 'total_deliveried_quantity',
                                 'mean': 'mean_deliveried_quantity',
                                 'max': 'max_delivered_quantity',
                                 'count': 'total_deliveries_number'})

            delivery_days = temp_zpp4.groupby('supplier')['date'].nunique().reset_index().rename(
                columns={'date': 'nunique_delivery_days'})
            delivery_days['deliveries_freq'] = temp_quanity['total_deliveries_number'] / delivery_days[
                'nunique_delivery_days']

            res_df = res_df.merge(delivery_days, how='left', on='supplier').fillna(0)
            res_df = res_df.merge(temp_quanity, how='left', on='supplier').fillna(0)
            del temp_quanity, delivery_days

            '''
                Разница заявленного объема и поставленного
                отношение поставленного объема к заявленному
            '''
            delivered_volume = temp_zpp4[temp_zpp4['id'].isin(temp_contracted_specs['id'])].groupby('supplier')[
                'quantity'].sum().reset_index()
            contracted_volume_zpp4 = temp_df[temp_df['id'].isin(temp_zpp4['id'])].groupby('supplier')[
                'volume_contracted'].sum().reset_index() \
                .rename(columns={'volume_contracted': 'volume_contracted_zpp4'})
            contracted_volume_all = temp_df.groupby('supplier')['volume_contracted'].sum().reset_index() \
                .rename(columns={'volume_contracted': 'volume_contracted_all'})

            temp = delivered_volume.merge(
                contracted_volume_zpp4.merge(contracted_volume_all, how='left', on='supplier'), how='left',
                on='supplier')

            temp['volume_dс_diff_zpp4'] = temp['quantity'] - temp['volume_contracted_zpp4']
            temp['volume_dс_diff_all'] = temp['quantity'] - temp['volume_contracted_all']
            temp['volume_dc_ratio_zpp4'] = temp['quantity'] / temp['volume_contracted_zpp4']
            temp['volume_dc_ratio_all'] = temp['quantity'] / temp['volume_contracted_all']

            res_df = res_df.merge(temp, how='left', on='supplier').fillna(0)
            del temp, delivered_volume, contracted_volume_zpp4, contracted_volume_all

            '''
                изменение цены (качество)
                среднее, максимальное, минимальное, медианное изменение цены по поставкам
                количество поставок с качеством выше/ниже заявленного
                процент поставок с качеством выше
            '''
            temp_quality = temp_zpp4.groupby('supplier')['price_change'].agg(
                ['mean', 'max', 'min', 'median']).reset_index() \
                .rename(columns={'mean': 'mean_deliveried_quality',
                                 'max': 'max_deliveried_quality',
                                 'min': 'min_deliveried_quality',
                                 'median': 'median_deliveried_quality'})
            res_df = res_df.merge(temp_quality, how='left', on='supplier').fillna(0)
            del temp_quality

            price_change_bin = temp_zpp4.groupby('supplier')['price_change_bin'].agg(['sum', 'count']).reset_index() \
                .rename(columns={'sum': 'good_quality_deliveries_number',
                                 'count': 'deliveries_count'})
            price_change_bin['bad_quality_deliveries_number'] = \
                price_change_bin['deliveries_count'] - price_change_bin['good_quality_deliveries_number']
            price_change_bin['good_quality_deliveries_percent'] = \
                price_change_bin['good_quality_deliveries_number'] / price_change_bin['deliveries_count']

            res_df = res_df.merge(price_change_bin, how='left', on='supplier').fillna(0)
            del price_change_bin

            '''
                Поставка ниже границ опциона
                Не поставлено в срок (bin)
                Не соблюден объем поставок (bin)
                Качество ввозимового в разрезе приемлемого качества базиса (мб бинарное “выше/ниже приемлемых цифр”)
            '''
            temp_option = temp_contracted_specs.groupby('supplier')['option_bin'].agg(['sum', 'count']).reset_index() \
                .rename(columns={'sum': 'spec_good_option_number', 'count': 'specs_number'})
            temp_option['spec_bad_option_number'] = temp_option['specs_number'] - temp_option['spec_good_option_number']
            res_df = res_df.merge(temp_option, how='left', on='supplier').fillna(0)
            del temp_option

            temp_days = temp_contracted_specs.groupby('supplier')['delivery_on_time_bin'].agg(['sum', 'count']) \
                .reset_index().rename(columns={'sum': 'deliveries_on_time',
                                               'count': 'total_deliveries'})
            temp_days['deliveries_not_on_time'] = temp_days['total_deliveries'] - temp_days['deliveries_on_time']
            res_df = res_df.merge(temp_days, how='left', on='supplier').fillna(0)
            del temp_days

            price_change_bin = temp_contracted_specs.groupby('supplier')['price_change_bin'].agg(['sum', 'count']) \
                .reset_index().rename(columns={'sum': 'good_quality_specs_number',
                                               'count': 'specs_count'})
            price_change_bin['bad_quality_specs_number'] = \
                price_change_bin['specs_count'] - price_change_bin['good_quality_specs_number']
            res_df = res_df.merge(price_change_bin, how='left', on='supplier').fillna(0)
            del price_change_bin

            results = pd.concat([results, res_df])

        results = results.drop(['deliveries_count', 'total_deliveries', 'specs_count'], axis=1)
        results = results.merge(
            self.SPECS_DF.groupby('supplier')['supplier_status'].agg(pd.Series.mode).reset_index(),
            how='left',
            on='supplier')

        new_columns_for_df = [
            'date', 'supplier', 'supplier_status',
            'bids_submitted', 'bids_contracted', 'bids_nocontracted', 'conv',
            'total_deliveries_number', 'specs_number',
            'spec_good_option_number', 'spec_bad_option_number',
            'deliveries_on_time', 'deliveries_not_on_time',
            'good_quality_specs_number', 'bad_quality_specs_number',
            'price_higher_count', 'price_not_changed_count', 'price_lower_count',
            'good_quality_deliveries_number', 'bad_quality_deliveries_number',
            'good_quality_deliveries_percent',
            'nunique_delivery_days', 'deliveries_freq',
            'sd_price_diff_mean', 'sd_price_diff_max', 'sd_price_diff_min',
            'first_spec_date', 'last_spec_date',
            'first_contracted_date', 'last_last_contracted_date',
            'first_delivery_date', 'last_delivery_date',
            'total_deliveried_quantity', 'mean_deliveried_quantity', 'max_delivered_quantity',
            'total_volume_contracted', 'volume_contracted_zpp4', 'volume_contracted_all',
            'volume_dс_diff_zpp4', 'volume_dс_diff_all',
            'volume_dc_ratio_zpp4', 'volume_dc_ratio_all',
            'quantity',
            'mean_deliveried_quality', 'max_deliveried_quality',
            'min_deliveried_quality', 'median_deliveried_quality']
        
        for c in new_columns_for_df:
            if c not in results.columns.tolist():
                results[c] = 0
        
        results = results[new_columns_for_df]

        return results

    def transform_data(self, features, save=True):
        """Преобразование данных в диапазон от 0 до 1, с сохранением преобразователей"""
        transformed_features = pd.DataFrame()

        for c in features.columns.tolist():
            if c in list(self.features_to_use.keys()):
                transformer = QuantileTransformer()
                transformed_features[c] = [x[0] for x in transformer.fit_transform(features[c].values.reshape(-1, 1))]
                if save:
                    joblib.dump(transformer, self.TRANSFORMERS_DIR + c + '_transformer.joblib')
        features_list = transformed_features.columns.tolist()
        transformed_features[['date', 'supplier', 'supplier_status']] = features[
            ['date', 'supplier', 'supplier_status']]
        transformed_features = transformed_features[['date', 'supplier', 'supplier_status'] + features_list]

        return transformed_features, features_list

    def transform_new_data(self, new_data):
        """Преобразование данных в диапазон от 0 до 1, с использованием сохраненных преобразователей"""
        transformed_features = pd.DataFrame()

        for c in new_data.columns.tolist():
            if c in list(self.features_to_use.keys()):
                transformer = joblib.load(self.TRANSFORMERS_DIR + c + '_transformer.joblib')
                transformed_features[c] = [x[0] for x in transformer.transform(new_data[c].values.reshape(-1, 1))]
        features_list = transformed_features.columns.tolist()
        transformed_features[['date', 'supplier', 'supplier_status']] = new_data[
            ['date', 'supplier', 'supplier_status']]
        transformed_features = transformed_features[['date', 'supplier', 'supplier_status'] + features_list]

        return transformed_features, features_list
