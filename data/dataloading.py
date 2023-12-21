"""
Модуль содержит функции и классы для загрузки данных для pytorch моделей
"""

import numpy as np
import pandas as pd
from random import shuffle
import torch
from torch.utils.data import Dataset, Sampler

import sys
import os
sys.path.append(os.path.abspath('..'))
from data.feature_generation import zpp4_preporarion, zpp4_preporarion2


class TimeSeriesDataset(Dataset):
    """Класс для загрузки данных из датафрейма для обучения автоэнкодера.

    Attributes:
        df (pd.DataFrame): Датафрейм с предобработанными данными
    """
    def __init__(self, df, learn_cols):
        self.df = df
        self.learn_cols = learn_cols

        self.specs = df['id'].unique()
        self.specs_date = [df.loc[df['id'] == spec_id]['delivery_period_end'].iloc[0]
                           for spec_id in self.specs]
        self.supplier = [df.loc[df['id'] == spec_id]['supplier'].iloc[0]
                         for spec_id in self.specs]

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        data = self.df.loc[self.df['id'] == self.specs[idx]]

        x = torch.tensor(data[self.learn_cols].values, dtype=torch.float)

        return x


def pad_collate(batch):
    """Функция добавляет паддинг к элементам в batch"""
    max_len = max([len(sample) for sample in batch])
    new_batch = []
    for sample in batch:
        repeat_n = max_len - len(sample)
        sample_x = torch.cat((sample, sample[-1].unsqueeze(0).repeat(repeat_n, 1)))

        new_batch.append(sample_x)

    x = torch.stack(new_batch)
    return x


class SequenceLengthSampler(Sampler):
    """Функция семплирует данные по поставкам по количеству записей.
    Это необходимо для уменьшения количества паддингов.
    """

    def __init__(self, ind_n_len, bucket_boundaries, batch_size=64, ):
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size

    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k],
                                         int(data_buckets[k].shape[0] / self.batch_size)))
        shuffle(iter_list)  # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list:
            yield i.tolist()  # as it was stored in an array

    def __len__(self):
        return len(self.ind_n_len)

    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
            np.less_equal(buckets_min, seq_length),
            np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id


def load_dataset():
    """Функция для загрузки и подготовки данных о поставках.
    """

    # таблица со спецификациями
    spec = pd.read_csv('../data/processed_data/specs.csv')
    spec['spec_date'] = pd.to_datetime(spec['spec_date'], format='%Y-%m-%d')
    spec['delivery_period_end'] = pd.to_datetime(spec['delivery_period_end'], format='%Y-%m-%d')

    # таблица с доставками
    zpp4 = pd.read_csv('../data/processed_data/zpp4.csv')
    zpp4['date'] = pd.to_datetime(zpp4['date'], format='%Y-%m-%d')
    zpp4['spec_date'] = pd.to_datetime(zpp4['spec_date'], format='%Y-%m-%d')

    # генерация переменных
    zpp4 = zpp4_preporarion(zpp4, spec)

    # выбираем колонки, которые будут участвовать в обучении, либо
    # колонки идентифкаторы ('id', 'supplier')
    zpp4 = zpp4[['id', 'supplier', 'delivery_period_end', 'lateness_percentage', 'weight_percentage', 'price_change']]

    # масштабируем переменную перед обучением модели
    zpp4['price_change'] = zpp4['price_change'] * 0.4

    dataset = TimeSeriesDataset(zpp4, learn_cols=['lateness_percentage', 'weight_percentage', 'price_change'])
    return dataset

def load_dataset2():
    """Функция для загрузки и подготовки данных о поставках.
    """

    # таблица со спецификациями
    spec = pd.read_csv('../data/processed_data/specs.csv')
    spec['spec_date'] = pd.to_datetime(spec['spec_date'], format='%Y-%m-%d')
    spec['delivery_period_end'] = pd.to_datetime(spec['delivery_period_end'], format='%Y-%m-%d')

    # таблица с доставками
    zpp4 = pd.read_csv('../data/processed_data/zpp4.csv')
    zpp4['date'] = pd.to_datetime(zpp4['date'], format='%Y-%m-%d')
    zpp4['spec_date'] = pd.to_datetime(zpp4['spec_date'], format='%Y-%m-%d')

    # генерация переменных
    zpp4 = zpp4_preporarion2(zpp4, spec)

    # выбираем колонки, которые будут участвовать в обучении и
    # колонки идентифкаторы ('id', 'supplier')
    zpp4 = zpp4[['id', 'supplier', 'delivery_period_end', 'time_persentage', 'weight_percentage', 'price_change']]

    # масштабируем переменную перед обучением модели
    zpp4['price_change'] = (zpp4['price_change'] + 5) * 0.01

    dataset = TimeSeriesDataset(zpp4, learn_cols=['time_persentage', 'weight_percentage', 'price_change'])
    return dataset