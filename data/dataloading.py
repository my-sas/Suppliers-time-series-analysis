import numpy as np
# import pandas as pd
from random import shuffle
import torch
from torch.utils.data import Dataset, Sampler


class TimeSeriesDataset(Dataset):
    def __init__(self, df, spec_cols, series_cols, time_cols):
        self.df = df
        self.spec_cols = spec_cols
        self.series_cols = series_cols
        self.time_cols = time_cols
        self.specs = df['id'].unique()
        self.specs_date = [df.loc[df['id'] == spec_id]['last_date'].iloc[0]
                           for spec_id in self.specs]
        self.supplier = [df.loc[df['id'] == spec_id]['supplier'].iloc[0]
                         for spec_id in self.specs]

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        data = self.df.loc[self.df['id'] == self.specs[idx]]

        h = torch.tensor(data[self.spec_cols].iloc[0].values, dtype=torch.float)
        x = torch.tensor(data[self.series_cols].values, dtype=torch.float)
        t = torch.tensor(data[self.time_cols].values, dtype=torch.float)

        return h, x, t


def pad_collate(batch):
    max_len = max([len(sample[1]) for sample in batch])
    new_batch = []
    for sample in batch:
        repeat_n = max_len - len(sample[1])

        sample_h = sample[0]
        sample_x = torch.cat((sample[1], sample[1][-1].unsqueeze(0).repeat(repeat_n, 1)))
        sample_t = torch.cat((sample[2], sample[2][-1].unsqueeze(0).repeat(repeat_n, 1)))

        new_batch.append((
            sample_h,
            sample_x,
            sample_t
        ))

    h = torch.stack([sample[0] for sample in new_batch])
    x = torch.stack([sample[1] for sample in new_batch])
    t = torch.stack([sample[2] for sample in new_batch])
    return h, x, t


class SequenceLengthSampler(Sampler):

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
