import torch
from torch.utils.data import Dataset, DataLoader
import collections
import pickle
from Sampler import get_diabetes_peptides
import pandas as pd
import numpy as np
import math


class SignedPairsDataset(Dataset):
    def __init__(self, samples, train_dicts):
        self.data = samples
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}
        vatox, vbtox, jatox, jbtox, mhctox = train_dicts
        self.vatox = vatox
        self.vbtox = vbtox
        self.jatox = jatox
        self.jbtox = jbtox
        self.mhctox = mhctox
        # self.pos_weight_factor = 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

    def aa_convert(self, seq):
        if seq == np.nan:
            seq = []
        else:
            seq = []
        return seq

    @staticmethod
    def get_max_length(x):
        return len(max(x, key=len))

    def seq_letter_encoding(self, seq):
        def _pad(_it, _max_len):
            return _it + [0] * (_max_len - len(_it))
        return [_pad(it, self.get_max_length(seq)) for it in seq]

    def seq_one_hot_encoding(self, tcr, max_len=28):
        tcr_batch = list(tcr)
        padding = torch.zeros(len(tcr_batch), max_len, 20 + 1)
        # TCR is converted to numbers at this point
        # We need to match the autoencoder atox, therefore -1
        for i in range(len(tcr_batch)):
            # missing alpha
            if tcr_batch[i] == [0]:
                continue
            tcr_batch[i] = tcr_batch[i] + [self.atox['X']]
            for j in range(min(len(tcr_batch[i]), max_len)):
                padding[i, j, tcr_batch[i][j] - 1] = 1
        return padding

    def label_encoding(self):
        pass

    def binary_encoding(self):
        pass

    def hashing_encoding(self):
        pass

    @staticmethod
    def binarize(num):
        l = []
        while num:
            l.append(num % 2)
            num //= 2
        l.reverse()
        # print(l)
        return l

    def collate(self, batch, tcr_encoding, cat_encoding):
        lst = []
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
        tcrb = [self.aa_convert(sample['tcrb']) for sample in batch]
        tcra = [self.aa_convert(sample['tcra']) for sample in batch]
        if tcr_encoding == 'AE':
            lst.append(torch.FloatTensor(self.seq_one_hot_encoding(tcra, max_len=34)))
            lst.append(torch.FloatTensor(self.seq_one_hot_encoding(tcrb)))
        elif tcr_encoding == 'LSTM':
            lst.append(torch.LongTensor(self.seq_letter_encoding(tcra)))
            # we do not sent the length, so that ae and lstm batch output be similar
            lst.append(torch.LongTensor(self.seq_letter_encoding(tcrb)))
        # Peptide
        peptide = [self.aa_convert(sample['peptide']) for sample in batch]
        lst.append(torch.LongTensor(self.seq_letter_encoding(peptide)))
        categorical = ['va', 'vb', 'ja', 'jb', 'mhc']
        cat_idx = [self.vatox, self.vbtox, self.jatox, self.jbtox, self.mhctox]
        for cat, idx in zip(categorical, cat_idx):
            batch_cat = ['UNK' if pd.isna(sample[cat]) else sample[cat] for sample in batch]
            batch_idx = list(map(lambda x: idx[x] if x in idx else 0, batch_cat))
            if cat_encoding == 'embedding':
                # label encoding
                batch_cat = torch.LongTensor(batch_idx)
            if cat_encoding == 'binary':
                # we need a matrix for the batch with the binary encodings
                # hyperparam ?
                max_len = 10
                def bin_pad(num, _max_len):
                    bin_list = self.binarize(num)
                    return [0] * (_max_len - len(bin_list)) + bin_list
                bin_mat = torch.tensor([bin_pad(v, max_len) for v in batch_idx]).float()
                batch_cat = bin_mat
            lst.append(batch_cat)
        # T cell type (or MHC class)

        t_type_dict = {'CD4': 1, 'CD8': 0, 'MHCII': 1, 'MHCI': 0, 0:0, 1:1, 'UNK' : -1}
        # nan and other values are 2
        t_type = [sample['t_cell_type'] for sample in batch]
        # print('t_type', t_type)
        convert_type = lambda x: t_type_dict[x] #if x in t_type_dict else 0
        t_type_tensor = torch.FloatTensor(list(map(convert_type, t_type)))
        lst.append(t_type_tensor)
        sign = [sample['sign'] for sample in batch]
        lst.append(torch.FloatTensor(sign))
        weight = [sample['weight'] for sample in batch]
        lst.append(torch.FloatTensor(weight))
        return lst
    pass


def get_index_dicts(train_samples):
    samples = train_samples
    all_va = [sample['va'] for sample in samples if not pd.isna(sample['va'])]
    vatox = {va: index for index, va in enumerate(sorted(set(all_va)), 1)}
    vatox['UNK'] = 0
    all_vb = [sample['vb'] for sample in samples if not pd.isna(sample['vb'])]
    vbtox = {vb: index for index, vb in enumerate(sorted(set(all_vb)), 1)}
    vbtox['UNK'] = 0
    all_ja = [sample['ja'] for sample in samples if not pd.isna(sample['ja'])]
    jatox = {ja: index for index, ja in enumerate(sorted(set(all_ja)), 1)}
    jatox['UNK'] = 0
    all_jb = [sample['jb'] for sample in samples if not pd.isna(sample['jb'])]
    jbtox = {jb: index for index, jb in enumerate(sorted(set(all_jb)), 1)}
    jbtox['UNK'] = 0
    all_mhc = [sample['mhc'] for sample in samples if not pd.isna(sample['mhc'])]
    mhctox = {mhc: index for index, mhc in enumerate(sorted(set(all_mhc)), 1)}
    mhctox['UNK'] = 0
    return [vatox, vbtox, jatox, jbtox, mhctox]

