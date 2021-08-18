import torch
from torch.utils.data import Dataset, DataLoader
import collections
import pickle
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
        self.hla = [letter for letter in 'HLA-BCDQR*:0123456789']
        self.htox = {amino: index for index, amino in enumerate(['PAD'] + self.hla)}
        self.pos_weight_factor = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sign = sample['sign']
        if sign == 0:
            weight = 1
        elif sign == 1:
            weight = 1
        sample['weight'] = weight
        return sample

    def aa_convert(self, seq):
        if seq == 'UNK':
            seq = []
        else:
            seq = [self.atox[aa] for aa in seq]
        return seq

    @staticmethod
    def get_max_length(x):
        return 14 #len(max(x, key=len))

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
        # I think that feature hashing is not relevant in this case
        pass

    @staticmethod
    def binarize(num):
        l = []
        while num:
            l.append(num % 2)
            num //= 2
        l.reverse()
        return l

    def convert(self, seq, type):
            if type == 'hla':
                seq = [self.htox[l] for l in seq]
            return seq
    def pad_sequence(self, seq):
        def _pad(_it, _max_len):
            return _it + [0] * (_max_len - len(_it))
        return [_pad(it, self.get_max_length(seq)) for it in seq]


    def collate(self, batch, tcr_encoding, cat_encoding):
        lst = []
        # TCRs
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
        tcrb = [self.aa_convert(sample['tcrb']) for sample in batch]
        tcra = [self.aa_convert(sample['tcra']) for sample in batch]
        if tcr_encoding == 'AE':
            lst.append(torch.FloatTensor(self.seq_one_hot_encoding(tcra, max_len=34)))
            lst.append(torch.FloatTensor(self.seq_one_hot_encoding(tcrb)))

        #MHC
        mhc = [self.convert(sample['mhc'], type='hla') for sample in batch]
        lst.append(torch.LongTensor(self.pad_sequence(mhc)))
        categorical = ['va', 'vb', 'ja', 'jb']
        cat_idx = [self.vatox, self.vbtox, self.jatox, self.jbtox]
        for cat, idx in zip(categorical, cat_idx):
            batch_cat = ['UNK' if pd.isna(sample[cat]) else sample[cat] for sample in batch]
            batch_idx = list(map(lambda x: idx[x] if x in idx else 0, batch_cat))
            #print(cat)
            if cat_encoding == 'embedding':
                # label encoding
                batch_cat = torch.LongTensor(batch_idx)
            if cat_encoding == 'binary':
                # we need a matrix for the batch with the binary encodings
                max_len = 10
                def bin_pad(num, _max_len):
                    bin_list = self.binarize(num)
                    return [0] * (_max_len - len(bin_list)) + bin_list
                bin_mat = torch.tensor([bin_pad(v, max_len) for v in batch_idx]).float()
                batch_cat = bin_mat
            lst.append(batch_cat)
        # T cell type (or MHC class)
        t_type_dict = {'CD4': 2, 'CD8': 1, 'MHCII': 2, 'MHCI': 1}
        # nan and other values are 2
        t_type = [sample['t_cell_type'] for sample in batch]
        convert_type = lambda x: t_type_dict[x] if x in t_type_dict else 0
        t_type_tensor = torch.FloatTensor(list(map(convert_type, t_type)))
        lst.append(t_type_tensor)
        # Sign
        sign = [sample['sign'] for sample in batch]
        lst.append(torch.FloatTensor(sign))
        factor = self.pos_weight_factor
        weight = [sample['weight'] for sample in batch]
        #print(weight)
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



def check():
    dct = 'Samples/'
    with open(dct + 'mcpas_human_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)
    with open(dct + 'mcpas_human_test_samples.pickle', 'rb') as handle:
        test = pickle.load(handle)
    dicts = get_index_dicts(train)
    vatox, vbtox, jatox, jbtox, mhctox = dicts
    train_dataset = SignedPairsDataset(train, dicts)
    test_dataset = SignedPairsDataset(test, dicts)

    train_dataset = DiabetesDataset(train, dicts, weight_factor=10)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=4,
                            collate_fn=lambda b: train_dataset.collate(b, tcr_encoding='lstm',
                                                                       cat_encoding='embedding'))
    for batch in train_dataloader:
        print(batch)
        break
        # exit()
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4,
                                  collate_fn=lambda b: train_dataset.collate(b, tcr_encoding='lstm',
                                                                             cat_encoding='embedding'))
    for batch in test_dataloader:
        print(batch)
        break
    print('successful')

# check()
