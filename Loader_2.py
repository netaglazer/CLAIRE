import torch
from torch.utils.data import Dataset, DataLoader
import collections
import pickle
# from Sampler import get_diabetes_peptides
import pandas as pd
import numpy as np
import math
# from collections import

# another problem - standatization of v,j,mhc format (mainly in mcpas)


class MHCPepDataset(Dataset):
    def __init__(self, datafile):
        self.data = pd.read_csv(datafile, engine='python')
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}
        self.hla = [letter for letter in 'HLA-BC*:0123456789']
        self.htox = {amino: index for index, amino in enumerate(['PAD'] + self.hla)}

    def __len__(self):
        return len(self.data)

    def invalid(self, seq, type):
        if type == 'aa':
            return pd.isna(seq) or any([aa not in self.amino_acids for aa in seq])
        elif type == 'hla':
            return pd.isna(seq) or any([l not in self.hla for l in seq])

    def __getitem__(self, index):
        def convert(seq, type):
            if seq == 'UNK':
                seq = [0]
            else:
                if type == 'aa':
                    seq = [self.atox[aa] for aa in seq]
                if type == 'hla':
                    seq = [self.htox[l] for l in seq]
            return seq
        peptide = self.data['Description'][index]
        len_p = len(peptide)
        if self.invalid(peptide, type='aa'):
            peptide = 'UNK'
            len_p = 1
        species = self.data['Name'][index]
        label = self.data['Qualitative Measure'][index]
        mhc = self.data['Allele Name'][index]
        len_m = len(mhc)
        if not (mhc.startswith('HLA-A*') or mhc.startswith('HLA-B*') or mhc.startswith('HLA-C*'))\
                or self.invalid(mhc, type='hla'):
            mhc = 'UNK'
            len_m = 0
        if label in ['Positive', 'Positive-High', 'Positive-Intermediate']:
            sign = 1
        elif label in ['Negative', 'Positive-Low']:
            sign = 0
        else:
            print(label)
        peptide = convert(peptide, type='aa')
        mhc = convert(mhc, type='hla')
        weight = 1
        # should we add negative factor?
        # if sign == 1:
        #     weight = 5
        # else:
        #     weight = 1
        sample = (peptide, len_p, mhc, len_m, float(sign), float(weight))
        return sample

    @staticmethod
    def get_max_length(x):
        return 14 #len(max(x, key=len))

    def pad_sequence(self, seq, max_len=14):
        def _pad(_it, _max_len):
            # ignore if too long and max_len is fixed
            if len(_it) > _max_len:
                return [0] * _max_len
            return _it + [0] * (_max_len - len(_it))
        if max_len is None:
            return [_pad(it, self.get_max_length(seq)) for it in seq]
        else:
            return [_pad(it, max_len) for it in seq]

    def one_hot_encoding(self, tcr, max_len=28):
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

    def ae_collate(self, batch):
        tcra, len_a, tcrb, len_b, peptide, len_p, sign, weight = zip(*batch)
        lst = []
        lst.append(torch.FloatTensor(self.one_hot_encoding(tcra, max_len=34)))
        lst.append(torch.FloatTensor(self.one_hot_encoding(tcrb)))
        lst.append(torch.LongTensor(self.pad_sequence(peptide)))
        lst.append(torch.LongTensor(len_p))
        lst.append(torch.FloatTensor(sign))
        lst.append(torch.FloatTensor(weight))
        return lst

    def cnn_collate(self, batch):
        peptide, len_p, mhc, len_m, sign, weight = zip(*batch)
        lst = []
        lst.append(torch.LongTensor(self.pad_sequence(peptide, max_len=11)))
        lst.append(torch.LongTensor(self.pad_sequence(mhc)))
        lst.append(torch.FloatTensor(sign))
        lst.append(torch.FloatTensor(weight))
        return lst

    def lstm_collate(self, batch):
        transposed = zip(*batch)
        lst = []
        for samples in transposed:
            if isinstance(samples[0], int):
                lst.append(torch.LongTensor(samples))
            elif isinstance(samples[0], float):
                lst.append(torch.FloatTensor(samples))
            elif isinstance(samples[0], collections.Sequence):
                lst.append(torch.LongTensor(self.pad_sequence(samples)))
        return lst
    pass


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
        # this will be label with learned embedding matrix (so not 1 dimension)
        # get all possible tags
        # in init ?
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
        # print(l)
        return l

    def convert(self, seq, type):
            if type == 'hla':
                # print(seq)
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
        #print(mhc)
        #print('pad', self.pad_sequence(mhc))
        #print('len pad', len(self.pad_sequence(mhc)[0]))
        lst.append(torch.LongTensor(self.pad_sequence(mhc)))
        # Peptide
        # peptide = [self.aa_convert(sample['peptide']) for sample in batch]
        # lst.append(torch.LongTensor(self.seq_letter_encoding(peptide)))
        # Categorical features - V alpha, V beta, J alpha, J beta, MHC
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
                # hyperparam ?
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
        #print(batch[0])
        weight = [sample['weight'] for sample in batch]
        #print(weight)
        lst.append(torch.FloatTensor(weight))
        # weight will be handled in trainer (it is not loader job) -
        # It is - this is how we mark diabetes to get heavier weight
        # lst.append(torch.FloatTensor(weight))
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


#
# class SinglePeptideDatasetWeighted(SignedPairsDataset):
#     def __init__(self, samples, train_dicts, weight_factor, d_weight_factor):
#         super().__init__(samples, train_dicts)
#         self.diabetes_peptides = get_diabetes_peptides('data/McPAS-TCR.csv')
#         self.weight_factor = weight_factor
#         self.d_weight_factor = d_weight_factor
#
#     def __getitem__(self, index):
#         sample = super().__getitem__(index)
#         if (sample['t_cell_type'] == 'CD4' or sample['t_cell_type'] == 'MHCII'):
#              # print('CD4')
#              sample['weight'] *= self.weight_factor
#         if sample['peptide'] in self.diabetes_peptides:
#             # print('diabetes')
#             sample['weight'] *= self.d_weight_factor
#
#         return sample
#     pass

class DiabetesDataset(SignedPairsDataset):
    def __init__(self, samples, train_dicts, weight_factor):
        super().__init__(samples, train_dicts)
        self.diabetes_peptides = get_diabetes_peptides('data/McPAS-TCR.csv')
        self.weight_factor = weight_factor

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        weight = sample['weight']
        peptide = sample['peptide']
        if peptide in self.diabetes_peptides:
            weight *= self.weight_factor
        return sample
    pass


class SinglePeptideDataset(SignedPairsDataset):
    def __init__(self, samples, train_dicts, peptide, force_peptide=False, spb_force=False):
        super().__init__(samples, train_dicts)
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}
        self.force_peptide = force_peptide
        self.spb_force = spb_force
        self.peptide = peptide

    def __getitem__(self, index):
        sample = self.data[index]
        # weight does not matter, we do not train with this dataset
        sample['weight'] = 1
        if self.force_peptide:
            # we keep the original positives, else is negatives
            if self.spb_force:
                if sample['peptide'] != self.peptide:
                    sample['sign'] = 0
                return sample
            # we do it only for MPS (and we have to check the true peptide)
            else:
                sample['peptide'] = self.peptide
                return sample
        else:
            # original spb task
            # print(sample['peptide'])
            if sample['peptide'] != self.peptide:
                # print(sample['peptide'])
                return None
            return sample
    pass


def check():
    dct = 'Samples/'
    with open(dct + 'mcpas_human_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)
    with open(dct + 'mcpas_human_test_samples.pickle', 'rb') as handle:
        test = pickle.load(handle)
    dicts = get_index_dicts(train)
    vatox, vbtox, jatox, jbtox, mhctox = dicts
    # print(len(vatox))
    # for v in vatox:
    #     print(v)
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