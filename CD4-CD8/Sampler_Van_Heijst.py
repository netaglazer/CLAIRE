import pandas as pd
import numpy as np
import random
import pickle
import time
from datetime import datetime as dt


def fix_list_hla(x):
    x_to_return = []
    for i in x:
        x_to_return.append(i.tolist()[0])
    return x_to_return

def weight(vc, x):
    weight_1 = float(vc[0] / vc[1])
    print(weight_1)
    if x == 1:
        return weight_1
    return 1

def read_data(datafile, file_key, human=True):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid_pep(seq):
        if(pd.isna(seq) or any([aa not in amino_acids for aa in seq])):
            return 'invalid'
        return seq
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])

    if file_key == 'new_dataset':
        data = pd.read_pickle(datafile)
        CD_list = ['CD8', 'CD4']
        data = data[data['t_cell_type'].isin(CD_list)]
        data = data.reset_index(drop=True)
        data['fixed_hla_list'] = data['hla'].apply(lambda x: fix_list_hla(x)) 
        data = data.replace({'CD8': 0, 'CD4': 1})
        print(data['t_cell_type'].value_counts())
        data = data.reset_index(drop=True)

        for index in range(len(data)):
            sample = {}
            sample['tcra'] = 'UNK' #data['CDR3.alpha.aa'][index]
            sample['tcrb'] = data['junction_aa'][index]
            sample['va'] = 'UNK' #data['TRAV'][index]
            sample['ja'] = 'UNK' #data['TRAJ'][index]
            sample['vb'] = data['v_call'][index]
            sample['jb'] = data['j_call'][index]
            sample['t_cell_type'] = data['t_cell_type'][index]
            sample['peptide'] = 'UNK' #data['Epitope.peptide'][index]
            sample['protein'] = 'UNK' # data['Antigen.protein'][index]
            sample['mhc'] = data['fixed_hla_list'][index]
            if invalid(sample['tcrb']):
                # print(sample['tcrb'])
                continue
            if human and data['Species'][index] != 'Human':
                print('sdgfsdf')
                continue
            if invalid(sample['tcra']):
                sample['tcra'] = 'UNK'
            all_pairs.append(sample)


    if file_key == 'cd_data':
        data = pd.read_csv(datafile, engine='python')
        CD_list = ['CD8', 'CD4']

        for index in range(len(data)):
            sample = {}
            sample['tcra'] = 'UNK'
            sample['tcrb'] = data['junction_aa'][index]
            sample['va'] = 'UNK'
            sample['ja'] = 'UNK'
            sample['vb'] = data['v_call'][index]
            sample['jb'] = data['j_call'][index]
            sample['t_cell_type'] = data['t_cell_type'][index]
            sample['peptide'] = 'UNK'
            sample['protein'] = 'UNK'
            sample['mhc'] = 'UNK'
            if invalid(sample['tcrb']):
                continue
            all_pairs.append(sample)



    elif file_key == 'vdjdb':
        print('vdjdb')
        data = pd.read_csv(datafile)
        print(data['MHC class'].value_counts())
        # first read all TRB, then unite with TRA according to sample id
        paired = {}
        for index in range(len(data)):
            # print(data.head())
            sample = {}
            id = int(data['complex.id'][index])
            type = data['Gene'][index]
            tcr = data['CDR3'][index]
            if type == 'TRB':
                sample['tcrb'] = tcr
                sample['tcra'] = 'UNK'
                sample['va'] = 'UNK'
                sample['ja'] = 'UNK'
                sample['vb'] = data['V'][index]
                sample['jb'] = data['J'][index]
                sample['peptide'] = data['Epitope'][index]
                sample['protein'] = data['Epitope gene'][index]
                sample['mhc'] = data['MHC A'][index]
                # here it's mhc class
                sample['t_cell_type'] = data['MHC class'][index]
                if invalid(tcr) or invalid(sample['peptide']):
                    continue
                # only TRB
                if id == 0:
                    all_pairs.append(sample)
                else:
                    paired[id] = sample
            if type == 'TRA':
                tcra = tcr
                if invalid(tcra):
                    tcra = 'UNK'
                sample = paired[id]
                sample['va'] = data['V'][index]
                sample['ja'] = data['J'][index]
                sample['tcra'] = tcra
                sample['t_cell_type'] = data['MHC class'][index]

                paired[id] = sample
        all_pairs.extend(list(paired.values()))
        
        
    print(len(all_pairs))
    # assimung each sample appears only once in the dataset
    train_pairs, test_pairs = train_test_split(all_pairs)
    df_train = pd.DataFrame.from_dict(train_pairs)
    df_test = pd.DataFrame.from_dict(test_pairs)
    vc = df_train['t_cell_type'].value_counts()
    df_train['weight'] = df_train['t_cell_type'].apply(lambda x: weight(vc, x))
    df_test['weight'] = df_test['t_cell_type'].apply(lambda x: weight(vc, x))
    train_pairs = df_train.to_dict('records')
    test_pairs = df_test.to_dict('records')
    return all_pairs, train_pairs, test_pairs


def read_all_data(datafile, file_key, human=True):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    if file_key == 'mcpas':
        data = pd.read_csv(datafile, engine='python')

        for index in range(len(data)):
            sample = {}
            sample['tcra'] = data['CDR3.alpha.aa'][index]
            sample['tcrb'] = data['CDR3.beta.aa'][index]
            sample['va'] = data['TRAV'][index]
            sample['ja'] = data['TRAJ'][index]
            sample['vb'] = data['TRBV'][index]
            sample['jb'] = data['TRBJ'][index]
            sample['t_cell_type'] = data['T.Cell.Type'][index]
            sample['peptide'] = data['Epitope.peptide'][index]
            sample['protein'] = data['Antigen.protein'][index]
            sample['mhc'] = data['MHC'][index]
            if invalid(sample['tcrb']) or invalid(sample['peptide']):
                continue
            if human and data['Species'][index] != 'Human':
                continue
            if invalid(sample['tcra']):
                sample['tcra'] = 'UNK'
            all_pairs.append(sample)
    elif file_key == 'vdjdb':
        data = pd.read_csv(datafile, engine='python', sep='\t')
        # first read all TRB, then unite with TRA according to sample id
        paired = {}
        for index in range(len(data)):
            sample = {}
            id = int(data['complex.id'][index])
            type = data['Gene'][index]
            tcr = data['CDR3'][index]
            if type == 'TRB':
                sample['tcrb'] = tcr
                sample['tcra'] = 'UNK'
                sample['va'] = 'UNK'
                sample['ja'] = 'UNK'
                sample['vb'] = data['V'][index]
                sample['jb'] = data['J'][index]
                sample['peptide'] = data['Epitope'][index]
                sample['protein'] = data['Epitope gene'][index]
                sample['mhc'] = data['MHC A'][index]
                # here it's mhc class
                sample['t_cell_type'] = data['MHC class'][index]
                if invalid(tcr) or invalid(sample['peptide']):
                    continue
                # only TRB
                if id == 0:
                    all_pairs.append(sample)
                else:
                    paired[id] = sample
            if type == 'TRA':
                tcra = tcr
                if invalid(tcra):
                    tcra = 'UNK'
                sample = paired[id]
                sample['va'] = data['V'][index]
                sample['ja'] = data['J'][index]
                sample['tcra'] = tcra
                paired[id] = sample
        all_pairs.extend(list(paired.values()))
    train_pairs = all_pairs
    return all_pairs, train_pairs



def train_test_split(all_pairs):
    '''
    Splitting the TCR-PEP pairs
    '''
    train_pairs = []
    test_pairs = []
    for pair in all_pairs:
        # 80% train, 20% test
        p = np.random.binomial(1, 0.80)
        if p == 1:
            train_pairs.append(pair)
        else:
            test_pairs.append(pair)
    return train_pairs, test_pairs


def positive_examples(pairs):
    pos_samples = []
    for sample in pairs:
        sample['sign'] = 1
        pos_samples.append(sample)
    return pos_samples


def negative_examples(pairs, all_pairs, size):
    '''
    Randomly creating intentional negative examples from the same pairs dataset.
    We match randomly tcr data with peptide data to make a sample
    '''
    neg_samples = []
    i = 0
    # tcrs = [tcr_data for (tcr_data, pep_data) in pairs]
    # peps = [pep_data for (tcr_data, pep_data) in pairs]
    while i < size:
        # choose randomly two samples. match tcr data with pep data
        pep_sample = random.choice(pairs)
        tcr_sample = random.choice(pairs)
        sample = {}
        sample['tcra'] = tcr_sample['tcra']
        sample['tcrb'] = tcr_sample['tcrb']
        sample['va'] = tcr_sample['va']
        sample['ja'] = tcr_sample['ja']
        sample['vb'] = tcr_sample['vb']
        sample['jb'] = tcr_sample['jb']
        sample['t_cell_type'] = tcr_sample['t_cell_type']
        sample['peptide'] = pep_sample['peptide']
        sample['protein'] = pep_sample['protein']
        sample['mhc'] = pep_sample['mhc']
        if sample not in all_pairs and sample not in neg_samples:
                sample['sign'] = 0
                neg_samples.append(sample)
                i += 1
    return neg_samples


def get_examples(datafile, file_key, human):
    all_pairs, train_pairs, test_pairs = read_data(datafile, file_key, human)
    train_pos = positive_examples(train_pairs)
    test_pos = positive_examples(test_pairs)
    train = train_pos # + train_neg
    random.shuffle(train)
    test = test_pos #+ test_neg
    random.shuffle(test)
    return train, test


def sample_data(datafile, file_key, train_file, test_file, human=True):
    print(datafile, file_key, train_file, test_file)
    train, test = get_examples(datafile, file_key, human)
    with open(str(train_file) + '.pickle', 'wb') as handle:
        pickle.dump(train, handle)
    with open(str(test_file) + '.pickle', 'wb') as handle:
        pickle.dump(test, handle)


def sample():
    t1 = time.time()
    print('sampling mcpas...')
    sample_data('new_dataset/trb_new_dataset.pickle', 'new_dataset', 'for_emerson_new_dataset_train_samples', 'for_emerson_new_dataset_test_samples', human=False)
    t2 = time.time()
    print('done in ' + str(t2 - t1) + ' seconds')
    pass




if __name__ == '__main__':
    sample()
    pass


