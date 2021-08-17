import pandas as pd
import numpy as np
import random
import pickle
import time

def weight(vc, x):
    # print(vc)
    weight_1 = float(vc[0] / vc[1])
    # print('vc', vc)
    # print('weight_1', weight_1)
    # print('weight_1', weight_1)
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

    if file_key == 'mcpas':
        data = pd.read_csv(datafile, engine='python')
        CD_list = ['CD8', 'CD4']
        data = data[data['T.Cell.Type'].isin(CD_list)]
        print('len data', len(data))
        data['Epitope.peptide'] = data['Epitope.peptide'].apply(lambda x: invalid_pep(x))
        data = data[data['Epitope.peptide'] != 'invalid']
        print('len data', len(data))

        print('columns', data.columns)
        data = data.reset_index(drop=True)

        # cd4_df = data[data['T.Cell.Type'] == 'CD4']
        # cd8_df = data[data['T.Cell.Type'] == 'CD8']
        # print('cd4_df', len(cd4_df))
        # print('cd8_df', len(cd8_df))
        # cd8_df = cd8_df.sample(n=len(cd4_df))
        # print('cd8_df', len(cd8_df))
        #
        # data = pd.concat([cd8_df, cd4_df], axis=0)
        # print(len(data))

        data = data.replace({'CD8': 0, 'CD4': 1})
        print(data['T.Cell.Type'].value_counts())
        data = data.reset_index(drop=True)


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
            if invalid(sample['tcrb']):
                print(sample['tcrb'])
                continue
            if human and data['Species'][index] != 'Human':
                print('sdgfsdf')
                continue
            if invalid(sample['tcra']):
                sample['tcra'] = 'UNK'
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
                print('TRA')
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
    # print('train_pairs', df_train.head())
    # print(pd.concat([df_train, df_test], axis=0)['t_cell_type'].value_counts())
    vc = df_train['t_cell_type'].value_counts()
    # print('vc', vc)
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
    # assimung each sample appears only once in the dataset
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
        p = np.random.binomial(1, 0.8)
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

# Removing this function - assuming every (tcrb,pep) pair appears only once in a dataset
# def is_negative(all_pairs, tcrb, pep):
#     for sample in all_pairs:
#         # we do not check for full sample match, this is enough
#         if sample['tcrb'] == tcrb and sample['peptide'] == pep:
#             return False
#     return True


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
    # train_neg = negative_examples(train_pairs, all_pairs, 5 * len(train_pos))
    # test_neg = negative_examples(test_pairs, all_pairs, 5 * len(test_pos))
    train = train_pos # + train_neg
    random.shuffle(train)
    test = test_pos #+ test_neg
    random.shuffle(test)
    return train, test

def get_all_examples(datafile, file_key, human):
    all_pairs, train_pairs, test_pairs = read_all_data(datafile, file_key, human)
    train_pos = positive_examples(train_pairs)
    test_pos = positive_examples(test_pairs)
    # train_neg = negative_examples(train_pairs, all_pairs, 5 * len(train_pos))
    # test_neg = negative_examples(test_pairs, all_pairs, 5 * len(test_pos))
    train = train_pos + train_neg
    random.shuffle(train)
    # test = test_pos + test_neg
    # random.shuffle(test)
    return train#, test


def sample_data(datafile, file_key, train_file, test_file, human=True):
    train, test = get_examples(datafile, file_key, human)
    with open(str(train_file) + '.pickle', 'wb') as handle:
        pickle.dump(train, handle)
    with open(str(test_file) + '.pickle', 'wb') as handle:
        pickle.dump(test, handle)

def sample_all_data(datafile, file_key, train_file, test_file, human=True):
    train = get_all_examples(datafile, file_key, human)
    with open('mcpas_all_data' + '.pickle', 'wb') as handle:
        pickle.dump(train, handle)
    # with open(str(test_file) + '.pickle', 'wb') as handle:
    #     pickle.dump(test, handle)



def sample():
    # t1 = time.time()
    # print('sampling vdjdb...')
    # sample_data('data/VDJDB_complete.tsv', 'vdjdb', 'vdjdb_train_samples', 'vdjdb_test_samples')
    # t2 = time.time()
    # print('done in ' + str(t2 - t1) + ' seconds')

    # t1 = time.time()
    # print('sampling human mcpas...')
    # sample_data('data/McPAS-TCR.csv', 'mcpas', 'mcpas_human_train_samples', 'mcpas_human_test_samples', human=True)
    # t2 = time.time()
    # print('done in ' + str(t2 - t1) + ' seconds')

    t1 = time.time()
    print('sampling mcpas...')
    sample_data('data/vdjdb_fixed.csv', 'vdjdb', 'vdjdb_train_samples', 'vdjdb_test_samples', human=False)
    t2 = time.time()
    print('done in ' + str(t2 - t1) + ' seconds')


    # t1 = time.time()
    # print('sampling human mcpas...')
    # sample_data('data/McPAS-TCR.csv', 'mcpas', 'mcpas_tuning_train_samples', 'mcpas_tuning_test_samples', human=True)
    # t2 = time.time()
    # print('done in ' + str(t2 - t1) + ' seconds')
    # t1 = time.time()

    # print('sampling vdjdb...')
    # sample_data('data/VDJDB_complete.tsv', 'vdjdb', 'vdjdb_tuning_train_samples', 'vdjdb_tuning_test_samples')
    # t2 = time.time()
    # print('done in ' + str(t2 - t1) + ' seconds')
    pass



def sample_all():
    t1 = time.time()
    print('sampling mcpas...')
    sample_all_data('data/McPAS-TCR.csv', 'mcpas', 'mcpas_train_samples', 'mcpas_test_samples', human=False)
    t2 = time.time()
    print('done in ' + str(t2 - t1) + ' seconds')
    pass


# todo sample united dataset

# Notice the different negative sampling - 5 random pairs instead of 5 random TCRs per random peptide


def get_diabetes_peptides(datafile):
    data = pd.read_csv(datafile, engine='python')
    d_peps = set()
    for index in range(len(data)):
        peptide = data['Epitope.peptide'][index]
        if pd.isna(peptide):
            continue
        pathology = data['Pathology'][index]
        if pathology == 'Diabetes Type 1':
            d_peps.add(peptide)
    return d_peps


def check():
    with open('mcpas_human_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)
    print(len(train))
    print(random.choice(train))
    pass

if __name__ == '__main__':
    sample()
    pass


