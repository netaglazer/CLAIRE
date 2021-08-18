import pandas as pd
import numpy as np
import random
import pickle
import time
import bisect
import collections
import time
from sklearn.model_selection import train_test_split as tts
    
def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result


def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]
    
def create_neg_hla(s):
    #s = s.drop(labels = [mhc])
    index = s.index
    values = s.values
    values = values/sum(values)
    weights = values
    population = index
    for i in range(1):
        temp_mhc = choice(population, weights) 
    return temp_mhc
    

def read_data(datafile, file_key, human=True):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])

    def invalid_hla(seq):
        hla = [letter for letter in 'HLA-BC*:0123456789']
        if( pd.isna(seq) or any([l not in hla for l in seq])):
            return 'invalid'
        return seq

    if file_key == 'mcpas':
        data = pd.read_csv(datafile, engine='python')
        data['MHC'] = data['MHC'].apply(lambda x: invalid_hla(x))
        data = data[data['MHC'] != 'invalid']
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
            # sample['peptide'] = data['Epitope.peptide'][index]
            # sample['protein'] = data['Antigen.protein'][index]
            sample['mhc'] = data['MHC'][index]
            if invalid(sample['tcrb']):
                continue
            if human and data['Species'][index] != 'Human':
                continue
            if invalid(sample['tcra']):
                sample['tcra'] = 'UNK'
            all_pairs.append(sample)
    # assimung each sample appears only once in the dataset
    df = pd.DataFrame.from_dict(all_pairs)
    df = df.drop_duplicates()
    tcrb_list = df['tcrb'].unique().tolist()
    #all_pairs = df.to_dict('records')
    
    
    
    train_tcrb, test_tcrb = train_test_split(tcrb_list)
    
    
    
    df_train = df[df['tcrb'].isin(train_tcrb)]
    df_test = df[df['tcrb'].isin(test_tcrb)]
    #all_pairs = df.to_dict('records')
    #train_pairs, test_pairs = train_test_split(all_pairs)
    test_pairs = df_test.to_dict('records')
    train_pairs = df_train.to_dict('records')
    all_pairs_df = pd.concat([df_train,df_test], axis = 0)
    all_pairs = all_pairs_df.to_dict('records')
    
    s = df_train.mhc.value_counts()
    return all_pairs, train_pairs, test_pairs, s



def train_test_split(all_pairs):
    '''
    Splitting the TCR-PEP pairs
    '''
    train_pairs = []
    test_pairs = []
    for pair in all_pairs:
        # 80% train, 20% test
        p = np.random.binomial(1,0.85)
        if p == 1:
            train_pairs.append(pair)
        else:
            test_pairs.append(pair)
    return train_pairs, test_pairs


def positive_examples(pairs):
    pos_samples = []
    for sample in pairs:
        temp_sample = sample.copy()
        temp_sample['sign'] = 1
        pos_samples.append(temp_sample)
    return pos_samples

# Removing this function - assuming every (tcrb,pep) pair appears only once in a dataset
# def is_negative(all_pairs, tcrb, pep):
#     for sample in all_pairs:
#         # we do not check for full sample match, this is enough
#         if sample['tcrb'] == tcrb and sample['peptide'] == pep:
#             return False
#     return True


def negative_examples(pairs, all_pairs, size, s):
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
        #mhc_sample = random.choice(pairs)
        mhc_sample = create_neg_hla(s)
        tcr_sample = random.choice(pairs)
        sample = {}
        sample['tcra'] = tcr_sample['tcra']
        sample['tcrb'] = tcr_sample['tcrb']
        sample['va'] = tcr_sample['va']
        sample['ja'] = tcr_sample['ja']
        sample['vb'] = tcr_sample['vb']
        sample['jb'] = tcr_sample['jb']
        sample['t_cell_type'] = tcr_sample['t_cell_type']
        sample['mhc'] = mhc_sample #mhc_sample['mhc']
        if sample not in all_pairs and sample not in neg_samples:
                sample['sign'] = 0
                neg_samples.append(sample)
                i += 1
    return neg_samples


def get_examples(datafile, file_key, human):
    all_pairs, train_pairs, test_pairs,s = read_data(datafile, file_key, human)
    train_pos = positive_examples(train_pairs)
    test_pos = positive_examples(test_pairs)
    train_neg = negative_examples(train_pairs, all_pairs, len(train_pos),s)
    test_neg = negative_examples(test_pairs, all_pairs, len(test_pos),s)
    train = train_pos + train_neg
    random.shuffle(train)
    test = test_pos + test_neg
    random.shuffle(test)
    return train, test


def sample_data(datafile, file_key, train_file, test_file,validation_file, human=True):
    train, test = get_examples(datafile, file_key, human)
    df_train = pd.DataFrame(train, columns=['tcra', 'tcrb', 'va', 'ja', 'vb', 'jb', 't_cell_type', 'mhc', 'sign'])
    df_test = pd.DataFrame(test, columns=['tcra', 'tcrb', 'va', 'ja', 'vb', 'jb', 't_cell_type', 'mhc', 'sign'])
    print(df_test['sign'].value_counts())
    print(df_train['sign'].value_counts())
    df_validation, df_test = tts(df_test, test_size=0.9)
    validation = df_validation.to_dict('records')
    test = df_test.to_dict('records')
    with open(str(train_file) + '.pickle', 'wb') as handle:
        pickle.dump(train, handle)
    with open(str(test_file) + '.pickle', 'wb') as handle:
        pickle.dump(test, handle)
    with open(str(validation_file) + '.pickle', 'wb') as handle:
        pickle.dump(validation, handle)



def sample():
    t1 = time.time()
    print('sampling mcpas...')
    sample_data('data/preprocessed_McPAS (2).csv', 'mcpas', 'final_mcpas_neg_mhc_2206_train_samples', 'final_mcpas_neg_mhc_2206_test_samples','final_mcpas_neg_mhc_2206_validation_samples', human=False)
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


