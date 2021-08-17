import pandas as pd
import numpy as np
import random
import pickle


def read_data(datafile, file_key):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    data = pd.read_csv(datafile, engine='python')
    CD_list = ['CD8', 'CD4']
    data = data[data['T.Cell.Type'].isin(CD_list)]
    data = data.replace({'CD8': 0, 'CD4': 1})
    data = data.reset_index(drop=True)

    for index in range(len(data)):
        # tcra = data['CDR3.alpha.aa'][index]
        tcrb = data['CDR3.beta.aa'][index]
        # v = data['TRBV'][index]
        # j = data['TRBJ'][index]
        # peptide = data['Epitope.peptide'][index]
        # protein = data['Antigen.protein'][index]
        # mhc = data['MHC'][index]
        cdr = data['T.Cell.Type'][index]
        if invalid(tcrb):
            continue
        # if invalid(tcra):
        #     tcra = 'UNK'
        tcr_data = (tcrb)
        cdr_data = (cdr)
        all_pairs.append((tcr_data, cdr_data))
    train_pairs, test_pairs = train_test_split(set(all_pairs))
    return all_pairs, train_pairs, test_pairs


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
    examples = []
    for pair in pairs:
        tcr_data, mhc_data = pair
        examples.append((tcr_data, mhc_data, 1))
    return examples


def is_negative(all_pairs, tcrb, mhc):
    for pair in all_pairs:
        tcr_data, mhc_data = pair
        if tcr_data == tcrb and mhc_data == mhc:
            return False
    return True


def negative_examples(pairs, all_pairs, size):
    '''
    Randomly creating intentional negative examples from the same pairs dataset.
    '''
    examples = []
    i = 0
    # Get tcr and peps lists
    tcrs = [tcr_data for (tcr_data, pep_data) in pairs]
    mhcs = [mhc_data for (tcr_data, mhc_data) in pairs]
    while i < size:
        # for j in range(5):
        mhc_data = random.choice(mhcs)
        tcr_data = random.choice(tcrs)
        if is_negative(all_pairs, tcr_data, mhc_data) and \
                (tcr_data, mhc_data, 0) not in examples:
                examples.append((tcr_data, mhc_data, 0))
                i += 1
    return examples


def get_examples(datafile, file_key):
    all_pairs, train_pairs, test_pairs = read_data(datafile, file_key)
    # train_pos = positive_examples(train_pairs)
    # test_pos = positive_examples(test_pairs)
    # train_neg = negative_examples(train_pairs, all_pairs, 5 * len(train_pos))
    # test_neg = negative_examples(test_pairs, all_pairs, 5 * len(test_pos))
    # train = train_pos + train_neg
    # random.shuffle(train)
    # test = test_pos + test_neg
    # random.shuffle(test)
    return train_pairs, test_pairs


def sample_data(datafile, file_key, train_file, test_file):
    train, test = get_examples(datafile, file_key)
    df_train = pd.DataFrame(train, columns=['TCR', 'Sign'])
    df_test = pd.DataFrame(test, columns=['TCR', 'Sign'])
    df_train.to_csv('mcpas_train_tcr_cdr.csv')
    df_test.to_csv('mcpas_test_tcr_cdr.csv')
    # with open(str(train_file) + '.pickle', 'wb') as handle:
    #     pickle.dump(train, handle)
    # with open(str(test_file) + '.pickle', 'wb') as handle:
    #     pickle.dump(test, handle)


# sample_data('data/McPAS-TCR_fixed.csv', 'mcpas', 'mcpas_train_tcr_cdr', 'mcpas_test_tcr_cdr')
# sample_data('data/VDJDB_complete.tsv', 'vdjdb', 'vdjdb_train_samples', 'vdjdb_test_samples')

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

