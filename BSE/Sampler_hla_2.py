import pandas as pd
import numpy as np
import random
import pickle
import pandas as pd
import random
import bisect
import collections
import time


def fix_hla(hla):
    hla = hla.split(':')[0]
    splited = (hla.split('*'))
    if len(splited) == 2:
        first = hla.split('*')[0]
        last = hla.split('*')[1]
        if len(last) == 2:
            return first+'*'+last
        if len(last) == 1:
            return first+'*0'+last
    f = hla[:5]
    l = hla[5:]
    if(len(l) == 1):
        return f+'*0'+l
    elif(len(l)==2):
        return f+'*'+l
    elif(hla == 'HLA-A*011'):
        return 'HLA-A*11'
    return hla

def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res
    
    
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
    
    
def create_new_hla(mhc, s, hla_amount):
    s = s.drop(labels = [mhc])
    index = s.index
    values = s.values
    values = values/sum(values)
    weights = values
    population = index
    l = [mhc]
    for i in range(hla_amount-1):
        temp_mhc = choice(population, weights) 
        l.append(temp_mhc)
        s = s.drop(labels = [temp_mhc])
        index = s.index
        values = s.values
        values = values/sum(values)
        weights = values
        population = index
    #print(l)
    return l
    
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
    

def read_data(datafile, file_key, hla_amount, human=True):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        bool = pd.isna(seq)
        return bool or any([aa not in amino_acids for aa in seq])

    def invalid_hla(seq):
        hla = [letter for letter in 'HLA-BC*:0123456789']
        if( pd.isna(seq) or any([l not in hla for l in seq])):
            return 'invalid'
        return seq

    if file_key == 'mcpas':
        print('datafile', datafile)
        data = pd.read_csv(datafile)
        
        data['MHC'] = data['MHC'].apply(lambda x: invalid_hla(x))
        #print('len', len(data))
        data = data[data['MHC'] != 'invalid']
        #print('len', len(data))

         
        data['MHC'] = data['MHC'].apply(lambda x: fix_hla(x))
        
        value_counts = data['MHC'].value_counts()
        value_counts = value_counts.sort_values( ascending=False)
        hlas = value_counts.index.tolist()
       
        first_list = data[data['MHC'] == hlas[0]].index.tolist()
        second_list = data[data['MHC'] == hlas[1]].index.tolist()
        n = value_counts[hlas[2]]
        first_list_to_remove = random.sample(first_list, int(1.5*n))
        second_list_to_remove = random.sample(second_list, int(1.2*n))
        
        data = data.drop(first_list_to_remove+second_list_to_remove)
        
        #hla_a2_l = data[data['MHC'] == 'HLA-A*02'].index.tolist()
        #hla_b7_l = data[data['MHC'] == 'HLA-B*07'].index.tolist()
        #list_to_remove = random.sample(hla_a2_l, 6000)
        #list_to_remove_2 = random.sample(hla_b7_l, 2000)
        #data = data.drop(list_to_remove+list_to_remove_2)
        data = data.reset_index(drop=True)
        #print('unique', data['MHC'].unique())
        s = data.MHC.value_counts()
        print(s)
        data['new_MHC'] = data['MHC'].apply(lambda x: create_new_hla(x, s,4))
        print(data['new_MHC'][0])
        data = explode(data, ['new_MHC'], fill_value='', preserve_index=True)
        data = data.reset_index(drop=True)

        for index in range(len(data)):
            sample = {}
            sample['tcra'] = data['CDR3.alpha.aa'][index]
            sample['tcrb'] = data['CDR3.beta.aa'][index]
            sample['va'] = data['va'][index]
            sample['ja'] = data['TRA'][index]
            sample['vb'] = data['TRBV'][index]
            sample['jb'] = data['TRBJ'][index]
            sample['t_cell_type'] = data['T.Cell.Type'][index]
            # sample['peptide'] = data['Epitope.peptide'][index]
            # sample['protein'] = data['Antigen.protein'][index]
            sample['mhc'] = data['new_MHC'][index]
            #sample['original_mhc'] = data['MHC'][index]
            if invalid(sample['tcrb']):
                continue
            #if human and data['Species'][index] != 'Human':
            #    continue
            if invalid(sample['tcra']):
                sample['tcra'] = 'UNK'
            all_pairs.append(sample)
            
    if file_key == 'new_dataset':
        print('datafile', datafile)
        data = pd.read_pickle(datafile)
        print(data.mhc.value_counts())
        data = explode(data, ['mhc'], fill_value='', preserve_index=True)
        data = data.reset_index(drop=True)
        data['MHC'] = data['mhc'].apply(lambda x: invalid_hla(x))
        #print('len', len(data))
        print(data.MHC.value_counts())
        print(data.shape)
        data = data[data['MHC'] != 'invalid']
        print(data.shape)
        print(data.columns)
        #print('len', len(data))
        
         
        #data['MHC'] = data['MHC'].apply(lambda x: fix_hla(x))
        
        value_counts = data['MHC'].value_counts()
        value_counts = value_counts.sort_values( ascending=False)
        hlas = value_counts.index.tolist()
        
        #first_list = data[data['MHC'] == hlas[0]].index.tolist()
        #second_list = data[data['MHC'] == hlas[1]].index.tolist()
        #n = value_counts[hlas[2]]
        #first_list_to_remove = random.sample(first_list, int(1.5*n))
        #second_list_to_remove = random.sample(second_list, int(1.2*n))
        
        #data = data.drop(first_list_to_remove+second_list_to_remove)
        
        #hla_a2_l = data[data['MHC'] == 'HLA-A*02'].index.tolist()
        #hla_b7_l = data[data['MHC'] == 'HLA-B*07'].index.tolist()
        #list_to_remove = random.sample(hla_a2_l, 6000)
        #list_to_remove_2 = random.sample(hla_b7_l, 2000)
        #data = data.drop(list_to_remove+list_to_remove_2)
        data = data.reset_index(drop=True)
        #print('unique', data['MHC'].unique())
        s = data.MHC.value_counts()
        print(s)
        #data['new_MHC'] = data['MHC'].apply(lambda x: create_new_hla(x, s,4))
        #print(data['new_MHC'][0])
        
        
        for index in range(len(data)):
            sample = {}
            sample['tcra'] = 'UNK' #data['CDR3.alpha.aa'][index]
            sample['tcrb'] = data['tcrb'][index]
            sample['va'] = 'UNK' #data['TRAV'][index]
            sample['ja'] = 'UNK' #data['TRAJ'][index]
            sample['vb'] = data['vb'][index]
            sample['jb'] = data['jb'][index]
            sample['t_cell_type'] = data['t_cell_type'][index]
            # sample['peptide'] = data['Epitope.peptide'][index]
            # sample['protein'] = data['Antigen.protein'][index]
            sample['mhc'] = data['MHC'][index]
            #sample['original_mhc'] = data['MHC'][index]
            if invalid(sample['tcrb']):
                continue
            #if human and data['Species'][index] != 'Human':
            #    continue
            if invalid(sample['tcra']):
                sample['tcra'] = 'UNK'
            all_pairs.append(sample)
    print('done for loop')

    # assimung each sample appears only once in the dataset
    df = pd.DataFrame.from_dict(all_pairs)
    df = df.drop_duplicates()
    tcrb_list = df['tcrb'].unique().tolist()
    
    #all_pairs = df.to_dict('records')
    train_tcrb, test_tcrb = train_test_split(tcrb_list)
    
    df_train = df[df['tcrb'].isin(train_tcrb)]
    df_test = df[df['tcrb'].isin(test_tcrb)]
    
    #train_pairs, test_pairs = train_test_split(all_pairs)
    
    #print('train_pairs,  ', train_pairs[0])
    #print('train_pairs, ', train_pairs[0])
    print('*****')
    #df_train = pd.DataFrame.from_dict(train_pairs)
    s = df_train.mhc.value_counts()
    #df_test = pd.DataFrame.from_dict(test_pairs)
    #df_train = df_train.drop(columns = ['original_mhc'])
    
    #df_test = df_test.drop(columns = ['mhc'])
    #df_test =  df_test.rename(columns={"original_mhc": "mhc"})
    #df_test =  df_test.drop_duplicates().reset_index(drop = True)
    test_pairs = df_test.to_dict('records')
    print('*/*/*/*/')
    train_pairs = df_train.to_dict('records')
    all_pairs_df = pd.concat([df_train,df_test], axis = 0)
    all_pairs = all_pairs_df.to_dict('records')
    #print('test_pairs,  ', test_pairs[0])
    print('kjhgfghjkl')
    return all_pairs, train_pairs, test_pairs, s


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
    #print('all pairs, ', all_pairs[0])
    #print('pairs,  ', pairs[0])
    '''
    Randomly creating intentional negative examples from the same pairs dataset.
    We match randomly tcr data with peptide data to make a sample
    '''
    #df = pd.DataFrame.from_dict(all_pairs)
    #s = df.mhc.value_counts()
    neg_samples = []
    i = 0
    # tcrs = [tcr_data for (tcr_data, pep_data) in pairs]
    # peps = [pep_data for (tcr_data, pep_data) in pairs]
    print(size)
    while i < size:
        # choose randomly two samples. match tcr data with pep data
        mhc_sample = create_neg_hla(s)
        #print('mhc_sample_ssssss:  ', mhc_sample)
        #mhc_sample = random.choice(pairs)
        #print('mhc_sample: ', mhc_sample)
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
        #print('neg sample  ', sample)
        #print('all pairs, ', all_pairs[0])
        #print('pairs,  ', pairs[0])
        #print('sample: ', sample)
        #print('all_pairs: ', all_pairs[0])
        if sample not in all_pairs + neg_samples:
            sample['sign'] = 0
            neg_samples.append(sample)
            if len(neg_samples)%10 == 0:
                print(len(neg_samples))
            i += 1
    return neg_samples


def get_examples(datafile, file_key, human):
    
    hla_amount = 4
    
    all_pairs, train_pairs, test_pairs, s = read_data(datafile, file_key, hla_amount, human)
    #print('all pairs, ', all_pairs[0])
    pos_samples = []
    train_pos = positive_examples(train_pairs)
    print('train_pos')
    test_pos = positive_examples(test_pairs)
    print('test_pos')
    #print('all pairs, ', all_pairs[0])
    train_neg = negative_examples(train_pairs, all_pairs, len(train_pos), s)
    print('train_neg')
    test_neg = negative_examples(test_pairs, all_pairs, len(test_pos), s)
    print('test_neg')
    #print('len(train_pos)!!!!!!! :  ', len(train_neg))
    train = train_pos + train_neg
    random.shuffle(train)
    test = test_pos + test_neg
    random.shuffle(test)
    return train, test
    
    

def sample_data(datafile, file_key, train_file, test_file, human=True):
    train, test = get_examples(datafile, file_key, human)
    df_train = pd.DataFrame(train, columns=['tcra', 'tcrb', 'va', 'ja', 'vb', 'jb', 't_cell_type', 'mhc', 'sign'])
    df_test = pd.DataFrame(test, columns=['tcra', 'tcrb', 'va', 'ja', 'vb', 'jb', 't_cell_type', 'mhc', 'sign'])
    print(df_test['sign'].value_counts())
    print(df_train['sign'].value_counts())
    with open(str(train_file) + '.pickle', 'wb') as handle:
        pickle.dump(train, handle)
    with open(str(test_file) + '.pickle', 'wb') as handle:
        pickle.dump(test, handle)



def sample():
    t1 = time.time()
    print('sampling new mcpas...')
    sample_data('new_dataset_for_emerson.pickle', 'new_dataset', 'new_dataset_train_samples', 'new_dataset_test_samples', human=False)
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
    print('**************')
    sample()
    pass


