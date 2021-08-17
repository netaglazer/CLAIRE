import numpy as np
import random
import pickle
from Loader_2 import SignedPairsDataset, SinglePeptideDataset, get_index_dicts
from Trainer_2 import ERGOLightning
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from argparse import Namespace
from argparse import ArgumentParser
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import Sampler
import csv
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import sys


def load_model(hparams, checkpoint_path, diabetes=False):
    model = ERGOLightning(hparams)
    checkpoint = torch.load(checkpoint_path, map_location='cuda:1')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def load_test(datafiles):
    train_pickle, test_pickle = datafiles
    with open(train_pickle, 'rb') as handle:
        train = pickle.load(handle)
    with open(test_pickle, 'rb') as handle:
        test = pickle.load(handle)
    train_dicts = get_index_dicts(train)
    return test, train_dicts



def get_new_tcrs_and_peps(datafiles):
    train_pickle, test_pickle = datafiles
    # open and read data
    # return TCRs and peps that appear only in test pairs
    with open(train_pickle, 'rb') as handle:
        train = pickle.load(handle)
    with open(test_pickle, 'rb') as handle:
        test = pickle.load(handle)
    train_peps = [sample['peptide'] for sample in train]
    train_tcrbs = [sample['tcrb'] for sample in train]
    test_peps = [sample['peptide'] for sample in test]
    test_tcrbs = [sample['tcrb'] for sample in test]
    new_test_tcrbs = set(test_tcrbs).difference(set(train_tcrbs))
    new_test_peps = set(test_peps).difference(set(train_peps))
    # print(len(set(test_tcrbs)), len(new_test_tcrbs))
    return new_test_tcrbs, new_test_peps


def auc_predict(model, test, train_dicts, peptide=None):
    #if peptide:
    #    test_dataset = SinglePeptideDataset(test, train_dicts, peptide, force_peptide=False)
    #else:
    test_dataset = SignedPairsDataset(test, train_dicts)
    # print(test_dataset.data)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda b: test_dataset.collate(b, tcr_encoding=model.tcr_encoding_model,
                                                                  cat_encoding=model.cat_encoding))
    outputs = []
    y_hat_list = []
    i=0
    l = test_dataset.data
    for batch_idx, batch in enumerate(loader):
        output = model.validation_step(batch, batch_idx)
        if output:
            l[batch_idx]['y_hat'] = output['y_hat'].item()
            #print(l[batch_idx])
        
        
            #outputs.append(output)
            print(batch_idx)
            #print(test_dataset[i])
            #print('output', output['y_hat'].item())
            #y_hat_list.append(output['y_hat'].item())
        else:
            print('gfdfghjkjhgfdsdfghjkjhgfd')
        i = i + 1
    #print(y_hat_list)
    #print('test_dataset',test_dataset[0])
    #df_test = pd.DataFrame.from_dict(test_dataset.data)
    #print(len(y_hat_list), len(df_test))
    #df_test['y_hat'] = y_hat_list
    #l = df_test.to_dict('records')
    with open('Emerson_split_15000_tcrb_test_samples_with_y_hat_cd.pickle', 'wb') as handle:
        pickle.dump(l, handle)
        
    #auc = model.validation_end(outputs)['val_auc']
    return new_test
    

def predictions(model, test, train_dicts, peptide=None):
    if peptide:
        test_dataset = SinglePeptideDataset(test, train_dicts, peptide, force_peptide=False)
    else:
        test_dataset = SignedPairsDataset(test, train_dicts)
    # print(test_dataset.data)
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0,
                        collate_fn=lambda b: test_dataset.collate(b, tcr_encoding=model.tcr_encoding_model,
                                                                  cat_encoding=model.cat_encoding))
    outputs = []
    for batch_idx, batch in enumerate(loader):
        output = model.validation_step(batch, batch_idx)
        if output:
            outputs.append(output)
    auc = model.validation_end(outputs)['val_auc']
    return auc



def evaluations():
    checkpoint_path = 'ERGO-II_tcr/tcr_cdr/version_cd_dataset_emerson_final/checkpoints/epoch=94.ckpt'

    print('1')
    args = {'gpu': 1,
            'dataset': 'temp_cd_p', 'tcr_encoding_model': 'AE', 'cat_encoding': 'embedding',
            'use_alpha': True, 'use_vj': True, 'use_mhc': True, 'use_t_type': True,
            'aa_embedding_dim': 10, 'cat_embedding_dim': 50,
            'lstm_dim': 500, 'encoding_dim': 100,
            'lr': 0.005, 'wd': 0.005,
            'dropout': 0.2}

    # version = 15
    # weight_factor = version
    hparams = Namespace(**args)
    model = load_model(hparams, checkpoint_path, diabetes=False)
    
    datafiles = ( 'temp_cd_p_train_samples.pickle', 'Emerson_pickle_15000_split_tcrb_test_samples.pickle' ) #Emerson_split_tcrb
    test, train_dicts = load_test(datafiles)
    # model = torch.load(hparams, checkpoint_path, diabetes=True)
    # diabetes_test_set(model)
    
    #diabetes_mps(hparams, model, 'diabetes/diabetes_data/known_specificity.csv', 'diabetes/diabetes_data/28pep_pool.csv')
    auc_predict(model, test, train_dicts)
    
    
    # diabetes_mps(hparams, model, 'diabetes_data/known_specificity.csv', pep_pool=4)
    # d_peps = list(Sampler.get_diabetes_peptides('data/McPAS-TCR.csv'))
    # for pep in d_peps:
    #     try:
    #         print(pep)
    #         spb_with_more_negatives(model, datafiles, peptide=pep)
    #     except ValueError:
    #         pass
    pass

evaluations()


# if __name__ == '__main__':
#     # get model file from version
#     model_dir = 'paper_models/'
#     version = sys.argv[1]
#     path = model_dir + 'version_' + version + '/checkpoints'
#     files = [f for f in listdir(path) if isfile(join(path, f))]
#     checkpoint_path = path + '/' + files[0]
#     # get args from version
#     args_dir = 'ERGO-II_paper_logs/paper_models/'
#     path = args_dir + 'version_' + version + '/meta_tags.csv'
#     with open(path, 'r') as file:
#         lines = file.readlines()
#         args = {}
#         for line in lines[1:]:
#             key, value = line.strip().split(',')
#             if key in ['dataset', 'tcr_encoding_model', 'cat_encoding']:
#                 args[key] = value
#             else:
#                 args[key] = eval(value)
#     hparams = Namespace(**args)
#     checkpoint = checkpoint_path
#     model = load_model(hparams, checkpoint, diabetes=False)
#     train_pickle = 'Samples/' + model.dataset + '_train_samples.pickle'
#     test_pickle = 'Samples/' + model.dataset + '_test_samples.pickle'
#     datafiles = train_pickle, test_pickle
#
#     # check auc of unfiltered spb (faster but maybe less accurate)
#     # print('LPRRSGAAGA u spb:', unfiltered_spb(model, datafiles, peptide='LPRRSGAAGA'))
#     # print('GILGFVFTL u spb:', unfiltered_spb(model, datafiles, peptide='GILGFVFTL'))
#     # print('NLVPMVATV u spb:', unfiltered_spb(model, datafiles, peptide='NLVPMVATV'))
#     # print('GLCTLVAML u spb:', unfiltered_spb(model, datafiles, peptide='GLCTLVAML'))
#     # print('SSYRRPVGI u spb:', unfiltered_spb(model, datafiles, peptide='SSYRRPVGI'))
#
#     print('KLGGALQAK u spb:', unfiltered_spb(model, datafiles, peptide='KLGGALQAK'))
#     print('GILGFVFTL u spb:', unfiltered_spb(model, datafiles, peptide='GILGFVFTL'))
#     print('NLVPMVATV u spb:', unfiltered_spb(model, datafiles, peptide='NLVPMVATV'))
#     print('AVFDRKSDAK u spb:', unfiltered_spb(model, datafiles, peptide='AVFDRKSDAK'))
#     print('RAKFKQLL u spb:', unfiltered_spb(model, datafiles, peptide='RAKFKQLL'))
#
#     # exit()
#     true_test = true_new_pairs(hparams, datafiles)
#     # TPP
#     # print('tpp i:', tpp_i(model, datafiles, true_test))
#     # print('tpp ii:', tpp_ii(model, datafiles, true_test))
#     # print('tpp iii:', tpp_iii(model, datafiles, true_test))
#     # McPAS SPB
#     # print('LPRRSGAAGA spb:', spb(model, datafiles, true_test, peptide='LPRRSGAAGA'))
#     # print('GILGFVFTL spb:', spb(model, datafiles, true_test, peptide='GILGFVFTL'))
#     # print('NLVPMVATV spb:', spb(model, datafiles, true_test, peptide='NLVPMVATV'))
#     # print('GLCTLVAML spb:', spb(model, datafiles, true_test, peptide='GLCTLVAML'))
#     # print('SSYRRPVGI spb:', spb(model, datafiles, true_test, peptide='SSYRRPVGI'))
#     pass
