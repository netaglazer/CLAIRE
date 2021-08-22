import pandas as pd
import numpy as np
import random
import pickle
from Loader_2 import SignedPairsDataset, SinglePeptideDataset, get_index_dicts 
from pre_trainer_2_MCPAS import ERGOLightning
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from argparse import Namespace
from argparse import ArgumentParser
import torch
#import Sampler


# todo all tests suggested in ERGO
#  TPP-I
#  TPP-II
#  TPP-III
# SPB       V
#  Protein SPB
#  MPS
# all test today
# then we could check a trained model and compare tests to first ERGO paper
# todo for SPB, check alpha+beta/beta ratio in data for the peptides



def load_model(hparams, checkpoint_path):
    # args = {'dataset': 'mcpas', 'tcr_encoding_model': 'LSTM', 'use_alpha': False,
    #         'embedding_dim': 10, 'lstm_dim': 500, 'encoding_dim': 'none', 'dropout': 0.1}
    # hparams = Namespace(**args)
    # model = ERGOLightning(hparams)
    # model.load_from_checkpoint('checkpoint_trial/version_4/checkpoints/_ckpt_epoch_27.ckpt')
    model = ERGOLightning(hparams)
    checkpoint = torch.load(checkpoint_path, map_location='cuda:1')
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_from_checkpoint('checkpoint')
    return model





# chack diabetes with different weight factor
# checkpoint_path = 'mcpas_without_alpha/version_8/checkpoints/_ckpt_epoch_35.ckpt'
# checkpoint_path = 'mcpas_without_alpha/version_5/checkpoints/_ckpt_epoch_40.ckpt'
# checkpoint_path = 'mcpas_without_alpha/version_10/checkpoints/_ckpt_epoch_46.ckpt'
# checkpoint_path = 'mcpas_without_alpha/version_20/checkpoints/_ckpt_epoch_63.ckpt'
# with alpha
# checkpoint_path = 'mcpas_with_alpha/version_2/checkpoints/_ckpt_epoch_31.ckpt'
# check2(checkpoint_path)



def Merge(dict1, dict2):
    dict2.update(dict1)
    return((dict2))

def hla_test_set(hparams, model):
    # 8 paired samples, 4 peptides
    # tcra, tcrb, pep

    samples = []
    
    #with open('preproc_mcpas_validation_0707_v.pickle', 'rb') as handle:
    #    test = pickle.load(handle)
    
    with open('preproc_mcpas_train_0707.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
            test = pickle.load(handle)

    #df_test = pd.DataFrame.from_dict(test)
    #del test
    #df_test.to_pickle('test_head_y_hat_cd_validation.pickle')
    #df_test = df_test.sort_values(by=['y_hat'])
    #print(int(0.1*len(df_test)))
    #df_test = df_test.head(int(0.01*len(df_test)))
    
    ############################################################
    #hlas = df_test.mhc.unique()
    #positive_df = df_test[df_test['sign'] == 1]
    #positive_hla = positive_df.mhc.value_counts()
    #negative_df = df_test[df_test['sign'] == 0]
    #negative_hla = negative_df.mhc.value_counts()
    #print(positive_hla)
    #print(negative_hla)
    
    #min_dict = {}
    #for i in hlas:
    #    if  i in list(positive_hla.index) and i in list(negative_hla.index):
    #        m = min(positive_hla[i],negative_hla[i])
    #        min_dict[i] = (min(positive_hla[i],negative_hla[i]))
        
    #dfs = []
    #print('min_dict.', min_dict.keys())
    #for i in list(min_dict.keys()):
        #print(i)
        #if(i == 'HLA-A*03' or i == 'HLA-B*07'):
    #    temp_neg = negative_df[negative_df['mhc'] == i]
    #    temp_neg = temp_neg.head(min_dict[i])# + int((len(temp_neg)-min_dict[i])/15))
    #    temp_pos = positive_df[positive_df['mhc'] == i]
    #    temp_pos = temp_pos.head(min_dict[i])# + int((len(temp_pos)-min_dict[i])/15))
    #    print(len(temp_pos), len(temp_neg))
    #    temp_df = pd.concat([temp_neg, temp_pos], axis = 0)
    #    #else:
            #temp_df = df_test[df_test['mhc'] == i]
    #    dfs.append(temp_df)
    #df_test = pd.concat(dfs).sample(frac=1)
    
    print('load_test')


    ###########################################################################################
    
    #with open('final_4_LE_split_new_counts_tcrb_mcpas_train_samples_cv.pickle', 'rb') as handle:
    #    train = pickle.load(handle)
        
    with open('preproc_mcpas_train_0707.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
            train = pickle.load(handle)
 
    #df_train = pd.DataFrame.from_dict(train)
    #del train
    #df_train.to_pickle('train_head_y_hat_cd.pickle')
    #df_train = df_train.sort_values(by=['y_hat'])
    #print(df_train.head(20)['y_hat'])
    #df_train = df_train.sample(int(0.05*len(df_train)))
    #df_train = df_train.head(int(0.01*len(df_train)))
    #print(df_train.columns)
    
    #hlas = df_train.mhc.unique()
    #positive_df = df_train[df_train['sign'] == 1]
    #positive_hla = positive_df.mhc.value_counts()
    #negative_df = df_train[df_train['sign'] == 0]
    #negative_hla = negative_df.mhc.value_counts()
    #print(positive_hla)
    #min_dict = {}
    #for i in hlas:
    #    if  i in list(positive_hla.index) and i in list(negative_hla.index):
    #        m = min(positive_hla[i],negative_hla[i])
    #        min_dict[i] = m
        
    #dfs = []
    #for i in list(min_dict.keys()):
    #    temp_neg = negative_df[negative_df['mhc'] == i]
    #    temp_neg = temp_neg.head(min_dict[i])# + int((len(temp_neg)-min_dict[i])/11))
    #    temp_pos = positive_df[positive_df['mhc'] == i]
    #    temp_pos = temp_pos.head(min_dict[i])# + int((len(temp_pos)-min_dict[i])/11))
    #    temp_df = pd.concat([temp_neg, temp_pos], axis = 0)
    #    print(len(temp_pos), len(temp_neg), len(temp_df))
    #    dfs.append(temp_df)
    
    #df_train = pd.concat(dfs, axis = 0).sample(frac=1)

    #df_train.to_pickle('train_head_y_hat_cd_15000.pickle')
    #print('train_ti pickle')
    #train = df_train.to_dict('records')
    

    test_dataset = SignedPairsDataset(test, get_index_dicts(train))
    for i in test_dataset.data:
        print(i)
        break;
    loader =  DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                      collate_fn=lambda b: test_dataset.collate(b, tcr_encoding= hparams.tcr_encoding_model,
                                                                 cat_encoding=hparams.cat_encoding))
    print('lennnnnnnn', len(test_dataset.data))
    outputs = []
    dict_list = []
    for batch_idx, batch in enumerate(loader):
        print('batch_idx', batch_idx)
        outputs.append(model.validation_step(batch, batch_idx))
        #print(model.validation_step(batch, batch_idx))
        #print(test_dataset.data[batch_idx])
        #print('\n')
        #print(outputs[-1])
        #print(test_dataset.data[batch_idx]) 
        temp_dict = Merge(outputs[-1], test_dataset.data[batch_idx])
        #print(temp_dict)
        temp_dict['y'] = int(temp_dict['y'].data) 
        temp_dict['y_hat'] = float(temp_dict['y_hat'].data)
        temp_dict['val_loss'] = float(temp_dict['val_loss'].data)  
        #print(temp_dict)
        ##
        #print(temp_dict)
        dict_list.append(temp_dict)
    df = pd.DataFrame.from_dict(dict_list)
    df = df.sort_values(by = ['tcrb'])
    
    print(df[['tcrb', 'y', 'y_hat']].head(20))
    
    df.to_csv('validation_mcpas_evaluation_df.csv')
    print('saved csv')
    
    pass


if __name__ == '__main__':
    #checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA/TCR_HLA_logs/tcr_hla_binding/version_2211/checkpoints/epoch=146.ckpt"
    #checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA/TCR_HLA_logs/tcr_hla_binding/version_1844/checkpoints/epoch=15.ckpt"
    #checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA/TCR_HLA_logs/tcr_hla_binding/version_tcrb_split/checkpoints/epoch=3.ckpt"
    #checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA/TCR_HLA_logs/tcr_hla_binding/version_tcrb_split_4/checkpoints/epoch=3.ckpt"
    #checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA/TCR_HLA_logs/tcr_hla_binding/version_cd_p_filter_5/checkpoints/epoch=94.ckpt" #500 
    #checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA/TCR_HLA_logs/tcr_hla_binding/version_emerson_15000_1/checkpoints/epoch=120.ckpt"
    checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA_simulated_emerson/pre_trainer_logs/tcr_hla_binding/version_29_04/checkpoints/epoch=3.ckpt"
    checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA_simulated_emerson/pre_trainer_logs/tcr_hla_binding/version_2021-04-29 22:56:16.158127/checkpoints/epoch=20.ckpt"
    checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA_simulated_emerson/pre_trainer_logs/tcr_hla_binding/version_2021-04-29 23:01:05.389886/checkpoints/epoch=131.ckpt"
    checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA_simulated_emerson/pre_trainer_logs/tcr_hla_binding/version_123456/checkpoints/epoch=16.ckpt"
    checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA_simulated_emerson/pre_trainer_logs/tcr_hla_binding/version_123456/checkpoints/epoch=16.ckpt"
    checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA_simulated_emerson/pre_trainer_logs/tcr_hla_binding/version_0307/checkpoints/epoch=6.ckpt"
    #{'lr': 0.001, 'dropout': 0.2, 'l2': 0.0001, 'encoding_dim' : 100}
    
    #validation
    checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA_simulated_emerson/pre_trainer_logs/tcr_hla_binding/version_mcpas_0707/checkpoints/epoch=40.ckpt"
    
    checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA_simulated_emerson/pre_trainer_logs/tcr_hla_binding/version_mcpas_0707/checkpoints/epoch=9.ckpt"
    
    
    
    args = {'dataset': 'final_hla_mcpas', 'tcr_encoding_model': 'AE', 'cat_encoding':'embedding',
    'use_alpha': True, 'use_vj': True, 'use_mhc': True, 'use_t_type': True, 'aa_embedding_dim':10,
                 'cat_embedding_dim': 50, 'embedding_dim': 10, 'lstm_dim': 500,'weight_factor':1,
                 'lr': 0.001, 'dropout': 0.2, 'wd': 0.0001, 'encoding_dim' : 100, 'stage': 1}

 
    hparams = Namespace(**args)
    checkpoint = checkpoint_path
    model = load_model(hparams, checkpoint)
    hla_test_set(hparams, model)
    pass

# it should be easy because the datasets are fixed and the model is saved in a lightning checkpoint
# tests might be implemented in lightning module

#
# args = {'dataset': 'mcpas', 'tcr_encoding_model': 'LSTM', 'use_alpha': False,
#            'embedding_dim': 10, 'lstm_dim': 500, 'encoding_dim': 'none', 'dropout': 0.1}
# hparams = Namespace(**args)
# model = ERGOLightning(hparams)
# logger = TensorBoardLogger("trial_logs", name="checkpoint_trial")
# early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
# trainer = Trainer(gpus=[2], logger=logger, early_stop_callback=early_stop_callback)
# trainer.fit(model)