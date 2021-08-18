import pandas as pd
import numpy as np
import random
import pickle
from Loader_val import SignedPairsDataset, SinglePeptideDataset, get_index_dicts
import sklearn.metrics as metrics 
from Trainer_2 import ERGOLightning
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from argparse import Namespace
from argparse import ArgumentParser
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score

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
    


def Merge(dict1, dict2):
    dict2.update(dict1)
    return((dict2))
    
    
def hla_test_set(hparams, model):

    samples = []
    
    with open(hparams.dataset + '_validation_samples.pickle', 'rb') as handle:
        validation = pickle.load(handle)
   
    with open(hparams.dataset + '_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)


    test_dataset = SignedPairsDataset(validation, get_index_dicts(train))
    for i in test_dataset.data:
        #print(i)
        break;
    loader =  DataLoader(validation, batch_size=1, shuffle=False, num_workers=0,
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
        dict_list.append(temp_dict)
    df = pd.DataFrame.from_dict(dict_list)
    acc = accuracy_score(df.y, df.y_hat.round())
    print('acc=', acc)
    auc = roc_auc_score(df.y, df.y_hat)
    print(auc)
    fpr, tpr, threshold = metrics.roc_curve(df.y, df.y_hat)
    roc_auc = metrics.auc(fpr, tpr)
        
    # method I: plt
    #plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'black')
    #plt.plot([], label = 'AUC = %0.2f' % roc_auc)
    #plt.plot([], label = 'accuracy = %0.2f' % acc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],color = 'black', linestyle='dashed')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.savefig('roc_curves/{0}_{1}_{2}.png'.format(hparams.dataset, auc,acc))
    plt.clf()
    pass





if __name__ == '__main__':
    #checkpoint_path = "/home/dsi/netaglazer/Projects/ERGO_TCR_HLA/TCR_HLA_logs/tcr_hla_binding/version_emerson_15000_1/checkpoints/epoch=120.ckpt" 
    #this is the final checkpoint for mcpas data
    #checkpoint_path = "/home/dsi/netaglazer/Projects/TCR_MHC_ERGO_tcr_hla_binding/TCR_MHC_new/tcr_mhc_binding/version_TCR_HLA_no_cv_18-04/checkpoints/epoch=63.ckpt"    
    #this is the final checkpoint for vdjdb data
    checkpoint_path = "/home/dsi/netaglazer/Projects/TCR_MHC_ERGO_tcr_hla_binding/TCR_MHC_new/tcr_mhc_binding/version_TCR_HLA_vdjdb_19-04/checkpoints/epoch=43.ckpt"
    
    
    #{'lr': 0.001, 'dropout': 0.2, 'l2': 0.0001, 'encoding_dim' : 100}
    
    args =     {'dataset': 'vdjdb_neg_mhc', 'tcr_encoding_model': 'AE', 'cat_encoding':'embedding',
                 'use_alpha': True, 'use_vj': True, 'use_mhc': True, 'use_t_type': True, 'aa_embedding_dim':10,
                 'cat_embedding_dim': 50, 'embedding_dim': 10, 'lstm_dim': 500,'weight_factor':1,
                 'lr': 0.001, 'dropout': 0.3, 'wd': 0.0005, 'encoding_dim' : 100}

 
    hparams = Namespace(**args)
    checkpoint = checkpoint_path
    model = load_model(hparams, checkpoint)
    hla_test_set(hparams, model)
    pass
