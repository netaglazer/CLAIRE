import pickle
import random
import nni
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from datetime import datetime
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from Loader_2 import SignedPairsDataset, DiabetesDataset, get_index_dicts #, SinglePeptideDatasetWeighted
from Models import PaddingAutoencoder, AE_Encoder, LSTM_Encoder, ERGO, CNN_Encoder
from Models_mcpas import PaddingAutoencoder, AE_Encoder, LSTM_Encoder, CNN_Encoder, ERGO
import logging
from random import randrange

_logger = logging.getLogger("nni")

FINAL_RESULTS = 0
DATE_TIME = str(datetime.now()).replace(' ', '_')

def Average(lst):
    return sum(lst) / len(lst)
    
# keep up the good work :)


class ERGOLightning(pl.LightningModule):

    def __init__(self, hparams):
        super(ERGOLightning, self).__init__()
        self.hparams = hparams
        self.dataset = hparams.dataset
        self.stage = hparams.stage
        # Model Type
        self.tcr_encoding_model = hparams.tcr_encoding_model
        self.use_alpha = hparams.use_alpha
        self.use_vj = hparams.use_vj
        self.use_mhc = hparams.use_mhc
        self.use_t_type = hparams.use_t_type
        self.cat_encoding = hparams.cat_encoding
        # Dimensions
        self.embedding_dim = hparams.embedding_dim
        # self.encoding_dim = hparams.encoding_dim
        self.aa_embedding_dim = hparams.aa_embedding_dim
        self.cat_embedding_dim = hparams.cat_embedding_dim
        self.lstm_dim = hparams.lstm_dim
        self.encoding_dim = hparams.encoding_dim
        self.dropout_rate = hparams.dropout
        self.lr = hparams.lr
        self.wd = hparams.wd
        self.hla_encoder = CNN_Encoder(self.embedding_dim, self.encoding_dim, vocab_size=22)
        # get train indicies for V,J etc 
        if self.cat_encoding == 'embedding':
            #with open('final_4_LE_split_new_counts_tcrb_mcpas_train_samples.pickle', 'rb') as handle:
            #    train = pickle.load(handle)
            
            #**with open('final_4_LE_split_new_counts_tcrb_mcpas_train_samples_cv.pickle', 'rb') as handle:
            
            #with open('validation_train_eval_list_4_1306.pickle', 'rb') as handle: #STAGE 2 OLD MCPAS
            #    train = pickle.load(handle)
            
            if self.stage == 1:
                with open('preproc_mcpas_train_0707.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
                    train_MCPAS = pickle.load(handle)
                    
                with open('memory_people_evaluation_train.pickle', 'rb') as handle: ##STAGE 1 NEW MEMORY 
                    train_memory = pickle.load(handle)
                    
                with open('new_dataset_train_samples.pickle', 'rb') as handle: #STAGE ONE NEW DATASET
                    train_new = pickle.load(handle)
                    
                with open('final_mcpas_neg_mhc_2206_train_samples.pickle', 'rb') as handle: #STAGE ONE NEW DATASET
                    train_real_mcpas = pickle.load(handle)
                    
                
                df = pd.DataFrame(train_real_mcpas)
                df['mhc'] = df['mhc'].apply(lambda x: x.split('-')[1][:1] + '*' + x.split('-')[1][1:])
                train_real_mcpas = df.to_dict('records')
                train = train_real_mcpas
                #train = train_MCPAS + train_memory + train_new
            
            else:
                with open('validation_mcpas_evaluation_train_v.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
                        train = pickle.load(handle)
            
            
            #with open('preproc_mcpas_evaluation_train.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
            #    train = pickle.load(handle)
                
                

            #with open('weighted_train_eval_list_4.pickle', 'rb') as handle:
            #    train = pickle.load(handle)

            #with open( 'emerson_train_eval_list_4.pickle', 'rb') as handle:  #500 
             #   train = pickle.load(handle)
            vatox, vbtox, jatox, jbtox, mhctox = get_index_dicts(train)
            self.v_vocab_size = len(vatox) + len(vbtox)
            self.j_vocab_size = len(jatox) + len(jbtox)
            self.mhc_vocab_size = len(mhctox)
        # TCR Encoder
        if self.tcr_encoding_model == 'AE':
            if self.use_alpha:
                self.tcra_encoder = AE_Encoder(encoding_dim=self.encoding_dim, tcr_type='alpha', max_len=34)
            self.tcrb_encoder = AE_Encoder(encoding_dim=self.encoding_dim, tcr_type='beta')
        elif self.tcr_encoding_model == 'LSTM':
            if self.use_alpha:
                self.tcra_encoder = LSTM_Encoder(self.aa_embedding_dim, self.lstm_dim, self.dropout_rate)
            self.tcrb_encoder = LSTM_Encoder(self.aa_embedding_dim, self.lstm_dim, self.dropout_rate)
            self.encoding_dim = self.lstm_dim
        # Peptide Encoder
        self.pep_encoder = LSTM_Encoder(self.aa_embedding_dim, self.lstm_dim, self.dropout_rate)
        # Categorical
        self.cat_encoding = hparams.cat_encoding
        if hparams.cat_encoding == 'embedding':
            if self.use_vj:
                self.v_embedding = nn.Embedding(self.v_vocab_size, self.cat_embedding_dim, padding_idx=0)
                self.j_embedding = nn.Embedding(self.j_vocab_size, self.cat_embedding_dim, padding_idx=0)
            if self.use_mhc:
                self.mhc_embedding = nn.Embedding(self.mhc_vocab_size, self.cat_embedding_dim, padding_idx=0)
        # different mlp sizes, depends on model input
        if self.cat_encoding == 'binary':
            self.cat_embedding_dim = 10
        mlp_input_size = self.lstm_dim + self.encoding_dim
        if self.use_vj:
            mlp_input_size += 2 * self.cat_embedding_dim
        if self.use_mhc:
            mlp_input_size += self.cat_embedding_dim
        if self.use_t_type:
            mlp_input_size += 1
        # MLP I (without alpha)
        self.mlp_dim1 = 301
        self.hidden_layer1 = nn.Linear(self.mlp_dim1, int(np.sqrt(self.mlp_dim1)))
        self.relu = torch.nn.LeakyReLU()
        self.output_layer1 = nn.Linear(int(np.sqrt(self.mlp_dim1)), 1)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        # MLP II (with alpha)
        if self.use_alpha:
            mlp_input_size += self.encoding_dim
            if self.use_vj:
                mlp_input_size += 2 * self.cat_embedding_dim
            self.mlp_dim2 = 501
            self.hidden_layer2 = nn.Linear(self.mlp_dim2, int(np.sqrt(self.mlp_dim2)))
            self.output_layer2 = nn.Linear(int(np.sqrt(self.mlp_dim2)), 1)

    def forward(self, tcr_batch, mhc_batch, cat_batch, t_type_batch):
        # PEPTIDE Encoder:
        # pep_encoding = self.pep_encoder(*pep_batch)
        # TCR Encoder:
        tcra, tcrb = tcr_batch
        tcrb_encoding = self.tcrb_encoder(*tcrb)
        # Categorical Encoding:
        #print('cat batch: ', cat_batch)
        va, vb, ja, jb = cat_batch
        # T cell type
        t_type = t_type_batch.view(len(t_type_batch), 1)
        # gather all features, int linear mlp so the order does not matter
        mlp_input = [tcrb_encoding]
        if self.use_vj:
            if self.cat_encoding == 'embedding':
                va = self.v_embedding(va)
                vb = self.v_embedding(vb)
                ja = self.j_embedding(ja)
                jb = self.j_embedding(jb)
            mlp_input += [vb, jb]
        if True: #self.use_mhc:
            if self.cat_encoding == 'embedding':
                mhc = self.hla_encoder(*mhc_batch)
                #print('mhc', mhc)
            mlp_input += [mhc]
        if self.use_t_type:
            mlp_input += [t_type]
            
        if  tcra:
            tcra_encoding = self.tcra_encoder(*tcra)
            mlp_input += [tcra_encoding]
            if self.use_vj:
                mlp_input += [va, ja]
            # MLP II Classifier
            concat = torch.cat(mlp_input, 1)
            
            hidden_output = self.dropout(self.relu(self.hidden_layer2(concat)))
            mlp_output = self.output_layer2(hidden_output)
        else:
            #print('MLP I Classifier')
            concat = torch.cat(mlp_input, 1)
            hidden_output = self.dropout(self.relu(self.hidden_layer1(concat)))
            mlp_output = self.output_layer1(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output

    def step(self, batch):
        #print('batch:  ', batch[0])
        # batch is none (might happen in evaluation)
        if not batch:
            return None
        # batch output (always)
        tcra, tcrb, mhc, va, vb, ja, jb, t_type, sign, weight,org_tcrb = batch
        #print('va: ', va)
        # print('step', mhc)
        if self.tcr_encoding_model == 'AE':
            len_a = torch.sum(tcra, dim=[1, 2]) - 1
        # len_p = torch.sum((pep > 0).int(), dim=1)
        if self.use_alpha:
            missing = (len_a == 0).nonzero(as_tuple=True)
            full = len_a.nonzero(as_tuple=True)
            if self.tcr_encoding_model == 'AE':
                tcra_batch_ful = (tcra[full],)
                tcrb_batch_ful = (tcrb[full],)
                tcrb_batch_mis = (tcrb[missing],)
            tcr_batch_ful = (tcra_batch_ful, tcrb_batch_ful)
            tcr_batch_mis = (None, tcrb_batch_mis)
            device = len_a.device
            y_hat = torch.zeros(len(sign)).to(device)
            # there are samples without alpha
            if len(missing[0]):
                mhc_mis = (mhc[missing], )
                #print('mhc_mis', mhc_mis)

                t_type_mis = t_type[missing]
                cat_mis = (va[missing], vb[missing], ja[missing], jb[missing])#,
                #print('step forworddddddd')
                y_hat_mis = self.forward(tcr_batch_mis, mhc_mis, cat_mis, t_type_mis).squeeze()
                y_hat[missing] = y_hat_mis
            # there are samples with alpha
            if len(full[0]):
                mhc_ful = (mhc[full], )
                # print('mhc_ful', mhc_ful)
                cat_ful = (va[full], vb[full],
                           ja[full], jb[full])
                           # mhc[full])
                t_type_ful = t_type[full]
                #print('step   **********************')
                y_hat_ful = self.forward(tcr_batch_ful,mhc_ful, cat_ful, t_type_ful).squeeze()
                y_hat[full] = y_hat_ful
        y = sign
        return y, y_hat, weight, #return all values to calculate new accuracy

    def training_step(self, batch, batch_idx):
        # REQUIRED
        self.train()
        
        y, y_hat, weight = self.step(batch)
        loss = F.binary_cross_entropy(y_hat, y, weight=weight)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        self.eval()
        tcra, tcrb, mhc, va, vb, ja, jb, t_type, sign, weight, org_tcrb = batch
       # print('va: ', va)
        if self.step(batch):
            y, y_hat, _ = self.step(batch)
            return {'val_loss': F.binary_cross_entropy(y_hat.view(-1, 1), y.view(-1, 1)), 'y_hat': y_hat, 'y': y, 'tcrb': org_tcrb}
        else:
            return None

    def validation_end(self, outputs):
        #print(outputs)
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'].view(-1, 1) for x in outputs])
        y_hat = torch.cat([x['y_hat'].view(-1, 1) for x in outputs])
        tcrb = [x['tcrb'] for x in outputs]
        tcrb  = [item for sublist in tcrb for item in sublist]
        print(len(y), len(y_hat), len(tcrb[0]))
        df = pd.DataFrame(list(zip(tcrb, y_hat.view(-1, 1).tolist(), y.view(-1, 1).tolist())), columns =['tcrb', 'y_hat','y'])
        
        df['tcrb'] = df['tcrb'].apply(lambda x: x[0])
        df['y_hat'] = df['y_hat'].apply(lambda x: x[0])
        df['y'] = df['y'].apply(lambda x: x[0])
        #df = df[df['y'] == 1]
        #print(df.head())
        df = df.sort_values(by=['tcrb', 'y_hat'], ascending=False)
        #df_1 = df_1.sort_values(by=['tcrb', 'y_hat'], ascending=False)
        df = df.drop_duplicates(['tcrb'])
        
        
        # auc = roc_auc_score(y.cpu(), y_hat.cpu())
        acc = accuracy_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy().round())
        print('\n')
        print('********************')
        print('NEW_ACC = ', len(df[df['y'] == 1])/len(df))
        print('acc=', acc)
        auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        print('auc=', auc)
        print('\n')
        
        
        fpr, tpr, threshold = metrics.roc_curve(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        
        # method I: plt
        #plt.title('Receiver Operating Characteristic')
        plt.figure(figsize=(3.5,3.5))
        plt.plot(fpr, tpr, 'black')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams.update({'font.size': 8})
        
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],color = 'black', linestyle='dashed')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('McPAS')
        plt.savefig('study_times/NM_on_McPAS{0}.png'.format(str(auc)+str(acc)))
        plt.clf()
        
        
        n = y_hat.detach().cpu().numpy()
        k = np.where(n > 0.5,int(1),int(0))
        #print('confusion_matrix')
        #print(confusion_matrix(y.detach().cpu().numpy(), k))
        C = confusion_matrix(y.detach().cpu().numpy(), k)
        t = y.detach().cpu().numpy()
        #print(len(y.detach().cpu().numpy()), len(k))
        tn = 0
        fp = 0
        tp = 0
        fn = 0
        for i in range(len(k)):
            if(t[i]==1 and k[i] == 0):
                fn = fn + 1
            if(t[i]==0 and k[i] == 1):
                fp = fp + 1
            if(t[i]==1 and k[i] == 1):
                tp = tp + 1
            if(t[i]==0 and k[i] == 0):
                tn = tn + 1
          

        print('tn:  ', tn, 'fp:  ', fp, 'tp:  ', tp, 'fnn:  ', fn )
        C = C / C.astype(np.float).sum(axis=1, keepdims=True)
        print(C)
        
        nni.report_intermediate_result(auc)
        tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc, 'val_acc': acc}
        global FINAL_RESULTS
        FINAL_RESULTS = auc
        return {'avg_val_loss': avg_loss, 'val_auc': auc, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pass

    def test_end(self, outputs):
        pass

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        #print('dataset', self.dataset)
        #train_eval_list_4
        
        #with open('emerson_train_eval_list_4.pickle', 'rb') as handle:
        #with open('Emerson_split_15000_tcrb_train_samples_with_y_hat_cd.pickle', 'rb') as handle:
        #**with open('final_4_LE_split_new_counts_tcrb_mcpas_train_samples_cv.pickle', 'rb') as handle:
        #with open('validation_train_eval_list_4_1306.pickle', 'rb') as handle:
        #    train = pickle.load(handle)
            
        if self.stage == 1:
            with open('preproc_mcpas_train_0707.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
                train = pickle.load(handle)
            with open('preproc_mcpas_train_0707.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
                train_MCPAS = pickle.load(handle)
            with open('final_mcpas_neg_mhc_2206_train_samples.pickle', 'rb') as handle: #STAGE ONE NEW DATASET
                train_real_mcpas = pickle.load(handle)
            df = pd.DataFrame(train_real_mcpas)
            df['mhc'] = df['mhc'].apply(lambda x: x.split('-')[1][:1] + '*' + x.split('-')[1][1:])
            train_real_mcpas = df.to_dict('records')
            train = train_real_mcpas

        else:
            with open('validation_mcpas_evaluation_train_v.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
                    train = pickle.load(handle)
        
        with open('preproc_mcpas_train_0707.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
            train_MCPAS = pickle.load(handle)
            
        with open('memory_people_evaluation_train.pickle', 'rb') as handle: ##STAGE 1 NEW MEMORY 
            train_memory = pickle.load(handle)
            
        with open('new_dataset_train_samples.pickle', 'rb') as handle: #STAGE ONE NEW DATASET
            train_new = pickle.load(handle)
            
        #train = train_MCPAS + train_memory + train_new
                    
        
                
        #with open('preproc_mcpas_evaluation_train.pickle', 'rb') as handle: ##STAGE 2 NEW MCPAS 
        #        train = pickle.load(handle)
                
        #with open('weighted_train_eval_list_4.pickle', 'rb') as handle:
        #    train = pickle.load(handle)
        
        
        print('load train')
        train_dataset = SignedPairsDataset(train, get_index_dicts(train))
        dl =  DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10,
                          collate_fn=lambda b: train_dataset.collate(b, tcr_encoding=self.tcr_encoding_model,
                                                                     cat_encoding=self.cat_encoding))

        return dl
        #df_train.to_pickle('train_head_y_hat_cd.pickle')
        #df_train = df_train.sort_values(by=['y_hat'])
        #df_train = df_train.sample(int(0.05*len(df_train)))
        #df_train = df_train.head(int(0.01*len(df_train)))
        #print(len(df_train))
        
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

                                                                     
    

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        #print('dataset', self.dataset)
        #test_eval_list_4
        #with open('emerson_test_eval_list_4.pickle', 'rb') as handle: #500
        #with open('Emerson_split_15000_tcrb_test_samples_with_y_hat_cd.pickle', 'rb') as handle: EMERSON DATASET IN STAGE **ONE**
        #with open('final_4_LE_split_new_counts_tcrb_mcpas_train_samples.pickle', 'rb') as handle:
        #with open('emerson_test_eval_list_4_15000.pickle', 'rb') as handle:   #FINAL CHECK OF PREPORMENCE ON EMERSON DATASET IN STAGE **TWO**
        #    test = pickle.load(handle)
        
        #with open('final_4_LE_split_new_counts_tcrb_mcpas_test_samples_cv.pickle', 'rb') as handle: #STAGE 1 OLD MCPAS 
        #with open('validation_test_eval_list_4_1306.pickle', 'rb') as handle:
        #    test = pickle.load(handle)
        
        
        with open('final_mcpas_neg_mhc_2206_test_samples.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
              test = pickle.load(handle)
              df = pd.DataFrame(test)
              df['mhc'] = df['mhc'].apply(lambda x: x.split('-')[1][:1] + '*' + x.split('-')[1][1:])
              test = df.to_dict('records')
        
        #with open('memory_train_v_people_test.pickle', 'rb') as handle: ##STAGE 1 NEW MEMORY 
        #        test = pickle.load(handle)
                
        #with open('Emerson_split_15000_tcrb_test_samples_with_y_hat_cd.pickle', 'rb') as handle: #EMERSON DATASET IN STAGE **ONE**
        #    test = pickle.load(handle)
        #df_test = pd.DataFrame.from_dict(test)
        #TAKE ONLY THE MOST CD8 - NESSECERY ONLY AT EMERSON DATASET STAGE1
        #df_test = pd.DataFrame.from_dict(test)
        #del test
        #df_test.to_pickle('test_head_y_hat_cd.pickle')
        #df_test = df_test.sort_values(by=['y_hat'])
        #print(int(0.01*len(df_test)))
        #df_test = df_test.head(int(0.01*len(df_test)))
        
        ############################################################
        #hlas = df_test.mhc.unique()
        #positive_df = df_test[df_test['sign'] == 1]
        #positive_hla = positive_df.mhc.value_counts()
        #negative_df = df_test[df_test['sign'] == 0]
        #negative_hla = negative_df.mhc.value_counts()

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
            #else:
                #temp_df = df_test[df_test['mhc'] == i]
         #   dfs.append(temp_df)
        #df_test = pd.concat(dfs).sample(frac=1)
        
        #print('load_test')
        
        #df_test['mhc'] = df_test['mhc'].apply(lambda x : x.split('-')[1])
        
        #test = df_test.to_dict('records')
        #del df_test
        

        #CHECK ON NEW_DATASET
        #with open('new_dataset_test_samples.pickle','rb') as handle: #NEW DATASET STAGE ONE
        #    test = pickle.load(handle)
        #df_test = pd.DataFrame(test)
        #df_test['mhc'] = df_test['mhc'].apply(lambda x : x.split('-')[1])
        #test = df_test.to_dict('records')
        
        #CHELK ON MEMORY
        #with open('memory_train_v_people_test.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
        #      test = pickle.load(handle)
              
        #check random accuracy
        #test_df = pd.DataFrame(test)
        #test_df = test_df[test_df['sign'] == 0]
        #test_df['sign'] = test_df['sign'].apply(lambda x: randrange(2))
        #test = test_df.to_dict('records')

        #with open('test_eval_list_4.pickle', 'rb') as handle:  
        #    test = pickle.load(handle)
        
        #with open('final_4_LE_split_new_counts_tcrb_mcpas_train_samples_cv.pickle', 'rb') as handle:
        #with open('validation_train_eval_list_4_1306.pickle', 'rb') as handle: #STAGE 2 OLS MC[AS
        #    train = pickle.load(handle)
        
        if self.stage == 1:
            with open('preproc_mcpas_train_0707.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
                train = pickle.load(handle)
            with open('preproc_mcpas_train_0707.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
                train_MCPAS = pickle.load(handle)
                
            with open('final_mcpas_neg_mhc_2206_train_samples.pickle', 'rb') as handle: #STAGE ONE NEW DATASET
                    train_real_mcpas = pickle.load(handle)
            df = pd.DataFrame(train_real_mcpas)
            df['mhc'] = df['mhc'].apply(lambda x: x.split('-')[1][:1] + '*' + x.split('-')[1][1:])
            train_real_mcpas = df.to_dict('records')
            
            train = train_real_mcpas

            
        else:
            with open('validation_mcpas_evaluation_train_v.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
                    train = pickle.load(handle)
                    
                    
                    
        with open('preproc_mcpas_train_0707.pickle', 'rb') as handle: ##STAGE 1 NEW MCPAS 
            train_MCPAS = pickle.load(handle)
            
        with open('memory_people_evaluation_train.pickle', 'rb') as handle: ##STAGE 1 NEW MEMORY 
            train_memory = pickle.load(handle)
            
        with open('new_dataset_train_samples.pickle', 'rb') as handle: #STAGE ONE NEW DATASET
            train_new = pickle.load(handle)
        #train = train_MCPAS + train_memory + train_new
            
        #with open('preproc_mcpas_evaluation_train.pickle', 'rb') as handle: ##STAGE 2 NEW MCPAS 
        #        train = pickle.load(handle)
            
        #with open('weighted_train_eval_list_4.pickle', 'rb') as handle:
        #    train = pickle.load(handle)

        test_dataset = SignedPairsDataset(test, get_index_dicts(train))
        l =   DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=10,
                          collate_fn=lambda b: test_dataset.collate(b, tcr_encoding=self.tcr_encoding_model,
                                                                     cat_encoding=self.cat_encoding))

        #df_test = pd.DataFrame.from_dict(test)
        #del test
        #df_test.to_pickle('test_head_y_hat_cd.pickle')
        #df_test = df_test.sort_values(by=['y_hat'])
        #print(int(0.01*len(df_test)))
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
         #   temp_neg = negative_df[negative_df['mhc'] == i]
         #   temp_neg = temp_neg.head(min_dict[i])# + int((len(temp_neg)-min_dict[i])/15))
         #   temp_pos = positive_df[positive_df['mhc'] == i]
         #   temp_pos = temp_pos.head(min_dict[i])# + int((len(temp_pos)-min_dict[i])/15))
         #   print(len(temp_pos), len(temp_neg))
         #   temp_df = pd.concat([temp_neg, temp_pos], axis = 0)
            #else:
                #temp_df = df_test[df_test['mhc'] == i]
          #  dfs.append(temp_df)
        #df_test = pd.concat(dfs).sample(frac=1)


        #################################################
        #df_test.to_pickle('test_head_y_hat_cd.pickle')
        
        
        #with open('Emerson_split_15000_tcrb_train_samples_with_y_hat_cd.pickle', 'rb') as handle:

        #with open('emerson_train_eval_list_4.pickle', 'rb') as handle: #500
        #    train = pickle.load(handle)
        #df_train = pd.DataFrame.from_dict(train)
        #del train
        #df_train = df_train.sort_values(by=['y_hat'])
        #df_train = df_train.head(int(0.01*len(df_train)))
        #print('len train', len(df_train))
        #hlas = df_train.mhc.unique()
        #positive_df = df_train[df_train['sign'] == 1]
        #positive_hla = positive_df.mhc.value_counts()
        #negative_df = df_train[df_train['sign'] == 0]
        #negative_hla = negative_df.mhc.value_counts()
        
        #min_dict = {}
        #for i in hlas:
        #    if  i in list(positive_hla.index) and i in list(negative_hla.index):
        #        m = min(positive_hla[i],negative_hla[i])
        #        min_dict[i] = (min(positive_hla[i],negative_hla[i]))
            
        #dfs = []
        #print('min_dict.', min_dict.keys())
        #for i in list(min_dict.keys()):
        #    temp_neg = negative_df[negative_df['mhc'] == i]
        #    temp_neg = temp_neg.head(min_dict[i])# + int((len(temp_neg)-min_dict[i])/10))
        #    temp_pos = positive_df[positive_df['mhc'] == i]
        #    temp_pos = temp_pos.head(min_dict[i])# + int((len(temp_pos)-min_dict[i])/10))
        #    temp_df = pd.concat([temp_neg, temp_pos], axis = 0)
        #    dfs.append(temp_df)
        #df_train = pd.concat(dfs).sample(frac=1)
        #train = df_train.to_dict('records')
        #del df_train
        
        return l

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        pass




def tcr_mhc_2(dict):
    parser = ArgumentParser()
    parser.add_argument('--iter', type=str, default='tcr_mhc')
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='Emerson')  # Emerson
    parser.add_argument('--tcr_encoding_model', type=str, default='AE')  # or AE
    parser.add_argument('--cat_encoding', type=str, default='embedding')
    parser.add_argument('--use_alpha', type=bool, default=True)
    parser.add_argument('--use_vj', type=bool, default=True)
    parser.add_argument('--use_mhc', type=bool, default=True)
    parser.add_argument('--use_t_type', type=bool, default=True)
    parser.add_argument('--aa_embedding_dim', type=int, default=10)
    parser.add_argument('--cat_embedding_dim', type=int, default=50)
    parser.add_argument('--lstm_dim', type=int, default=500)
    parser.add_argument('--encoding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=dict['lr']) #=1e-3)
    parser.add_argument('--wd', type=float, default= dict['l2'])#1e-5)
    parser.add_argument('--dropout', type=float, default = dict['dropout']) #=0.1)
    parser.add_argument('--weight_factor', type=int, default=1)
    parser.add_argument('--stage', type=int, default=1)
    #parser.add_argument('--diabetes_factor', type=int, default=5)

    hparams = parser.parse_args()
    print(hparams)
    model = ERGOLightning(hparams)
    # version flags
    version = ''
    version += str(hparams.iter)
    if hparams.dataset == 'mcpas_human':
        version += 'h'
    elif hparams.dataset == 'vdjdb':
        version += 'v'
    elif hparams.dataset == 'mcpas':
        version += 'm'
    else:
        version += 'x'
    if hparams.tcr_encoding_model == 'AE':
        version += 'e'
    elif hparams.tcr_encoding_model == 'LSTM':
        version += 'l'
    if hparams.use_alpha:
        version += 'a'
    if hparams.use_vj:
        version += 'j'
    if hparams.use_mhc:
        version += 'h'
    if hparams.use_t_type:
        version += 't'
    logger = TensorBoardLogger("pre_trainer_logs", name="tcr_hla_binding", version='server_mcpas')
    early_stop_callback = EarlyStopping(monitor='val_acc', patience=20, mode='max')
    trainer = Trainer(logger=logger, early_stop_callback=early_stop_callback, gpus=[2]) #[0]) #, distributed_backend='ddp_cpu')
    trainer.fit(model)


if __name__ == '__main__':
    # ergo_ii_experiment()
    # weighted_ergo_ii_experiment_all_data()
    # weighted_ergo_ii_experiment()
    # diabetes_experiment()
    # ergo_ii_tuning()
    
    
    #final_results = []
    #mcpas_test = pd.read_pickle('final_4_LE_split_new_counts_tcrb_mcpas_test_samples.pickle')
    #mcpas_train = pd.read_pickle('final_4_LE_split_new_counts_tcrb_mcpas_train_samples.pickle')
    #mcpas = np.array(mcpas_test + mcpas_train)
    #kf = KFold(2)
    
    try:
        dict = nni.get_next_parameter()
    except Exception as exception:
        _logger.exception(exception)
        print('exception')
        raise
    #for train_index, test_index in kf.split(mcpas):
    #    train = mcpas[train_index]
        #print(len())
        #with open('final_4_LE_split_new_counts_tcrb_mcpas_train_samples_cv.pickle', 'wb') as handle:
        #    pickle.dump(train, handle)

        #test = mcpas[test_index]
        #with open('final_4_LE_split_new_counts_tcrb_mcpas_test_samples_cv.pickle', 'wb') as handle:
        #    pickle.dump(test, handle)
        #args = prepare(RCV_CONFIG)
        #of 0.005, drop out of 0.2, and learning rate of 0.0005
    dict = {'lr': 0.0007, 'dropout': 0.2, 'l2': 0.001, 'encoding_dim' : 100}
    print(dict)
    print(dict)
    tcr_mhc_2(dict)
    #final_results.append(FINAL_RESULTS)
    #print('FINAL_RESULTS', FINAL_RESULTS)

        
    # args = prepare(RCV_CONFIG)
    #dict = {'lr': 0.0001, 'dropout': 0.2, 'l2': 0.0007, 'encoding_dim' : 100}
    #tcr_mhc_2(dict)
    nni.report_final_result(Average(final_results))
    pass


# NOTE: fix sklearn import problem with this in terminal:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dsi/speingi/anaconda3/lib/

# see logs
# tensorboard --logdir dir