import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from Loader import SignedPairsDataset, get_index_dicts
from Models import PaddingAutoencoder, AE_Encoder, LSTM_Encoder, ERGO
import pickle
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve
import sklearn.metrics as metrics
from argparse import ArgumentParser
import nni
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

_logger = logging.getLogger("nni")

FINAL_RESULTS = 0


class CLAIRELightning(pl.LightningModule):
    def __init__(self, hparams):
        super(CLAIRELightning, self).__init__()
        self.hparams = hparams
        self.dataset = hparams.dataset
        # Model Type
        self.tcr_encoding_model = hparams.tcr_encoding_model
        self.use_alpha = hparams.use_alpha
        self.use_vj = hparams.use_vj
        self.use_mhc = hparams.use_mhc
        self.use_t_type = hparams.use_t_type
        self.cat_encoding = hparams.cat_encoding
        # Dimensions
        self.aa_embedding_dim = hparams.aa_embedding_dim
        self.cat_embedding_dim = hparams.cat_embedding_dim
        self.lstm_dim = hparams.lstm_dim
        self.encoding_dim = hparams.encoding_dim
        self.dropout_rate = hparams.dropout
        self.lr = hparams.lr
        self.wd = hparams.wd
        # get train indicies for V,J etc
        if self.cat_encoding == 'embedding':
            with open(self.dataset + '_train_samples.pickle' , 'rb') as handle: 
                train = pickle.load(handle)
            vatox, vbtox, jatox, jbtox, mhctox = get_index_dicts(train)
            self.v_vocab_size = len(vatox) + len(vbtox)
            print(self.v_vocab_size)
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
        # MLP I (without alpha)
        self.mlp_dim1 = 200
        self.hidden_layer1 = nn.Linear(self.mlp_dim1, int(np.sqrt(self.mlp_dim1)))
        self.relu = torch.nn.LeakyReLU()
        self.output_layer1 = nn.Linear(int(np.sqrt(self.mlp_dim1)), 1)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        # MLP II (with alpha)
        if self.use_alpha:
            mlp_input_size += self.encoding_dim
            if self.use_vj:
                mlp_input_size += 2 * self.cat_embedding_dim
            self.mlp_dim2 = 400
            self.hidden_layer2 = nn.Linear(self.mlp_dim2, int(np.sqrt(self.mlp_dim2)))
            self.output_layer2 = nn.Linear(int(np.sqrt(self.mlp_dim2)), 1)

    def forward(self, tcr_batch, cat_batch):
        # TCR Encoder:
        tcra, tcrb = tcr_batch
        tcrb_encoding = self.tcrb_encoder(*tcrb)
        # Categorical Encoding:
        va, vb, ja, jb = cat_batch
        # gather all features, int linear mlp so the order does not matter
        mlp_input = [tcrb_encoding] #pep_encoding
        if self.use_vj:
            if self.cat_encoding == 'embedding':
                va = self.v_embedding(va)
                vb = self.v_embedding(vb)
                ja = self.j_embedding(ja)
                jb = self.j_embedding(jb)
            mlp_input += [vb, jb]
        concat = torch.cat(mlp_input, 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer1(concat)))
        mlp_output = self.output_layer1(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output

    def step(self, batch):
        if not batch:
            return None
        tcra, tcrb, pep, va, vb, ja, jb, mhc, t_type, sign, weight = batch
        if self.tcr_encoding_model == 'LSTM':
            # get lengths for lstm functions
            len_b = torch.sum((tcrb > 0).int(), dim=1)
            len_a = torch.sum((tcra > 0).int(), dim=1)
        if self.tcr_encoding_model == 'AE':
            len_a = torch.sum(tcra, dim=[1, 2]) - 1
        if self.use_alpha:
            missing = (len_a == 0).nonzero(as_tuple=True)
            full = len_a.nonzero(as_tuple=True)
            if self.tcr_encoding_model == 'LSTM':
                tcra_batch_ful = (tcra[full], len_a[full])
                tcrb_batch_ful = (tcrb[full], len_b[full])
                tcrb_batch_mis = (tcrb[missing], len_b[missing])
            elif self.tcr_encoding_model == 'AE':
                tcra_batch_ful = (tcra[full],)
                tcrb_batch_ful = (tcrb[full],)
                tcrb_batch_mis = (tcrb[missing],)
            tcr_batch_ful = (tcra_batch_ful, tcrb_batch_ful)
            tcr_batch_mis = (None, tcrb_batch_mis)
            device = len_a.device
            y_hat = torch.zeros(len(sign)).to(device)
            # there are samples without alpha
            if len(missing[0]):
                cat_mis = (va[missing], vb[missing],
                           ja[missing], jb[missing])
                t_type_mis = t_type[missing]
                y_hat_mis = self.forward(tcr_batch_mis, cat_mis).squeeze()
                y_hat[missing] = y_hat_mis
            # there are samples with alpha
            if len(full[0]):
                cat_ful = (va[full], vb[full],
                           ja[full], jb[full])
                y_hat_ful = self.forward(tcr_batch_ful, cat_ful).squeeze()
                y_hat[full] = y_hat_ful
        y = t_type
        return y, y_hat, weight

    def training_step(self, batch, batch_idx):
        self.train()
        y, y_hat, weight = self.step(batch)
        loss = F.binary_cross_entropy(y_hat, y, weight=weight)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        self.eval()
        if self.step(batch):
            tcra, tcrb, pep, va, vb, ja, jb, mhc, t_type, sign, weight = batch
            y, y_hat, _ = self.step(batch)
            return {'val_loss': F.binary_cross_entropy(y_hat.view(-1, 1), y.view(-1, 1)), 'y_hat': y_hat, 'y': y}
        else:
            return None

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'].view(-1, 1) for x in outputs])
        y_hat = torch.cat([x['y_hat'].view(-1, 1) for x in outputs])
        auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        acc = accuracy_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy().round())
        nni.report_intermediate_result(auc)
        fpr, tpr, threshold = metrics.roc_curve(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(3,3))
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams.update({'font.size': 8.5})
        plt.plot(fpr, tpr, 'black')
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],color = 'black', linestyle='dashed')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.tight_layout()
        plt.savefig('roc/{0}.png'.format(self.dataset))
        plt.clf()

        tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc}
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
        with open(self.dataset + '_train_samples.pickle', 'rb') as handle:
            train = pickle.load(handle)
            
        train_dataset = SignedPairsDataset(train, get_index_dicts(train))
        return DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10,
                          collate_fn=lambda b: train_dataset.collate(b, tcr_encoding=self.tcr_encoding_model,
                                                                     cat_encoding=self.cat_encoding))

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL 
        with open(self.dataset + '_test_samples.pickle', 'rb') as handle:
            test = pickle.load(handle)
            # print('test', test[0])

        with open(self.dataset + '_train_samples.pickle' , 'rb') as handle:
            train = pickle.load(handle)

        test_dataset = SignedPairsDataset(test, get_index_dicts(train))
        return DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=10,
                          collate_fn=lambda b: test_dataset.collate(b, tcr_encoding=self.tcr_encoding_model,
                                                                     cat_encoding=self.cat_encoding))

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        pass




def tcr_mhc(dict):
    parser = ArgumentParser()
    parser.add_argument('--iter', type=str, default='D_cd4_5_5_5')
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='mcpas')  # or vdjdb
    parser.add_argument('--tcr_encoding_model', type=str, default='AE')  # or AE
    parser.add_argument('--cat_encoding', type=str, default='embedding')
    parser.add_argument('--use_alpha', type=bool, default=True)
    parser.add_argument('--use_vj', type=bool, default=True)
    parser.add_argument('--use_mhc', type=bool, default=True)
    parser.add_argument('--use_t_type', type=bool, default=True)
    parser.add_argument('--aa_embedding_dim', type=int, default=10)
    parser.add_argument('--cat_embedding_dim', type=int, default=50)
    parser.add_argument('--lstm_dim', type=int, default=500)
    parser.add_argument('--encoding_dim', type=int, default=100) #100)
    parser.add_argument('--lr', type=float, default= dict['lr']) #1e-3
    parser.add_argument('--wd', type=float, default= dict['l2']) #1e-5)
    parser.add_argument('--dropout', type=float, default=dict['dropout'])#0.1)
    parser.add_argument('--weight_factor', type=int, default=1)

    hparams = parser.parse_args()
    print(hparams)
    model = CLAIRELightning(hparams)
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
    logger = TensorBoardLogger("ERGO-II_tcr", name="tcr_cdr", version = '1107')
    early_stop_callback = EarlyStopping(monitor='val_auc', patience=100, mode='max')
    trainer = Trainer(logger=logger, early_stop_callback=early_stop_callback,gpus=[1] )#distributed_backend='ddp_cpu')
    trainer.fit(model)
    print('done fitting')



if __name__ == '__main__':
    dict = {'lr': 0.0005, 'dropout': 0.1, 'l2': 0.0005, 'encoding_dim': 100}
    tcr_mhc(dict)
    # nni.report_final_result(FINAL_RESULTS)
    pass

# see logs
# tensorboard --logdir dir
