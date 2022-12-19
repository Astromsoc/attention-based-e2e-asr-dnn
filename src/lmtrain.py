"""
    Training an additional language model to assist beam search.
"""

import os
import json
import time
import yaml
import wandb
import argparse
import Levenshtein
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchsummaryX import summary

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import *
from src.constants import *
from src.train import Trainer
from src.models import MultiheadCrossAttention
from src.modules import LockedLSTM, AutoRegDecoderLSTMCell



class lmDatasetTrainDev(Dataset):
    """
        Dataloader for training and dev sets: both features & labels
    """
    def __init__(
        self, trans_dir: str=None, pred_dir: str=None, label_to_idx: dict=None
    ):
        super().__init__()
        # bookkeeping
        self.trans_dir = trans_dir
        self.pred_dir = pred_dir
        self.label_to_idx = label_to_idx
        self.sos_idx = self.label_to_idx['<sos>']
        self.eos_idx = self.label_to_idx['<eos>']
        
        # load all filenames
        trans_fns = sorted([
            f"{self.trans_dir}/{f}" for f in os.listdir(self.trans_dir) if f.endswith('.npy')
        ])


        # load files
        self.predictions = [
            torch.tensor([self.sos_idx] + [self.label_to_idx[t] for t in p] + [self.eos_idx])
                          for p in tqdm([l.strip() for l in open(self.pred_dir, 'r')], 
                                        leave=False, desc='loading predictions...')
        ]
        self.transcripts = [
            torch.tensor([self.label_to_idx[p] for p in np.load(f)]) 
            for f in tqdm(trans_fns, leave=False, desc='loading transcripts...') 
            if f.endswith('.npy')
        ]
        # dataset size
        assert len(self.predictions) == len(self.transcripts)
        self.size = len(self.transcripts)

    
    def __len__(self):
        return self.size

    
    def __getitem__(self, index):
        return self.predictions[index], self.transcripts[index]

    
    def collate_fn(self, batch):
        """
            Collate function for training and dev sets, 4 returns
        """
        predictions = [u[0] for u in batch]
        transcripts = [u[1] for u in batch]

        # obtain original lengths for both mfccs & transcripts
        prediction_lens = [len(p) for p in predictions]
        transcript_lens = [len(t) for t in transcripts]

        # pad both mfccs & transcripts
        predictions = pad_sequence(
            predictions, batch_first=True, padding_value=self.eos_idx
        )
        transcripts = pad_sequence(
            transcripts, batch_first=True, padding_value=self.eos_idx
        )        
        return (predictions, transcripts, 
                torch.tensor(prediction_lens), torch.tensor(transcript_lens))



class Rewriter(nn.Module):
    """
        BiLSTM attention based seq2seq model that rescore / transform LAS outputs
    """
    def __init__(
        self,
        # embeddings
        vocab_size: int=30,
        emb_dim: int=256,
        # encoders
        enc_lstm_layers: int=3,
        enc_lstm_hid_dim: int=256,
        enc_dropouts: list=[0.3, 0.3],
        # attention
        att_proj_dim: int=128,
        att_heads: int=4,
        att_dropout: float=0.2,
        # decoder: 2 LSTM cells (similar to speller)
        dec_lstm_layers: int=2,
        dec_lstm_hid_dim: int=256,
        dec_lstm_out_dim: int=128,
        dec_lstm_dropout: float=0.3,
        # trivials
        CHR_PAD_IDX: int=29,
        CHR_MAX_STEPS: int=600,
        CHR_SOS_IDX: int=0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_lstm_layers = enc_lstm_layers
        self.enc_lstm_hid_dim = enc_lstm_hid_dim
        self.enc_dropouts = enc_dropouts
        self.att_proj_dim = att_proj_dim
        self.att_heads = att_heads
        self.att_dropout = att_dropout
        self.dec_lstm_layers = dec_lstm_layers
        self.dec_lstm_hid_dim = dec_lstm_hid_dim
        self.dec_lstm_out_dim = dec_lstm_out_dim
        self.dec_lstm_dropout = dec_lstm_dropout
        self.CHR_PAD_IDX = CHR_PAD_IDX
        self.CHR_MAX_STEPS = CHR_MAX_STEPS
        self.CHR_SOS_IDX = CHR_SOS_IDX

        # character embeddings
        self.char_emb = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.emb_dim,
            padding_idx=self.CHR_PAD_IDX
        )
        # encoder lstms
        self.enc_lstm = LockedLSTM(
            lstm_input_dim=self.emb_dim,
            uniform_hid_dim=self.enc_lstm_hid_dim,
            lstm_layers=self.enc_lstm_layers,
            bidirectional=True,
            init_dropout=self.enc_dropouts[0],
            mid_dropout=self.enc_dropouts[-1]
        )
        # multihead attention
        self.mha = MultiheadCrossAttention(
            enc_out_dim=self.enc_lstm_hid_dim * 2,
            # using birectional encoder by default
            dec_out_dim=self.dec_lstm_out_dim,
            proj_dim=self.att_proj_dim,
            heads=self.att_heads,
            dropout=self.att_dropout
        )
        # decoder lstms
        self.dec_lstm = AutoRegDecoderLSTMCell(
            att_proj_dim=self.att_proj_dim,
            dec_emb_dim=self.emb_dim,
            dec_hid_dim=self.dec_lstm_hid_dim,
            dec_out_dim=self.dec_lstm_out_dim,
            dec_mid_dropout=self.dec_lstm_dropout
        )
        # classification head: assuming emb_dim = 2 * att_proj_dim
        self.cls = nn.Linear(self.emb_dim, self.vocab_size)
        # weight tying
        self.cls.weight = self.char_emb.weight

        # initial priors
        self.init_query = nn.Parameter(torch.rand((1, self.dec_lstm_out_dim)), requires_grad=True)
        self.init_hiddens = [(
            nn.Parameter(torch.zeros((1, self.dec_lstm_hid_dim)), requires_grad=True),
            nn.Parameter(torch.zeros((1, self.dec_lstm_hid_dim)), requires_grad=True)
        ), (
            nn.Parameter(torch.zeros((1, self.dec_lstm_out_dim)), requires_grad=True),
            nn.Parameter(torch.zeros((1, self.dec_lstm_out_dim)), requires_grad=True)
        )]


    def forward(self, x, lx, dec_y=None, tf_rate: float=1.0, init_force=None):
        # embedding
        x = self.char_emb(x)
        # encoder
        enc_h, enc_l = self.enc_lstm(x, lx)
        # dims
        batch_size, enc_max_len, enc_dim = enc_h.size()

        # update attention keys & vals
        self.mha.wrapup_encodings(enc_h, enc_l)

        # max dec length
        if self.training:
            gold_label_emb = self.char_emb(dec_y)
            dec_max_len = dec_y.size(-1)
        else:
            dec_max_len = self.CHR_MAX_STEPS

        # initial sos tokens
        chars = torch.full((batch_size, ), fill_value=self.CHR_SOS_IDX,
                          dtype=torch.long, device=x.device)

        # initial hidden
        hiddens = [
            [u.expand(batch_size, self.dec_lstm_hid_dim).to(x.device) 
             for u in self.init_hiddens[0]],
            [u.expand(batch_size, self.dec_lstm_out_dim).to(x.device) 
             for u in self.init_hiddens[1]]
        ]
        # initial queries & contexts
        queries = self.init_query.expand(batch_size, self.dec_lstm_out_dim).to(x.device)
        context, att_wgts = self.mha(queries, return_wgts=True)

        att_wgts_list = [att_wgts[0].detach().cpu()]
        pred_logits = list()

        for t in range(dec_max_len):
            # character embedding from last step
            char_emb = self.char_emb(chars)
            if self.training and t > 0:
                if torch.rand(1).item() <= tf_rate:
                    char_meb = gold_label_emb[:, t - 1, :]
            
            # update hidden states
            hiddens = self.dec_lstm(char_emb, context, hiddens)

            # update context
            context, att_wgts = self.mha(hiddens[-1][0], return_wgts=True)
            
            # obtaining output characters
            projected_queries = self.mha.queries.view(batch_size, -1)
            dec_out = torch.cat((projected_queries, context), dim=-1)
            char_logits = self.cls(dec_out)

            # greedy search
            chars = char_logits.argmax(-1)
            pred_logits.append(char_logits)
            att_wgts_list.append(att_wgts[0].detach().cpu())
        
        # concatenate all characters
        pred_logits = torch.stack(pred_logits, dim=1)
        att_wgts_list = torch.cat(att_wgts_list, dim=1).transpose(-2, -1)

        return pred_logits, att_wgts_list




def main(args):
    # find the device
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"\n\nRunning on [{device}]...\n")

    # load configs
    trncfgs_dict = yaml.safe_load(open(args.config_file))

    # add configs derived from constants
    trncfgs_dict['model']['configs']['vocab_size'] = len(VOCAB)
    trncfgs_dict['model']['configs']['CHR_SOS_IDX'] = VOCAB_MAP['<sos>']
    trncfgs_dict['model']['configs']['CHR_PAD_IDX'] = VOCAB_MAP['<eos>']
    # add vocabs to dict for inference use
    trncfgs_dict['VOCAB'] = VOCAB
    trncfgs_dict['VOCAB_MAP'] = VOCAB_MAP
    trncfgs_dict['EOS_IDX'] = trncfgs_dict['model']['configs']['CHR_PAD_IDX']
    trncfgs_dict['SOS_IDX'] = trncfgs_dict['model']['configs']['CHR_SOS_IDX']
    # transformation
    trncfgs = cfgClass(trncfgs_dict)

    # seed
    torch.manual_seed(trncfgs.seed)
    np.random.seed(trncfgs.seed)

    # exp folder
    tgt_folder = time.strftime("%Y%m%d-%H%M%S")[2:]
    if trncfgs.wandb.use:
        wandb.init(**trncfgs.wandb.configs, config=trncfgs)
        tgt_folder = wandb.run.name
    tgt_folder = f"{trncfgs.EXP_FOLDER}/{tgt_folder}"
    os.makedirs(tgt_folder, exist_ok=True)
    # save the configuration copy into the folder
    json.dump(trncfgs_dict, open(f"{tgt_folder}/config.json", 'w'), indent=4)
    # subdirectories
    for subdir in ('imgs', 'ckpts'):
        os.makedirs(f"{tgt_folder}/{subdir}", exist_ok=True)

    # load data
    trnDataset = lmDatasetTrainDev(
        trans_dir=trncfgs.TRN_FOLDER, 
        pred_dir=trncfgs.TRN_PRED_DIR,
        label_to_idx=VOCAB_MAP
    )
    trnLoader = DataLoader(
        trnDataset,
        batch_size=trncfgs.batch_size,
        num_workers=trncfgs.num_workers,
        collate_fn=trnDataset.collate_fn,
        shuffle=True,
        pin_memory=True
    )
    devDataset = lmDatasetTrainDev(
        trans_dir=trncfgs.DEV_FOLDER, 
        pred_dir=trncfgs.DEV_PRED_DIR,
        label_to_idx=VOCAB_MAP
    )
    devLoader = DataLoader(
        devDataset,
        batch_size=trncfgs.batch_size,
        num_workers=trncfgs.num_workers,
        collate_fn=devDataset.collate_fn
    )

    # model
    model = Rewriter(**trncfgs.model.configs)
    model.to(device)
    
    # model summary
    model.eval()
    x, _, lx, _ = next(iter(trnLoader))
    with torch.inference_mode():
        print(f"\n\nModel Summary: \n{summary(model, x.to(device), lx)}\n\n")
    
    # criterion
    criterion = nn.CrossEntropyLoss(reduction='none')

    # scaler
    scaler = torch.cuda.amp.GradScaler() if trncfgs.scaler.use else None

    # build trainer
    trainer = Trainer(
        model=model, vocab=VOCAB, trn_loader=trnLoader, dev_loader=devLoader,
        trncfgs=trncfgs, criterion=criterion, scaler=scaler, 
        tf_rate=trncfgs.tf_rate, saving_dir=tgt_folder, device=device, 
        accu_grad=trncfgs.accu_grad, grad_norm=trncfgs.grad_norm, 
        eval_ld_interval=trncfgs.eval_ld_interval,
        SOS_IDX=VOCAB_MAP['<sos>'], EOS_IDX=VOCAB_MAP['<eos>']
    )

    # train 
    trainer.train_eval(trncfgs.epochs)

    # save the results log
    json.dump([
        trainer.train, trainer.dev
    ], open(f"{tgt_folder}/log.json", 'w'), indent=4)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training Rewriter LM.")

    parser.add_argument(
        '--config-file',
        '-c',
        type=str,
        default='./config/rewriter.yml',
        help='filepath to the configuration file.'
    )

    args = parser.parse_args()
    main(args)