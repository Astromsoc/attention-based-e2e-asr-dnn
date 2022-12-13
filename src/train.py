"""
    Driver script for model training.
"""

import os
import json
import math
import time
import yaml
import torch
import wandb
import shutil
import argparse
import Levenshtein
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torchsummaryX import summary

from src.utils import *
from src.constants import *
from src.models import ListenAttendSpell



class Trainer:

    def __init__(
            self, 
            model,
            vocab,
            trn_loader,
            dev_loader,
            trncfgs,
            criterion,
            scaler,
            tf_rate,
            saving_dir,
            device,
            accu_grad: int=3,
            grad_norm: float=5.0,
            SOS_IDX: int=0,
            EOS_IDX: int=29
        ):
        self.model = model
        self.vocab = vocab
        self.trn_loader = trn_loader
        self.dev_loader = dev_loader
        self.trncfgs = trncfgs
        self.criterion = criterion
        self.scaler = scaler
        self.tf_rate = tf_rate
        self.accu_grad = accu_grad
        self.saving_dir = saving_dir
        self.device = device
        self.accu_grad = accu_grad
        self.grad_norm = grad_norm
        self.SOS_IDX = SOS_IDX
        self.EOS_IDX = EOS_IDX
        # take vocab_size out for convenience
        self.vocab_size = self.model.spell.dec_vocab_size
        # initiate the model & take to device
        self.model.apply(self.init_function).to(device)
        
        # optimizer
        self.optimizer = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
        }[trncfgs.optimizer.name](
            self.model.parameters(), **trncfgs.optimizer.configs
        )
        # scheduler
        self.batch_scheduler = (CosineAnnealingWithWarmup(
            self.optimizer, num_batches=len(self.trn_loader),
            **(self.trncfgs.batch_scheduler.configs)
        ) if self.trncfgs.batch_scheduler.use else None)
        self.epoch_scheduler = None
        # tf rate scheduler
        if self.trncfgs.tf_rate_scheduler.use:
            self.tf_configs = self.trncfgs.tf_rate_scheduler.configs
        # dropout scheduler
        if self.trncfgs.dropout_scheduler.use:
            self.dropout_configs = self.trncfgs.dropout_scheduler.configs

        # reset stats
        self.reset_stats()


    def train_epoch(self):
        # mode change
        self.model.train()
        # total loss and ppl
        total_loss, total_ppl = 0, 0
        # batch bar
        batch_bar = tqdm(total=len(self.trn_loader), dynamic_ncols=True, leave=False, 
                         position=0, desc=f'training epoch[{self.epoch}]...')
        # loop
        for i, (x, y, lx, ly) in enumerate(self.trn_loader):
            # dim stats
            batch_size, dec_max_len = y.size()
            # build masks
            y_mask = (torch.arange(0, dec_max_len, dtype=torch.int64)
                           .unsqueeze(0).expand(batch_size, dec_max_len))           # (batch_size, dec_max_len)
            y_mask = y_mask >= y_mask.new(ly).unsqueeze(-1)                         # (batch_size, dec_max_len)
            y_mask = y_mask.to(self.device).flatten()
            # take to device
            x, y = x.to(self.device), y.to(self.device)
            # feed in the model
            if self.scaler:
                with torch.cuda.amp.autocast():
                    pred_logits, att_wgts = self.model(x, lx, y, self.tf_rate)
                    # compute loss
                    loss = self.criterion(
                        pred_logits.view(-1, self.vocab_size),                      # (batch_size * dec_max_len, vocab_size)
                        y.view(-1)                                                  # (batch_size * dec_max_len)
                    ).masked_fill(y_mask, 0).mean() / self.accu_grad
                    self.scaler.scale(loss).backward()
                    # perplexity
                    ppl = torch.exp(loss)
            else:
                pred_logits, att_wgts = self.model(x, lx, y, self.tf_rate)
                # compute loss
                loss = (self.criterion(pred_logits.view(-1, self.vocab_size), y.flatten())
                            .masked_fill(y_mask, 0)
                            .mean() / self.accu_grad)
                loss.backward()
                # perplexity
                ppl = torch.exp(loss)
            # show
            total_loss += loss.item()
            total_ppl += ppl.item() 
            # update batch bar
            batch_bar.set_postfix(
                avg_loss=f"{total_loss / (i + 1):.6f}",
                avg_ppl=f"{total_ppl / (i + 1):.6f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.4f}",
                tf_rate=f"{self.tf_rate:.2f}"
            )
            batch_bar.update()
            
            """
                accumulated gradient update
            """
            if (i + 1) % self.accu_grad == 0:
                # unscale optimizer
                self.scaler.unscale_(self.optimizer)
                # clip gradient norm
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm
                )
                if math.isnan(grad_norm):
                    print(f"Gradient seems to explode[batch={self.batch}]. No params udpate.")
                else:
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                # clear and start over
                self.optimizer.zero_grad()
                # learning rate updates
                if self.batch_scheduler:
                    self.batch_scheduler.step()
                    if self.trncfgs.wandb.use:
                        wandb.log({'learning-rate': self.optimizer.param_groups[0]['lr']})
            # increase batch count
            self.batch += 1
        # finish
        batch_bar.close()
        del x, y, lx, ly
        torch.cuda.empty_cache()

        return total_loss / (i + 1), total_ppl / (i + 1), att_wgts
    

    def evaluate_epoch(self):
        # mode
        self.model.eval()
        # stats
        total_loss, total_ppl, total_ld = 0, 0, 0
        # batch bar
        batch_bar = tqdm(total=len(self.dev_loader), dynamic_ncols=True, leave=False, 
                         position=0, desc=f'evaluating epoch[{self.epoch}]...')
        with torch.inference_mode():
            for i, (x, y, lx, ly) in enumerate(self.dev_loader):
                # dims
                batch_size, dec_max_len = y.size()
                # masks
                y_mask = (torch.arange(0, dec_max_len, dtype=torch.int64)
                               .unsqueeze(0).expand(batch_size, dec_max_len))       # (batch_size, dec_max_len)
                y_mask = y_mask >= y_mask.new(ly).unsqueeze(-1)                     # (batch_size, dec_max_len)
                y_mask = y_mask.to(self.device).flatten()
                # take to device
                x, y = x.to(self.device), y.to(self.device)
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        pred_logits, _ = self.model(x, lx, dec_y=None)
                        # compute loss
                        # truncating only the part matching ground truth
                        loss = (self.criterion(pred_logits[:, :dec_max_len, :].reshape(-1, self.vocab_size), y.flatten())
                                    .masked_fill(y_mask, 0).mean())
                else:
                    pred_logits, _ = self.model(x, lx, dec_y=None)
                    # compute loss
                    loss = (self.criterion(pred_logits[:, :dec_max_len, :].reshape(-1, self.vocab_size), y.flatten())
                                .masked_fill(y_mask, 0).mean())
                # perplexity
                ppl = torch.exp(loss)
                # total
                total_loss += loss.item()
                total_ppl += ppl.item()

                # greedy search
                pred_chars = self.greedy_search_stepwise(pred_logits)
                # compute levenshtein distance
                total_ld += self.batch_levenshtein(pred_chars, y)

                # update batch_bar
                batch_bar.set_postfix(
                    avg_loss=f"{total_loss/(i + 1):.6f}",
                    avg_ppl=f"{total_ppl/(i + 1):.6f}",
                    avg_ld=f"{total_loss/(i + 1):.6f}",
                )
                batch_bar.update()
            # finish
            batch_bar.close()
            del x, y, lx, ly
            torch.cuda.empty_cache()

        return total_loss / (i + 1), total_ppl / (i + 1), total_ld / (i + 1)


    def train_eval(self, epochs: int):
        # check if the input epochs is larger than records
        while self.epoch < epochs:
            # tf rate / dropout scheduling
            if self.trncfgs.tf_rate_scheduler.use:
                self.tf_rate_step()
            if self.trncfgs.dropout_scheduler.use:
                self.dropout_step()
            # training
            trn_loss, trn_ppl, att_wgts = self.train_epoch()
            # add to records
            self.train['loss'].append(trn_loss)
            self.train['ppl'].append(trn_ppl)
            # evaluating
            dev_loss, dev_ppl, dev_ld = self.evaluate_epoch()
            # add to records
            self.dev['loss'].append(dev_loss)
            self.dev['ppl'].append(dev_ppl)
            self.dev['ld'].append(dev_ld)
            # wandb logging
            if self.trncfgs.wandb.use:
                wandb.log({'avg_trn_loss': trn_loss, 'avg_trn_ppl': trn_ppl,
                           'dev_loss': dev_loss, 'dev_ppl': dev_ppl, 'dev_ld': dev_ld,})
            # save model
            self.save_model()
            self.epoch += 1
            # epoch-level lr scheduling
            if self.epoch_scheduler:
                self.epoch_scheduler.step()
                if self.trncfgs.wandb.use:
                    wandb.log({'learning-rate': self.optimizer.param_groups[0]['lr']})
            # visualize att_wgts map
            pay_attention_multihead(
                att_wgts=att_wgts, epoch=self.epoch, root_dir=f"{self.saving_dir}/imgs"
            )


    def reset_stats(self):
        self.epoch = 0
        self.batch = 0
        self.train = {
            'epoch': 0, 'batch': 0, 
            'loss': list(), 'ppl': list()
        }
        self.dev = {
            'epoch': 0, 'batch': 0, 
            'loss': list(), 'ld': list(), 'ppl': list()
        }
        self.reset_beststats()
    

    def reset_beststats(self):
        self.min_loss = {'epoch': 0, 'batch': 0, 'loss': float('inf'), 'ld': float('inf'), 'ppl': float('inf')}
        self.min_ld = {'epoch': 0, 'batch': 0, 'loss': float('inf'), 'ld': float('inf'), 'ppl': float('inf')}
        self.min_ppl = {'epoch': 0, 'batch': 0, 'loss': float('inf'), 'ld': float('inf'), 'ppl': float('inf')}
        self.saved_epochs = list()


    def save_model(self):
        tag = 'min'
        epoch_record = {
            'epoch': self.epoch, 'batch': self.batch, 
            'loss': self.dev['loss'][-1], 'ld': self.dev['ld'][-1], 'ppl': self.dev['ppl'][-1]
        }
        # min loss
        if epoch_record['loss'] <= self.min_loss['loss']:
            # update results
            self.min_loss.update(epoch_record)
            tag += '-loss'
        elif epoch_record['ld'] <= self.min_ld['ld']:
            # update results
            self.min_ld.update(epoch_record)
            tag += '-ld'
        elif epoch_record['ppl'] <= self.min_ppl['ppl']:
            # update results
            self.min_ppl.update(epoch_record)
            tag += '-ppl'
        # save model checkpoints (just once)
        if len(tag) > 3:
            # remove extra if exceeding
            if len(self.saved_epochs) >= self.trncfgs.max_savings:
                # remove the earliest model
                ckpts = os.listdir(f"{self.saving_dir}/ckpts")
                no = self.saved_epochs.pop(0)
                fn = [c for c in ckpts if c.endswith(f"epoch[{no}].pt")][0]
                os.remove(f"{self.saving_dir}/ckpts/{fn}")
            # save new
            torch.save(epoch_record | {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, f"{self.saving_dir}/ckpts/minloss-epoch[{self.epoch}].pt")
            self.saved_epochs.append(self.epoch)
            print(f"\nBest model saved: epoch[{self.epoch}], tag=[{tag}]\n")
    

    @staticmethod
    def init_function(layer):
        if isinstance(layer, nn.LSTM) or isinstance(layer, nn.LSTMCell):
            for p in layer.parameters():
                nn.init.uniform_(p.data, -0.1, 0.1)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
    

    @staticmethod
    def greedy_search_stepwise(logits_tensor):
        """
            greedy search for logits tensor per time step
            Args:
                logits_tensor: (batch_size, seq_len, vocab_size)
        """
        return logits_tensor.argmax(dim=-1)
    

    def batch_levenshtein(self, pred_chars, gold_chars, return_avg: bool=False):
        """
            compute avg. Levenshtein distance batch-wise
            Args:
                pred_chars: (batch_size, max_seq_len)
                gold_chars: (batch_size, max_seq_len)
                return_avg: (bool) whether to divide by batch size
        """
        ld_total = 0
        for b in range(pred_chars.size(0)):
            pred_str = self.idx_to_str(pred_chars[b])
            gold_str = self.idx_to_str(gold_chars[b])
            ld_total += Levenshtein.distance(pred_str, gold_str)
        return ld_total / b if return_avg else ld_total


    def idx_to_str(self, idx_seq):
        """
            Args:
                idx_seq: (max_seq_len, )
        """
        out_str = ''
        for idx in idx_seq:
            if idx == self.SOS_IDX:
                continue
            elif idx == self.EOS_IDX:
                break
            else:
                out_str += self.vocab[idx]
        return out_str
    

    def tf_rate_step(self):
        if self.epoch > self.tf_configs['high_epochs']:
            self.tf_rate = np.random.uniform(low=self.tf_configs['min_val'])
    

    def dropout_step(self):
        if self.epoch in self.dropout_configs.keys():
            ratio = self.dropout_configs[self.epoch]
        else:
            return
        # reset all dropouts
        # listener
        self.model.listen.init_dropout *= ratio
        self.model.listen.mid_dropout *= ratio
        self.model.listen.final_dropout *= ratio
        # speller
        self.model.spell.att_dropout *= ratio
        self.model.spell.dec_emb_dropout *= ratio
        self.model.spell.dec_lstm_dropout *= ratio



def main(args):
    # find the device
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"\n\nRunning on [{device}]...\n")

    # load configs
    trncfgs = cfgClass(yaml.safe_load(open(args.config_file)))

    # seed
    torch.manual_seed(trncfgs.seed)
    np.random.seed(trncfgs.seed)

    # exp folder
    tgt_folder = time.strftime("%Y%m%d-%H%M%S")[2:]
    if trncfgs.wandb.use:
        wandb.init(**trncfgs.wandb.configs, config=trncfgs)
        tgt_folder = wandb.run.names
    tgt_folder = f"{trncfgs.EXP_FOLDER}/{tgt_folder}"
    os.makedirs(tgt_folder, exist_ok=True)
    # save the configuration copy into the folder
    shutil.copy(args.config_file, f"{tgt_folder}/config.yml")
    # subdirectories
    for subdir in ('imgs', 'ckpts'):
        os.makedirs(f"{tgt_folder}/{subdir}", exist_ok=True)

    # data loading
    trnDataset = datasetTrainDev(
        stdDir=trncfgs.TRN_FOLDER, 
        keepTags=True, 
        labelToIdx=VOCAB_MAP
    )
    devDataset = datasetTrainDev(
        stdDir=trncfgs.DEV_FOLDER, 
        keepTags=True, 
        labelToIdx=VOCAB_MAP
    )
    trnLoader = DataLoader(
        trnDataset,
        batch_size=trncfgs.batch_size,
        num_workers=trncfgs.num_workers,
        collate_fn=collate_train_dev,
        shuffle=True,
        pin_memory=True
    )
    devLoader = DataLoader(
        devDataset,
        batch_size=trncfgs.batch_size,
        num_workers=trncfgs.num_workers,
        collate_fn=collate_train_dev
    )
    print(f"\nA total of [{len(trnLoader)}] batches in training set, and [{len(devLoader)}] in dev set.\n")

    # model building
    trncfgs.model.configs['speller_configs']['dec_vocab_size'] = len(VOCAB)
    model = ListenAttendSpell(**trncfgs.model.configs)
    model.to(device)
    
    # model summary
    show_computation_model_summary = False
    if show_computation_model_summary:
        model.eval()
        x, _, lx, _ = next(iter(trnLoader))
        with torch.inference_mode():
            print(f"\n\nModel Summary: \n{summary(model, x.to(device), lx)}\n\n")
    else:
        print(f"\n\nTruncated Model Summary:\n{model}\n\n")

    # criterion
    criterion = nn.CrossEntropyLoss(reduction='none')

    # scaler
    scaler = torch.cuda.amp.GradScaler() if trncfgs.scaler.use else None

    # trainer class
    trainer = Trainer(
        model=model, vocab=VOCAB, trn_loader=trnLoader, dev_loader=devLoader,
        trncfgs=trncfgs, criterion=criterion, scaler=scaler, 
        tf_rate=trncfgs.tf_rate, saving_dir=tgt_folder, device=device, 
        accu_grad=trncfgs.accu_grad, grad_norm=trncfgs.grad_norm, 
        SOS_IDX=VOCAB_MAP['<sos>'], EOS_IDX=VOCAB_MAP['<eos>']
    )

    # train 
    trainer.train_eval(trncfgs.epochs)

    # save the results log
    json.dump([
        trainer.train, trainer.dev
    ], open(f"{tgt_folder}/log.json", 'w'), indent=4)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training E2E Attention-Based ASR. (LAS)")

    parser.add_argument(
        '--config-file',
        '-c',
        type=str,
        default='./config/sample_attention.yml',
        help='filepath to the configuration file.'
    )

    args = parser.parse_args()
    main(args)