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
import argparse
import Levenshtein
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torchsummaryX import summary

from src.utils import *
from src.models import ListenAttendSpell
from src.constants import VOCAB, VOCAB_MAP


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
            milestone_dir,
            device,
            accu_grad: int=3,
            grad_norm: float=5.0,
            eval_ld_interval: int=4,
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
        self.last_tf_turn = (-1, float('inf'))
        self.accu_grad = accu_grad
        self.saving_dir = saving_dir
        self.milestone_dir = milestone_dir
        self.device = device
        self.accu_grad = accu_grad
        self.grad_norm = grad_norm
        self.eval_ld_interval = 4
        self.init_force = self.trncfgs.init_force
        self.SOS_IDX = SOS_IDX
        self.EOS_IDX = EOS_IDX
        # take vocab_size out for convenience
        self.vocab_size = self.model.spell.dec_vocab_size
        # # initiate the model & take to device
        # self.model.apply(self.init_function).to(device)
        
        # optimizer
        self.optimizer = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
        }[trncfgs.optimizer.name](
            self.model.parameters(), **trncfgs.optimizer.configs
        )
        # scheduler
        self.batch_scheduler = CosineAnnealingWithWarmup(
            self.optimizer, num_batches=len(self.trn_loader),
            **(self.trncfgs.batch_scheduler.configs)
        ) if self.trncfgs.batch_scheduler.use else None
        self.epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=3, mode='min'
        ) if self.trncfgs.epoch_scheduler.use else None
        # tf rate scheduler
        if self.trncfgs.tf_rate_scheduler.use:
            self.tf_configs = self.trncfgs.tf_rate_scheduler.configs
        # dropout scheduler
        if self.trncfgs.dropout_scheduler.use:
            self.dropout_configs = self.trncfgs.dropout_scheduler.configs

        # reset stats
        self.reset_stats()
        # loading checkpoints
        if self.trncfgs.finetune.use:
            self.load_model()
            self.reset_beststats()
        if self.trncfgs.finetune.reinit_lr:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.trncfgs.optimizer.configs['lr']


    def train_epoch(self):
        # mode change
        self.model.train()
        # total loss and ppl
        total_loss, total_ppl = 0, 0
        # batch bar
        batch_bar = tqdm(total=len(self.trn_loader), dynamic_ncols=True, leave=False, 
                         position=0, desc=f'training epoch[{self.epoch}]...')
        # whether to apply init diag attention
        init_force = self.init_force if self.epoch < 10 else False
        # loop
        for i, (x, y, lx, ly) in enumerate(self.trn_loader):
            # remove <sos> in the beginning of ys
            y, ly = y[:, 1:], ly - 1
            # dim stats
            batch_size, dec_max_len = y.size()
            # build masks
            y_mask = (torch.arange(0, dec_max_len, dtype=torch.int64)
                           .unsqueeze(0).expand(batch_size, dec_max_len))           # (batch_size, dec_max_len)
            y_mask = y_mask < y_mask.new(ly).unsqueeze(-1)                          # (batch_size, dec_max_len)
            y_mask = y_mask.to(self.device).flatten().to(torch.int)
            y_nonpadded_sum = y_mask.sum()
            # take to device
            x, y = x.to(self.device), y.to(self.device)
            # feed in the model
            if self.scaler:
                with torch.cuda.amp.autocast():
                    pred_logits, att_wgts = self.model(x, lx, y, self.tf_rate, init_force)
                    # compute loss
                    loss = (self.criterion(
                        pred_logits.view(-1, self.vocab_size),                      # (batch_size * dec_max_len, vocab_size)
                        y.view(-1)                                                  # (batch_size * dec_max_len)
                    ) * y_mask).sum() / (y_nonpadded_sum * self.accu_grad)
                    self.scaler.scale(loss).backward()
                    # perplexity
                    ppl = torch.exp(loss)
            else:
                pred_logits, att_wgts = self.model(x, lx, y, self.tf_rate, init_force)
                # compute loss
                loss = (self.criterion(pred_logits.view(-1, self.vocab_size), y.flatten())
                        * y_mask).sum() / (y_nonpadded_sum * self.accu_grad)
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
                """
                    TODO: check full logics of unscale_ implementation
                        https://github.com/pytorch/pytorch/blob/master/torch/cuda/amp/grad_scaler.py
                """
                # if math.isnan(grad_norm):
                #     print(f"Gradient seems to explode at [batch={self.batch}]. No params udpate.")
                # else:
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
                # remove <sos> in the beginning of ys
                y, ly = y[:, 1:], ly - 1
                # dims
                batch_size, dec_max_len = y.size()
                # masks
                y_mask = (torch.arange(0, dec_max_len, dtype=torch.int64)
                               .unsqueeze(0).expand(batch_size, dec_max_len))       # (batch_size, dec_max_len)
                y_mask = y_mask < y_mask.new(ly).unsqueeze(-1)                     # (batch_size, dec_max_len)
                y_mask = y_mask.to(self.device).flatten().to(torch.int)
                y_nonpadded_sum = y_mask.sum()
                # take to device
                x, y = x.to(self.device), y.to(self.device)
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        pred_logits, _ = self.model(x, lx, dec_y=None)
                        # compute loss
                        # truncating only the part matching ground truth
                        loss = (self.criterion(pred_logits[:, :dec_max_len, :].reshape(-1, self.vocab_size), y.flatten())
                                    * y_mask).sum() / y_nonpadded_sum
                else:
                    pred_logits, _ = self.model(x, lx, dec_y=None)
                    # compute loss
                    loss = (self.criterion(pred_logits[:, :dec_max_len, :].reshape(-1, self.vocab_size), y.flatten())
                                * y_mask).sum() / y_nonpadded_sum
                # perplexity
                ppl = torch.exp(loss)
                # total
                total_loss += loss.item()
                total_ppl += ppl.item()

                # greedy search
                pred_chars = pred_logits.argmax(dim=-1)
                # compute levenshtein distance
                # if self.eval_ld_interval == 1 or self.epoch % self.eval_ld_interval == 0:
                total_ld += self.batch_levenshtein(pred_chars, y, ly)

                # update batch_bar
                batch_bar.set_postfix(
                    avg_loss=f"{total_loss / (i + 1):.6f}",
                    avg_ppl=f"{total_ppl / (i + 1):.6f}",
                    avg_ld=f"{total_ld / (i + 1) if total_ld else -1:.6f}",
                )
                batch_bar.update()
            # finish
            batch_bar.close()
            del x, y, lx, ly
            torch.cuda.empty_cache()

        return (total_loss / len(self.dev_loader), total_ppl / len(self.dev_loader), 
                total_ld / len(self.dev_loader) if total_ld else -1)


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
            # visualize att_wgts map
            pay_attention_multihead(
                att_wgts=att_wgts, epoch=self.epoch, root_dir=f"{self.saving_dir}/imgs"
            )
            # add to records
            self.train['loss'].append(trn_loss)
            self.train['ppl'].append(trn_ppl)
            # evaluating
            dev_loss, dev_ppl, dev_ld = self.evaluate_epoch()
            # add to records
            self.dev['loss'].append(dev_loss)
            self.dev['ppl'].append(dev_ppl)
            if dev_ld <= 0:
                dev_ld = self.dev['ld'][-1]
            self.dev['ld'].append(dev_ld)
            # wandb logging
            if self.trncfgs.wandb.use:
                wandb.log({'avg_trn_loss': trn_loss, 'avg_trn_ppl': trn_ppl,
                           'dev_loss': dev_loss, 'dev_ppl': dev_ppl, 'dev_ld': dev_ld})
            # save model
            self.save_model()
            self.epoch += 1
            # epoch-level lr scheduling
            if self.dev['ld'][-1] <= 20 and self.epoch_scheduler:
                self.epoch_scheduler.step(dev_ld)
                if self.trncfgs.wandb.use:
                    wandb.log({'learning-rate': self.optimizer.param_groups[0]['lr']})


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
        save_for_rewriter = True if (self.epoch > 0 and (self.epoch + 1) % 10 == 0) else False
        epoch_record = {
            'epoch': self.epoch, 'batch': self.batch, 
            'loss': self.dev['loss'][-1], 'ld': self.dev['ld'][-1], 'ppl': self.dev['ppl'][-1]
        }
        # min loss
        if epoch_record['loss'] <= self.min_loss['loss']:
            # update results
            self.min_loss.update(epoch_record)
            tag += '-loss'
        if epoch_record['ld'] < self.min_ld['ld']:
            # update results
            self.min_ld.update(epoch_record)
            tag += '-ld'
        if epoch_record['ppl'] <= self.min_ppl['ppl']:
            # update results
            self.min_ppl.update(epoch_record)
            tag += '-ppl'
        # save model checkpoints (just once)
        if len(tag) > 3 or save_for_rewriter:
            # remove extra if exceeding
            if len(tag) > 3 and len(self.saved_epochs) >= self.trncfgs.max_savings:
                # remove the earliest model
                ckpts = os.listdir(f"{self.saving_dir}/ckpts")
                no = self.saved_epochs.pop(0)
                fn = [c for c in ckpts if c.endswith(f"epoch[{no}].pt")][0]
                os.remove(f"{self.saving_dir}/ckpts/{fn}")
            # save new
            epoch_record.update({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            })
            # add training & validation histories
            epoch_record.update({
                'train_loss': self.train['loss'], 'train_ppl': self.train['ppl'], 
                'dev_loss': self.dev['loss'], 'dev_ppl': self.dev['ppl'], 'dev_ld': self.dev['ld']
            })
            # saving best models
            if len(tag) > 3:
                torch.save(epoch_record, f"{self.saving_dir}/ckpts/{tag}-epoch[{self.epoch}].pt")
                self.saved_epochs.append(self.epoch)
                print(f"\nBest model saved: epoch[{self.epoch}], tag=[{tag}]\n")
            # saving milestone models
            if save_for_rewriter:
                torch.save(epoch_record, f"{self.milestone_dir}/epoch[{self.epoch}].pt")
                print(f"\nMilestone model archived: epoch[{self.epoch}]\n")
            
    

    def load_model(self):
        assert self.trncfgs.finetune.use
        assert os.path.exists(self.trncfgs.finetune.checkpoint)
        loaded = torch.load(self.trncfgs.finetune.checkpoint, map_location=torch.device(self.device))
        self.model.load_state_dict(loaded['model_state_dict'])
        self.optimizer.load_state_dict(loaded['optimizer_state_dict'])
        self.epoch = loaded['epoch']
        self.batch = loaded['batch']
        for d in (self.min_ld, self.min_loss, self.min_ppl):
            d['loss'] = loaded['loss']
            d['ppl'] = loaded['ppl']
            d['ld'] = loaded['ld']
        # historical records: if exists
        self.train['loss'] = loaded.get('train_loss', list())
        self.train['ppl'] = loaded.get('train_ppl', list())
        self.dev['loss'] = loaded.get('dev_loss', list())
        self.dev['ppl'] = loaded.get('dev_ppl', list())
        self.dev['ld'] = loaded.get('dev_ld', list())
        print(f"\n\nSuccessfully loaded from checkpoint [{self.trncfgs.finetune.checkpoint}]!")
        print(f"Resuming training from epoch[{self.epoch}]!\n\n")
    

    @staticmethod
    def init_function(layer):
        if isinstance(layer, nn.LSTM) or isinstance(layer, nn.LSTMCell):
            for p in layer.parameters():
                nn.init.uniform_(p.data, -0.1, 0.1)
        # elif isinstance(layer, nn.Linear):
        #     nn.init.xavier_uniform_(layer.weight)
        #     nn.init.zeros_(layer.bias)
        """
            uncomment the Linear initialization part if you want to fail
        """
    

    def batch_levenshtein(self, pred_chars, gold_chars, gold_lens):
        """
            compute avg. Levenshtein distance batch-wise
            Args:
                pred_chars: (batch_size, max_seq_len)
                gold_chars: (batch_size, max_seq_len)
                gold_lens: (batch_size, )
        """
        ld_total = 0
        for b in range(pred_chars.size(0)):
            pred_str = self.idx_to_str(pred_chars[b])
            gold_str = self.idx_to_str(gold_chars[b][:gold_lens[b]])
            ld_total += Levenshtein.distance(pred_str, gold_str)
        return ld_total / pred_chars.size(0)
    

    def batch_levenshtein_toy(self, pred_chars, gold_chars, gold_lens):
        ld_total = 0
        for b in range(pred_chars.size(0)):
            pred_str = self.idx_to_str(pred_chars[b], True)
            gold_str = self.idx_to_str(gold_chars[b][:gold_lens[b]], True)
            ld_total += Levenshtein.distance(pred_str, gold_str)
        return ld_total / pred_chars.size(0)


    def idx_to_str(self, idx_seq, return_list: bool=False):
        """
            Args:
                idx_seq: (max_seq_len, )
        """
        out_list = list()
        for idx in idx_seq:
            if idx == self.SOS_IDX:
                continue
            elif idx == self.EOS_IDX:
                break
            else:
                out_list.append(self.vocab[idx])
        return out_list if return_list else ''.join(out_list)
    

    def tf_rate_step(self):
        if (self.epoch > 0 
            and self.dev['ld'] and self.dev['ld'][-1] <= 20
            and self.tf_rate > self.tf_configs['lowest']):
            if (self.epoch - self.last_tf_turn[0] > self.tf_configs['interval']
                and self.dev['ld'][-1] < self.last_tf_turn[1]):
                # record this change
                self.tf_rate -= self.tf_configs['factor']
                self.last_tf_turn = (self.epoch, self.dev['ld'][-1])
    

    def dropout_step(self):
        if self.epoch in self.dropout_configs.keys():
            ratio = self.dropout_configs[self.epoch]
            print(f"\n\nChange the dropout rate by [{ratio}] in epoch [{self.epoch}]!\n\n")
        else:
            return
        # reset all dropouts
        # listener
        self.model.listen.init_dropout *= ratio
        self.model.listen.mid_dropout *= ratio
        self.model.listen.final_dropout *= ratio
        # speller
        self.model.spell.att_dropout *= ratio
        self.model.spell.attention.dropout *= ratio
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
    trncfgs_dict = yaml.safe_load(open(args.config_file))
    # toy dataset loading
    useMini = False

    # original vocabs
    from src.constants import VOCAB, VOCAB_MAP

    if trncfgs_dict['TRN_FOLDER'].startswith('mini'):
        useMini = True
        # LABELS
        dev_labels = np.load(f"{trncfgs_dict['TRN_FOLDER']}/dev_labels.npy")
        VOCAB_MAP           = dict(zip(np.unique(dev_labels), range(len(np.unique(dev_labels))))) 
        VOCAB_MAP["[PAD]"]  = len(VOCAB_MAP)
        VOCAB               = list(VOCAB_MAP.keys())
    # add configs derived from constants
    trncfgs_dict['model']['configs']['speller_configs']['dec_vocab_size'] = len(VOCAB)
    trncfgs_dict['model']['configs']['speller_configs']['CHR_SOS_IDX'] = VOCAB_MAP['[SOS]'] if useMini else VOCAB_MAP['<sos>']
    trncfgs_dict['model']['configs']['speller_configs']['CHR_PAD_IDX'] = VOCAB_MAP['[EOS]'] if useMini else VOCAB_MAP['<eos>']
    # add vocabs to dict for inference use
    trncfgs_dict['VOCAB'] = VOCAB
    trncfgs_dict['VOCAB_MAP'] = VOCAB_MAP
    trncfgs_dict['EOS_IDX'] = trncfgs_dict['model']['configs']['speller_configs']['CHR_PAD_IDX']
    trncfgs_dict['SOS_IDX'] = trncfgs_dict['model']['configs']['speller_configs']['CHR_SOS_IDX']
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
    os.makedirs(trncfgs.MST_FOLDER, exist_ok=True)
    # save the configuration copy into the folder
    json.dump(trncfgs_dict, open(f"{tgt_folder}/config.json", 'w'), indent=4)
    # subdirectories
    for subdir in ('imgs', 'ckpts'):
        os.makedirs(f"{tgt_folder}/{subdir}", exist_ok=True)

    if trncfgs.TRN_FOLDER.startswith('mini'):
        # data loading
        trnDataset = datasetTrainDevToy(
            root_dir=trncfgs.TRN_FOLDER, 
            subset='train',
            keep_tags=True, 
            label_to_idx=VOCAB_MAP,
            EOS_IDX=VOCAB_MAP['[EOS]'],
            use_specaug=trncfgs.use_specaug
        )
        devDataset = datasetTrainDevToy(
            root_dir=trncfgs.TRN_FOLDER, 
            subset='dev',
            keep_tags=True, 
            label_to_idx=VOCAB_MAP,
            EOS_IDX=VOCAB_MAP['[EOS]'],
            use_specaug=False
        )
        trnLoader = DataLoader(
            trnDataset,
            batch_size=trncfgs.batch_size,
            num_workers=trncfgs.num_workers,
            collate_fn=trnDataset.collate_fn,
            shuffle=True,
            pin_memory=True
        )
        devLoader = DataLoader(
            devDataset,
            batch_size=trncfgs.batch_size,
            num_workers=trncfgs.num_workers,
            collate_fn=devDataset.collate_fn,
        )
    else:
        # data loading
        trnDataset = datasetTrainDev(
            stdDir=trncfgs.TRN_FOLDER, 
            keepTags=True, 
            labelToIdx=VOCAB_MAP,
            useSpecAug=trncfgs.use_specaug
        )
        devDataset = datasetTrainDev(
            stdDir=trncfgs.DEV_FOLDER, 
            keepTags=True, 
            labelToIdx=VOCAB_MAP,
            useSpecAug=False
        )
        trnLoader = DataLoader(
            trnDataset,
            batch_size=trncfgs.batch_size,
            num_workers=trncfgs.num_workers,
            collate_fn=trnDataset.collate_fn,
            shuffle=True,
            pin_memory=True
        )
        devLoader = DataLoader(
            devDataset,
            batch_size=trncfgs.batch_size,
            num_workers=trncfgs.num_workers,
            collate_fn=devDataset.collate_fn,
        )
    
    print(f"\nA total of [{len(trnLoader)}] batches in training set, and [{len(devLoader)}] in dev set.\n")

    # model building
    model = ListenAttendSpell(**trncfgs.model.configs)
    model.to(device)
    
    # model summary
    show_computation_model_summary = True
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
        tf_rate=trncfgs.tf_rate, saving_dir=tgt_folder, milestone_dir=trncfgs.MST_FOLDER, 
        device=device, accu_grad=trncfgs.accu_grad, grad_norm=trncfgs.grad_norm, 
        eval_ld_interval=trncfgs.eval_ld_interval,
        SOS_IDX=VOCAB_MAP['[SOS]'] if trncfgs.TRN_FOLDER.startswith('mini') else VOCAB_MAP['<sos>'], 
        EOS_IDX=VOCAB_MAP['[EOS]'] if trncfgs.TRN_FOLDER.startswith('mini') else VOCAB_MAP['<eos>']
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
        default='./config/sample-attention.yml',
        help='filepath to the configuration file.'
    )

    args = parser.parse_args()
    main(args)