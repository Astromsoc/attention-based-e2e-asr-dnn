"""
    Driver script for model training.
"""

import os
import math
import torch
import wandb
import argparse
import Levenshtein

from tqdm import tqdm
from src.utils import *
from src.models import ListenAttendSpell



class Trainer:

    def __init__(
            self, 
            model,
            vocab,
            trn_loader,
            val_loader,
            criterion,
            optimizer,
            batch_scheduler,
            epoch_scheduler,
            scaler,
            tf_rate,
            saving_dir,
            device,
            accu_grad: int=3,
            grad_norm: float=5.0,
            SOS_IDX: 0,
            EOS_IDX: 29
        ):
        self.model = model
        self.vocab = vocab
        self.trn_loader = trn_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_scheduler = batch_scheduler
        self.epoch_scheduler = epoch_scheduler
        self.scaler = scaler
        self.tf_rate = tf_rate
        self.accu_grad = accu_grad
        self.saving_dir = saving_dir
        self.device = device
        # take vocab_size out for convenience
        self.vocab_size = self.model.spell.dec_vocab_size
        # initiate the model & take to device
        self.model.apply(self.init_function).to(device)
    

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
            y_mask = (torch.arange(0, dec_max_len, dtype=torch.int64, device=self.device)
                           .unsqueeze(0).expand(batch_size, dec_max_len))           # (batch_size, dec_max_len)
            y_mask = y_mask >= y_mask.new(ly).unsqueeze(-1)                         # (batch_size, dec_max_len)
            # take to device
            x, y = x.to(self.device), y.to(self.device)
            # feed in the model
            if self.scaler:
                with torch.cuda.amp.autocast():
                    pred_logits, att_wgts = self.model(x, lx, y, self.tf_rate)
                    # compute loss
                    loss = self.criterion(
                        pred_logits.view(-1, self.vocab_size),                           # (batch_size * dec_max_len, vocab_size)
                        y.flatten()                                                 # (batch_size * dec_max_len)
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
                lr=f"{self.optimizer.param_groups[0]['lr'].item():.4f}",
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
                    self.model.parameters, self.grad_norm
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
            # increase batch count
            self.batch += 1
        # finish
        batch_bar.close()
        del x, y, lx, ly
        torch.cuda.empty_cache()

        return total_loss / (i + 1), total_ld / (i + 1)
    

    def evaluate_epoch(self):
        # mode
        self.model.eval()
        # stats
        total_loss, total_ppl = 0, 0
        # batch bar
        batch_bar = tqdm(total=len(self.val_loader), dynamic_ncols=True, leave=False, 
                         position=0, desc=f'evaluating epoch[{self.epoch}]...')
        with torch.inference_mode():
            for i, (x, y, lx, ly) in enumerate(self.val_loader):
                # dims
                batch_size, dec_max_len = y.size()
                # masks
                y_mask = (torch.arange(0, dec_max_len, dtype=torch.int64, device=self.device)
                               .unsqueeze(0).expand(batch_size, dec_max_len))       # (batch_size, dec_max_len)
                y_mask = y_mask >= y_mask.new(ly).unsqueeze(-1)                     # (batch_size, dec_max_len)
                # take to device
                x, y = x.to(self.device), y.to(self.device)
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        pred_logits, _ = self.model(x, lx, y=None)
                        # compute loss
                        loss = (self.criterion(pred_logits.view(-1, self.vocab_size), y.flatten())
                                    .masked_fill(y_mask, 0).mean())
                else:
                    pred_logits, _ = self.model(x, lx, y=None)
                    # compute loss
                    loss = (self.criterion(pred_logits.view(-1, self.vocab_size), y.flatten())
                                .masked_fill(y_mask, 0).mean())
                # perplexity
                ppl = torch.exp(loss)
                # total
                total_loss += loss.item()
                total_ppl += ppl.item()

                # greedy search
                pred_chars = self.greedy_search_tensor(pred_logits)
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


    def train(self, epochs: int):
        # check if the input epochs is larger than records
        while self.epoch < epochs:
            # training
            trn_loss, trn_ppl = self.train_epoch()
            # add to records
            self.train['loss'].append(trn_loss)
            self.train['ppl'].append(trn_ppl)
            # evaluating
            val_loss, val_ppl, val_ld = self.evaluate_epoch()
            # add to records
            self.val['loss'].append(val_loss)
            self.val['ppl'].append(val_ppl)
            self.val['ld'].append(val_ld)
            # save model
            self.save_model()
            self.epoch += 1
            if self.epoch_scheduler:
                self.epoch_scheduler.step()


    def reset_stats(self):
        self.epoch = 0
        self.batch = 0
        self.train = {
            'epoch': 0, 'batch': 0, 
            'loss': list(), 'ppl': list()
        }
        self.val = {
            'epoch': 0, 'batch': 0, 
            'loss': list(), 'ld': list(), 'ppl': list()
        }
        self.reset_beststats()
    

    def reset_beststats(self):
        self.min_loss = {'epoch': 0, 'batch': 0, 'loss': 0, 'ld': 0, 'ppl': 0}
        self.min_ld = {'epoch': 0, 'batch': 0, 'loss': 0, 'ld': 0, 'ppl': 0}
        self.min_ppl = {'epoch': 0, 'batch': 0, 'loss': 0, 'ld': 0, 'ppl': 0}


    def save_model(self):
        tag = 'min'
        # min loss
        if self.val['loss'] <= self.min_loss['loss']:
            # update results
            self.min_loss.update(self.val)
            tag += '-loss'
        elif self.val['ld'] <= self.min_ld['ld']:
            # update results
            self.min_ld.update(self.val)
            tag += '-ld'
        elif self.val['ppl'] <= self.min_ppl['ppl']:
            tag += 'ppl'
        # save model checkpoints (just once)
        if len(tag) > 3:
            torch.save({
                'epoch': self.epoch, 
                'batch': self.batch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.val['loss'], 
                'ld': self.val['ld'],
                'ppl': self.val['ppl'],
            }, f"{self.saving_dir}/minloss-epoch[{self.epoch}].pt")
    

    @staticmethod
    def init_function(layer):
        if isinstance(layer, nn.LSTMCell) or isinstance(layer, nn.LSTM):
            nn.init.uniform_(layer.weight, -0.1, 0.1)
            nn.init.zero_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    

    @staticmethod
    def greedy_search_tensor(logits_tensor):
        """
            greedy search for logits tensor per time step
            Args:
                logits_tensor: (batch_size, seq_len, vocab_size)
        """
        return logits_tensor[:, :, -1].argmax(dim=-1)
    

    @staticmethod
    def batch_levenshtein(pred_chars, gold_chars, return_avg: bool=False):
        """
            compute avg. Levenshtein distance batch-wise
            Args:
                pred_chars: (batch_size, max_seq_len)
                gold_chars: (batch_size, max_seq_len)
                return_avg: (bool) whether to divide by batch size
        """
        ld_total = 0
        for b in range(pred_chars.size(0)):
            pred_str = self.idx_to_str([pred_chars[b]])
            gold_str = self.idx_to_str([gold_chars[b]])
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





def main(args):
    # find the device
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"\n\nRunning on [{device}]...\n")







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