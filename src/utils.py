"""
    Utility classes and functions.
    Partially imported from my implementation in other repo:
        https://github.com/Astromsoc/seq-to-seq-auto-phoneme-recognition/blob/master/src/utils.py
"""


import os
import math
import numpy as np
import seaborn as sns
from tqdm import tqdm
from Levenshtein import distance
from matplotlib import pyplot as plt

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader



class cfgClass(object):
    """
        Convert configuration dictionary into a simple class instance.
    """
    def __init__(self, cfg_dict: dict):
        # initial buildup
        self.__dict__.update(cfg_dict)
        for k, v in self.__dict__.items():
            if not k.endswith('configs') and isinstance(v, dict):
                self.__dict__.update({k: cfgClass(v)})



class datasetTrainDev(Dataset):
    """
        Dataloader for training and dev sets: both features & labels
    """
    def __init__(
        self, 
        mfccDir: str=None, transDir: str=None, stdDir: str=None, 
        labelToIdx: dict=None, keepTags: bool=False
    ):
        # using default structure (./mfcc + ./transcript/raw)
        if stdDir:
            mfccDir = f"{stdDir}/mfcc"
            transDir = f"{stdDir}/transcript/raw"
        # bookkeeping
        self.mfccDir = mfccDir
        self.transDir = transDir
        self.labelToIdx = labelToIdx
        self.keepTags = keepTags
        
        # load all filenames
        mfccFNs = sorted([
            f"{mfccDir}/{f}" for f in os.listdir(mfccDir) if f.endswith('.npy')
        ])
        transFNs = sorted([
            f"{transDir}/{f}" for f in os.listdir(transDir) if f.endswith('.npy')
        ])

        # whether to keep <sos> / <eos>
        if not self.keepTags: 
            l, r = 1, -1

        # load files
        self.mfccs = [
            torch.from_numpy(np.load(f)) 
            for f in tqdm(mfccFNs, leave=False, desc='loading mfccs...') if f.endswith('.npy')
        ]
        self.transcripts = [
            torch.tensor([self.labelToIdx[p] for p in (np.load(f) if self.keepTags else np.load(f)[l: r])]) 
            for f in tqdm(transFNs, leave=False, desc='loading transcripts...') if f.endswith('.npy')
        ]

        # dataset size
        self.size = len(self.mfccs)

    
    def __len__(self):
        return self.size

    
    def __getitem__(self, index):
        return self.mfccs[index], self.transcripts[index]



class datasetTest(Dataset):
    """
        Dataset for test set: only features
    """
    def __init__(
        self, 
        mfccDir: str=None
    ):
        # bookkeeping
        self.mfccDir = mfccDir

        # load all filenames
        mfccFNs = sorted([
            f"{mfccDir}/{f}" for f in os.listdir(mfccDir) if f.endswith('.npy')
        ])
        # load files
        self.mfccs = [
            torch.from_numpy(np.load(f)) 
            for f in tqdm(mfccFNs, leave=False, desc='loading mfccs...') 
            if f.endswith('.npy')
        ]
        # dataset size
        self.size = len(self.mfccs)
    

    def __len__(self):
        return self.size
    

    def __getitem__(self, index):
        return self.mfccs[index]



def collate_train_dev(
        batch, mfcc_padding=0, trans_padding=0
    ):
    """
        Collate function for training and dev sets, 4 returns
    """
    mfccs = [u[0] for u in batch]
    transcripts = [u[1] for u in batch]

    # sort the mfccs given lengths
    idx = sorted(np.arange(len(mfccs)), key=lambda x: len(mfccs[x]), reverse=True)
    mfccs = [mfccs[i] for i in idx]
    transcripts = [transcripts[i] for i in idx]

    # obtain original lengths for both mfccs & transcripts
    mfcc_lens = [len(m) for m in mfccs]
    transcript_lens = [len(t) for t in transcripts]


    # pad both mfccs & transcripts
    mfccs = pad_sequence(
        mfccs, batch_first=True, padding_value=mfcc_padding
    )
    transcripts = pad_sequence(
        transcripts, batch_first=True, padding_value=trans_padding
    )
    return mfccs, transcripts, torch.tensor(mfcc_lens), torch.tensor(transcript_lens)



def collate_test(
        batch, mfcc_padding=0
    ):
    """
        Collate function for test set: 2 returns
    """
    mfccs = batch
    # obtain original lengths
    mfcc_lens = torch.tensor([len(m) for m in mfccs])
    # pad 
    mfccs = pad_sequence(
        mfccs, batch_first=True, padding_value=mfcc_padding
    )
    return mfccs, mfcc_lens



class CosineAnnealingWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_batches,
                 warmup_epochs=1, max_epochs=10,
                 init_lr=0.001, min_lr=1e-6):
        self.optimizer = optimizer
        # bookkeeping
        self.num_batches = num_batches
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.init_lr = init_lr
        self.min_lr = min_lr
        # build up
        self.build_schedule()
        # step counter
        self.step_count = 0
        
    def build_schedule(self):
        total_batches = self.num_batches * self.max_epochs
        self.total_batches = total_batches
        self.lr_list = np.zeros((total_batches, ))
        warmup_batches = int(self.num_batches * self.warmup_epochs)
        # linear warmup
        self.lr_list[:warmup_batches] = np.linspace(
            self.min_lr, self.init_lr, warmup_batches
        )
        # cosine annealing
        left_batches = total_batches - warmup_batches
        self.lr_list[warmup_batches:] = np.array([self.min_lr + 
            (self.init_lr - self.min_lr) * math.cos(i * math.pi/ left_batches)
            for i in range(left_batches)
        ])
    
    def step(self):
        for group in self.optimizer.param_groups:
            group['lr'] = (self.lr_list[self.step_count] 
                           if self.step_count <= self.total_batches
                           else self.min_lr)
        self.step_count += 1



def compute_levenshtein(h, y, lh, ly, decoder, LABELS):
    # decode the output (taking the best output from beam search)
    # h <- (batch, seq_len, n_labels)
    beam_results, _, _, out__lens = decoder.decode(h, lh)
    total_dist = 0
    batchSize = len(beam_results)
    for b in range(batchSize):
        pred_str = ''.join(LABELS[l] for l in beam_results[b, 0, :out__lens[b, 0]])
        true_str = ''.join(LABELS[l] for l in y[b, :ly[b]])
        total_dist += distance(pred_str, true_str)
    return total_dist / batchSize



def generate_batch_predictions(h, lh, decoder, LABELS):
    # decode the output (taking the best output from beam search)
    # h <- (batch, seq_len, n_labels)
    beam_results, _, _, out__lens = decoder.decode(h, lh)
    return [
        ''.join(LABELS[l] for l in beam_results[b, 0, :out__lens[b, 0]])
        for b in range(len(beam_results))
    ]



def pay_attention_multihead(att_wgts, epoch: int, root_dir: str='.'):
    """
        Args:
            att_wgts: (num_heads, char_len, enc_len) 
    """
    att_wgts = att_wgts.transpose((0, 2, 1))
    num_heads = att_wgts.shape[0]
    n_rows = int(math.sqrt(num_heads))
    n_cols = num_heads // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    fig.suptitle(f"Attention Map [Epoch={epoch}]")
    fig.supxlabel('Output Character Count')
    fig.supylabel('Compressed Input Frame Count')
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            sns.heatmap(att_wgts[i], cmap='coolwarm', ax=axes[r, c])
            axes[r, c].set_title(f"Attention Head #[{i}]")
    img_fp = f"{root_dir}/attention-map-epoch{epoch}.png"
    fig.savefig(img_fp, dpi=128)
    print(f"\nAttention map successfully saved to [{img_fp}].\n")
