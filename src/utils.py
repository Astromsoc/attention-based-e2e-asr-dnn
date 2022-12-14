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
from torchaudio import transforms as tat

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
        labelToIdx: dict=None, keepTags: bool=False, useSpecAug: bool=False
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
        self.useSpecAug = useSpecAug
        
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

        # transforms
        if self.useSpecAug:
            self.freq_masker = tat.FrequencyMasking(2)
            self.time_masker = tat.TimeMasking(30)

    
    def __len__(self):
        return self.size

    
    def __getitem__(self, index):
        return self.mfccs[index], self.transcripts[index]

    
    def collate_fn(
        self, batch, mfcc_padding=0, trans_padding=0, 
        freq_transforms=False, 
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
        
        # apply augmentation
        if self.use_specaug:
            x_batch_pad = self.time_masker(self.time_masker(self.time_masker(
                        self.freq_masker(x_batch_pad)
            )))
        
        return mfccs, transcripts, torch.tensor(mfcc_lens), torch.tensor(transcript_lens)



class datasetTest(Dataset):
    """
        Dataset for test set: only features
    """
    def __init__(
        self, 
        stdDir: str=None
    ):
        # bookkeeping
        self.stdDir = stdDir
        self.mfccDir = f"{self.stdDir}/mfcc"

        # load all filenames
        mfccs = sorted([
            f"{self.mfccDir}/{f}" for f in os.listdir(self.mfccDir) if f.endswith('.npy')
        ])
        idx = sorted(np.arange(len(mfccs)), key=lambda x: len(mfccs[x]), reverse=True)
        mfccs = [mfccs[i] for i in idx]
        
        # load files
        self.mfccs = [
            torch.from_numpy(np.load(f)) 
            for f in tqdm(mfccs, leave=False, desc='loading mfccs...') 
            if f.endswith('.npy')
        ]
        # dataset size
        self.size = len(self.mfccs)
    

    def __len__(self):
        return self.size
    

    def __getitem__(self, index):
        return self.mfccs[index]
    
    
    def collate_fn(
            self, batch, mfcc_padding=0
        ):
        """
            Collate function for test set: 2 returns
        """
        mfccs = batch
        # obtain original lengths
        mfcc_lens = torch.tensor([len(m) for m in mfccs])
        # sort the mfccs given lengths
        idx = sorted(np.arange(len(mfccs)), key=lambda x: len(mfccs[x]), reverse=True)
        mfccs = [mfccs[i] for i in idx]
        # pad 
        mfccs = pad_sequence(
            mfccs, batch_first=True, padding_value=mfcc_padding
        )
        return mfccs, mfcc_lens



class datasetTrainDevToy(Dataset):
    """
        Dataloader for training and dev sets: both features & labels (Toy Dataset)
    """
    def __init__(
        self, root_dir: str=None, subset: str='train', label_to_idx: dict=None, 
        keep_tags: bool=True, EOS_IDX: str=None, use_specaug: bool=True
    ):
        # bookkeeping
        self.root_dir = root_dir
        self.subset = subset
        self.label_to_idx = label_to_idx
        self.keep_tags = keep_tags
        # using default structure (./mfcc + ./transcript/raw)
        mfcc_dir = f"{root_dir}/{subset}.npy"
        trans_dir = f"{root_dir}/{subset}_labels.npy"

        # whether to keep <sos> / <eos>
        if not self.keep_tags: 
            l, r = 1, -1
        
        # prepare for inputs
        self.EOS_IDX = EOS_IDX

        # load files
        self.mfccs = np.load(mfcc_dir)[:, :, :15]
        self.transcripts = [np.array([self.label_to_idx[yy] for yy in y]) for y in np.load(trans_dir)]

        # dataset size
        self.size = len(self.mfccs)

        # spec augs
        self.use_specaug = use_specaug
        self.freq_masker = None
        self.time_masker = None
        if self.use_specaug:
            self.freq_masker = tat.FrequencyMasking(2)
            self.time_masker = tat.TimeMasking(2)

    
    def __len__(self):
        return self.size

    
    def __getitem__(self, index):
        return torch.tensor(self.mfccs[index]), torch.tensor(self.transcripts[index])

        
    def collate_fn(self, batch):

        x_batch, y_batch = list(zip(*batch))

        x_lens      = [x.shape[0] for x in x_batch] 
        y_lens      = [y.shape[0] for y in y_batch] 

        x_batch_pad = pad_sequence(x_batch, batch_first=True, padding_value=self.EOS_IDX)
        y_batch_pad = pad_sequence(y_batch, batch_first=True, padding_value=self.EOS_IDX) 

        if self.use_specaug:
            x_batch_pad = self.time_masker(self.time_masker(self.time_masker(
                        self.freq_masker(x_batch_pad)
            )))
        
        return x_batch_pad, y_batch_pad, torch.tensor(x_lens), torch.tensor(y_lens)



class datasetTestToy(Dataset):
    """
        Dataloader for training and dev sets: both features & labels (Toy Dataset)
    """
    def __init__(
        self, root_dir: str=None, EOS_IDX: str=None
    ):
        # bookkeeping
        self.root_dir = root_dir
        # using default structure (./mfcc + ./transcript/raw)
        mfcc_dir = f"{root_dir}/dev.npy"

        # prepare for inputs
        self.EOS_IDX = EOS_IDX

        # load files
        self.mfccs = np.load(mfcc_dir)[:, :, :15]
        
        # dataset size
        self.size = len(self.mfccs)

    
    def __len__(self):
        return self.size

    
    def __getitem__(self, index):
        return torch.tensor(self.mfccs[index])

        
    def collate_fn(self, batch):

        x_batch = batch
        x_lens      = [x.shape[0] for x in x_batch] 

        x_batch_pad = pad_sequence(x_batch, batch_first=True, padding_value=self.EOS_IDX)
        
        return x_batch_pad, torch.tensor(x_lens)




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


def pay_attention_multihead(att_wgts, epoch: int, root_dir: str='.'):
    """
        Args:
            att_wgts: (num_heads, char_len, enc_len) 
    """
    num_heads = att_wgts.shape[0]
    n_rows = int(math.sqrt(num_heads))
    n_cols = num_heads // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    fig.suptitle(f"Attention Map [Epoch={epoch}]")
    fig.supxlabel('Output Character Count')
    fig.supylabel('Compressed Input Frame Count')
    if num_heads == 1:
        sns.heatmap(att_wgts[0], cmap='coolwarm')
    else:
        for r in range(n_rows):
            for c in range(n_cols):
                i = r * n_cols + c
                if n_rows > 1:
                    sns.heatmap(att_wgts[i], cmap='coolwarm', ax=axes[r, c])
                    axes[r, c].set_title(f"Attention Head #[{i}]")
                else:
                    sns.heatmap(att_wgts[i], cmap='coolwarm', ax=axes[c])
                    axes[c].set_title(f"Attention Head #[{i}]")
    img_fp = f"{root_dir}/attention-map-epoch{epoch}.png"
    fig.savefig(img_fp, dpi=128)
    print(f"\nAttention map successfully saved to [{img_fp}].\n")
