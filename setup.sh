#!/bin/bash
conda create -n las python=3.9
conda activate las

# basic packages
pip install python-levenshtein torchsummaryX wandb
pip install numpy matplotlib seaborn

# conda installed packages
yes | conda install torchaudio pytorch -c pytorch
yes | conda install tmux

# download datasets
kaggle competitions download -c 11-785-f22-hw4p2