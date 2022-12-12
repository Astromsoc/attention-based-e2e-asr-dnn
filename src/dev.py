"""
    Utility functions for development
"""

import os
import shutil
import numpy as np


def extract_mini(root_dir: str='./data', out_dir: str='./mini', ratio=0.1):
    trn = f"{root_dir}/train-clean-100"
    val = f"{root_dir}/dev-clean"

    for subroot in (trn, val):
        mfcc_dir = f"{subroot}/mfcc"
        trans_dir = f"{subroot}/transcript/raw"

        mfcc_fns = sorted([
            os.path.join(mfcc_dir, f) for f in os.listdir(mfcc_dir) if f.endswith('.npy')
        ])
        trans_fns = sorted([
            os.path.join(trans_dir, f) for f in os.listdir(trans_dir) if f.endswith('.npy')
        ])

        out_num = int(ratio * len(mfcc_fns))
        fns = np.random.choice(mfcc_fns, size=out_num)
        for tag in ('mfcc', 'transcript/raw'):
            out_subdir = f"{subroot}/{tag}".replace(root_dir, out_dir)
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)
            for fn in enumerate(fns):
                fn = fn[1]
                shutil.copy(fn, fn.replace(root_dir, out_dir))




if __name__ == '__main__':
    extract_mini()
