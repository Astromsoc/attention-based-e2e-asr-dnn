"""
    Utility functions for development
"""

import os
import shutil
import numpy as np


def extract_mini(root_dir: str='./data', out_dir: str='./small', ratio=0.05):
    trn = f"{root_dir}/train-clean-100"
    val = f"{root_dir}/dev-clean"

    for subroot in (trn, val):
        mfcc_dir = f"{subroot}/mfcc"
        trans_dir = f"{subroot}/transcript/raw"

        all_fns = sorted([
            f for f in os.listdir(mfcc_dir) if f.endswith('.npy')
        ])
        out_num = int(ratio * len(all_fns))
        fns = np.random.choice(all_fns, size=out_num)
        for tag in ('mfcc', 'transcript/raw'):
            out_subdir = f"{subroot}/{tag}".replace(root_dir, out_dir)
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)
            for fn in enumerate(fns):
                fn = fn[1]
                if tag != 'mfcc':
                    fn = fn.replace('_', '-')
                shutil.copy(f"{subroot}/{tag}/{fn}", f"{out_subdir}/{fn}".replace('_', '-'))


def uniform_filenames(root_dir: str='./data'):
    trn = f"{root_dir}/train-clean-100"
    val = f"{root_dir}/dev-clean"
    tst = f"{root_dir}/test-clean"

    for subdir in (trn, val, tst):
        subdir = subdir + '/mfcc'
        for f in os.listdir(subdir):
            if f.endswith('.npy'):
                os.rename(
                    f"{subdir}/{f}",
                    f"{subdir}/{f.replace('_', '-')}"
                )



if __name__ == '__main__':

    # replace filenames
    uniform_filenames()

    # extract dev dataset
    extract_mini()
