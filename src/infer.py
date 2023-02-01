"""
    Driver script for model testing.
        Mostly borrowed from my previous work:
            https://github.com/Astromsoc/seq-to-seq-auto-phoneme-recognition/blob/master/src/infer.py
"""

import yaml
import argparse
import pandas as pd

from tqdm import tqdm

from src.utils import *
from src.models import *
from src.constants import *



def idx_to_str(idx_seq, vocab: list, sos_idx: int, eos_idx: int):
    """
        Args:
            idx_seq: (max_seq_len, )
    """
    out_list = list()
    for idx in idx_seq:
        if idx == sos_idx:
            continue
        elif idx == eos_idx:
            break
        else:
            out_list.append(vocab[idx])
    return ''.join(out_list)
    


def infer_one_checkpoint(
        model_cfgs, infcfgs, checkpoint_filepath, loader, tag,
        template_filepath, scaler, device, VOCAB, SOS_IDX, EOS_IDX
    ):

    print(f"\n\nRunning inference on checkpoint [{checkpoint_filepath}]...\n")
    # reconstruct the model
    model = ListenAttendSpell(**model_cfgs.model.configs)

    # load from checkpoint
    ckpt = torch.load(checkpoint_filepath, map_location=torch.device(device))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    all_preds = list()

    # iterate over batches
    for b, batch in tqdm(enumerate(loader), total=len(loader)):
        x, lx = batch
        x = x.to(device)

        if device.startswith("cuda") and scaler is not None:
            with torch.cuda.amp.autocast():
                pred_logits, att_wgts = model(x, lx)
        else:
            pred_logits, att_wgts = model(x, lx)

        # obtain batch predictions
        if infcfgs.use_greedy:
            batch_preds = [idx_to_str(pl.argmax(-1), VOCAB, SOS_IDX, EOS_IDX) for pl in pred_logits]
        all_preds.extend(batch_preds)

    # output csv filename: adapted from checkpoint name
    out_filepath = checkpoint_filepath.replace('.pt', f'-{tag}.csv').replace('ckpts', 'preds')
    # check existence of output folder
    out_folder = out_filepath.rsplit('/', 1)[0]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # generate csv file
    raw_df = pd.read_csv(template_filepath)
    raw_df.label = all_preds
    raw_df.to_csv(out_filepath, index=False)
    
    return all_preds




def main(args):
    infcfgs = cfgClass(yaml.safe_load(open(args.config_file, 'r')))
    exp_folder = infcfgs.exp_folder

    # find the device
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"\n\nRunning on [{device}]...\n")

    # load configs & model checkpoints from given experiment folder
    model_cfgs = cfgClass(yaml.safe_load(open(f"{exp_folder}/config.json", 'r')))

    use_mini = True if model_cfgs.TRN_FOLDER.startswith('mini') else False
    VOCAB = model_cfgs.VOCAB
    VOCAB_MAP = model_cfgs.VOCAB_MAP
    EOS_IDX = model_cfgs.EOS_IDX
    SOS_IDX = model_cfgs.SOS_IDX

    """
        load data & build data loaders
    """

    if use_mini:
        someDataset = datasetTestToy(stdDir=infcfgs.SOME_FOLDER)
        someLoader = DataLoader(
            someDataset,
            batch_size=infcfgs.batch_size,
            num_workers=infcfgs.num_workers,
            collate_fn=someDataset.collate_fn
        )

    someDataset = datasetTest(stdDir=infcfgs.SOME_FOLDER)
    someLoader = DataLoader(
        someDataset,
        batch_size=infcfgs.batch_size,
        num_workers=infcfgs.num_workers,
        collate_fn=someDataset.collate_fn
    )
    tag = 'trn' if 'train' in infcfgs.SOME_FOLDER else 'dev' if 'dev' in infcfgs.SOME_FOLDER else 'tst'
    print(f"\nA total of [{len(someLoader)}] batches in test set.\n")

    # load the template for test answer generation
    template_filepath = f"{infcfgs.SOME_FOLDER}/transcript/random_submission.csv"
    if not os.path.exists(template_filepath):
        template_filepath = f"{infcfgs.SOME_FOLDER}/transcript/processed.csv"
    assert os.path.exists(template_filepath)

    # build scaler
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    # load all checkpoints
    ckpts = [f for f in os.listdir(f"{exp_folder}/ckpts") if f.endswith('.pt')]

    # experiments
    if infcfgs.run_all:
        for fp in ckpts:
            infer_one_checkpoint(
                model_cfgs=model_cfgs, infcfgs=infcfgs, checkpoint_filepath=f"{exp_folder}/ckpts/{fp}", 
                loader=someLoader, template_filepath=template_filepath, tag=tag,
                scaler=scaler, device=device, VOCAB=VOCAB, EOS_IDX=EOS_IDX, SOS_IDX=SOS_IDX
            )
    elif f"-epoch[{infcfgs.epoch_num}].pt" in ' '.join(ckpts):
        fp = [f for f in ckpts if f.endswith(f"-epoch[{infcfgs.epoch_num}].pt")][0]
        infer_one_checkpoint(
            model_cfgs=model_cfgs, infcfgs=infcfgs, checkpoint_filepath=f"{exp_folder}/ckpts/{fp}", 
            loader=someLoader, template_filepath=template_filepath, tag=tag,
            scaler=scaler, device=device, VOCAB=VOCAB, EOS_IDX=EOS_IDX, SOS_IDX=SOS_IDX
        )
    
    if infcfgs.run_avg:
        model = ListenAttendSpell(**model_cfgs.model.configs)
        base_dict = dict(model.named_parameters())
        state_dict = dict()
        for fp in ckpts:
            state_dict[fp] = torch.load(f"{exp_folder}/ckpts/{fp}", map_location=torch.device(device))['model_state_dict']
        for k in base_dict.keys():
            base_dict[k] = None
            for wgts in state_dict.values():
                if base_dict[k] is None:
                    base_dict[k] = wgts[k] / len(ckpts) 
                else:
                    base_dict[k] += wgts[k] / len(ckpts) 
        # save the avg ckpt
        torch.save({'model_state_dict': base_dict}, f"{exp_folder}/ckpts/avg-all.pt")
        # run
        infer_one_checkpoint(
            model_cfgs=model_cfgs, infcfgs=infcfgs, checkpoint_filepath=f"{exp_folder}/ckpts/avg-all.pt", 
            loader=someLoader, template_filepath=template_filepath, tag=tag,
            scaler=scaler, device=device, VOCAB=VOCAB, EOS_IDX=EOS_IDX, SOS_IDX=SOS_IDX
        )




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Driver script for model inference.')

    parser.add_argument(
        '--config-file',
        '-c',
        default='./config/infer.yml',
        type=str,
        help='Filepath of configuration yaml file to be read.'
    )
    args = parser.parse_args()

    main(args)