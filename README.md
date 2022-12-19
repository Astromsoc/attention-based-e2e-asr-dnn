# Attention Based End-to-End Speech-to-Text Deep Neural Network
For the project: Attention-based End-to-End Speech-to-Text Deep Neural Network, as part of homework 4 under the course 11-785: Intro to Deep Learning @ CMU 2022 Fall.


**Note**: Overlapped sections and codes are directly taken from my previous implementation of *Sequence-to-Sequence Frame-Level Phoneme Recognition*, for partial fulfillment of homework 3 of this course.

 
**Author**: Siyu Chen (schen4)

**Wandb Projects**: [full experiments](https://wandb.ai/astromsoc/785-hw4-full/overview);
[test experiments](https://wandb.ai/astromsoc/785-hw4-test/overview);
[rewriter experiments](https://wandb.ai/astromsoc/785-hw4-rewriter/overview)

**Github Repo**: [Click Here](https://github.com/Astromsoc/attention-based-e2e-asr-dnn) (will be made public after the end of this semester, as is required) 


### Repository Structure
```
.
├── LICENSE
├── README.md
├── config
│   ├── infer.yml
│   ├── lm-infer.yml
│   ├── rewriter.yml
│   └── sample-attention.yml
├── data
│   ├── __init__.py
│   └── extract_mini.py
├── imgs
│   └── example
│       └── attention-map-epoch-1.png
├── setup.sh
└── src
    ├── constants.py
    ├── dev.py
    ├── infer.py
    ├── lminfer.py
    ├── lmtrain.py
    ├── models.py
    ├── modules.py
    ├── train.py
    └── utils.py

5 directories, 19 files
```




### Language Model: Rewriter

Different from the original proposal of using LM to rescore transcriptions retrieved from a beam search, I implemented a seq2seq model that aims to "auto-correct" the mistakes that the LAS ASR model predictions tend to make, by training it on prediction-groundtruth pairs on training set, evaluating and saving the best checkpoints on dev set, and correcting predictions coming from less-than-perfect predictions. However, since this is also an attention-based model that takes time to converge with obvious attention weight diagonal, I haven't got a good checkpoint that generates sensible results without harming the original baseline quality of model predictions. Still, this is a funny direction that I'd love to work on after this competition and would update on the progress (in this github repo) once I have any.



### Best Model Architecture

The best result I've obtained on the Kaggle leaderboard is a Levenshtein distance of `7.27544` (on the entire sector). It is obtained by training the following network for 150 epochs:

```
model:
  tag: base-LAS
  configs:
    listener_configs:
      input_dim: 15
      uniform_hid_dim: 512
      lstm_layers: 1
      plstm_layers: 3
      bidirectional: true
      init_dropout: 0.3 --> 0.2
      mid_dropout: 0.3 --> 0.2
      final_dropout: 0.3 --> 0.2
    speller_configs:
      # encoder + attention
      att_proj_dim: 256
      att_heads: 1
      att_dropout: 0.0
      # decoder embeddings
      dec_emb_dim: 512
      dec_emb_dropout: 0.0
      # decoder lstm cells
      dec_lstm_hid_dim: 512
      dec_lstm_out_dim: 256
      dec_lstm_dropout: 0.3 --> 0.2
      # trivials
      CHR_MAX_STEPS: 600
      USE_GREEDY: true
```
With hyperparameters set as below:

```
# optimizer
optimizer:
  name: adamw
  configs:
    lr: 0.001
    weight_decay: 5.0e-6
    amsgrad: true

# scalar
scaler:
  use: true
```

And a pretraining + finetuning with changed hyperparameters following this scheme:
#### Stage 1. Pretraining 
*exp: peachy-microwave-101*

All listener dropouts and decoder dropouts added with spectrogram augmentation added, converged after 20 epochs, with a teacher-forcing rate of 1.0 constantly. It reached an initial low LD of `~17.633`.

#### Stage 2. Finetuning: tf-rate 1.0 --> 0.9
*exp: serene-sunset-108*

LD descended to `~12.919`.

#### Stage 3. Finetuning: tf-rate 0.9 --> 0.8
*exp: youthful-wave-110*

LD descended to `~10.098`.

#### Stage 4. Finetuning: tf-rate 0.8 --> 0.7
*exp: fearless-field-114*

LD descended to `~8.389`.

#### Stage 5. Finetuning: tf-rate 0.7 --> 0.6, learning rate scheduler turned on
*exp: elated-sea-116*

Scheduler Configs:

```
self.epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
	self.optimizer, factor=0.5, patience=4, mode='min'
)
```

LD descended to `~7.356`.

#### Stage 6. Finetuning: tf-rate 0.6 --> 0.5, dropouts 0.3 --> 0.2 (no obvious improvements)
*exp: eabsurd-microwave-121*

LD oscillates around `~7.5`.
