"""
    Model classes in use.
        granularity: components (Listener, Attention, Speller) & full LAS model
"""
import math
import torch
import numpy as np
import torch.nn as nn
from torchsummaryX import summary 

from src.utils import pay_attention_multihead
from src.modules import LockedLSTM, pyramLockedLSTM, AutoRegDecoderLSTMCell



class Listener(nn.Module):
    """
        Giant Listener class in the LAS model.
    """
    def __init__(
        self, 
        input_dim: int=15,
        uniform_hid_dim: int=256,
        lstm_layers: int=1,
        plstm_layers: int=3,
        bidirectional: bool=True,
        init_dropout: float=0.2,
        mid_dropout: float=0.3,
        final_dropout: float=0.4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.uniform_hid_dim = uniform_hid_dim
        self.lstm_layers = lstm_layers
        self.plstm_layers = plstm_layers
        self.bidirectional = bidirectional
        self.init_dropout = init_dropout
        self.mid_dropout = mid_dropout
        self.final_dropout = final_dropout

        # initial BiLSTM layer(s)
        self.base = LockedLSTM(
            lstm_input_dim=self.input_dim,
            uniform_hid_dim=self.uniform_hid_dim,
            lstm_layers=self.lstm_layers,
            bidirectional=self.bidirectional,
            init_dropout=self.init_dropout,
            mid_dropout=self.mid_dropout
        )
        self.pyramid = pyramLockedLSTM(
            plstm_input_dim=(int(self.bidirectional) + 1) * self.uniform_hid_dim,
            uniform_hid_dim=self.uniform_hid_dim,
            plstm_layers=self.plstm_layers,
            bidirectional=self.bidirectional,
            mid_dropout=self.mid_dropout,
            final_dropout=self.final_dropout
        )
        

    def forward(self, x, lx):
        """
            Args:
                x: (batch_size, padded_seq_len, input_dim)
                lx: (batch_size, )
        """
        return self.pyramid(*self.base(x, lx))



class MultiheadCrossAttention(nn.Module):
    """
        Naive implementation of multihead attention.
        Note:
            This class implementation mostly follows the starter codes 
                given in 11-751: Speech Recognition and Understanding (Fall 2022)
                but is adapted from this self-attention version to accommodate cross attention
    """
    def __init__(
        self, 
        enc_out_dim: int=512,
        dec_out_dim: int=128,
        proj_dim: int=128,
        heads: int=4,
        dropout: float=0.1
    ):
        super().__init__()
        assert proj_dim % heads == 0
        self.enc_out_dim = enc_out_dim
        self.dec_out_dim = dec_out_dim
        self.proj_dim = proj_dim
        self.heads = heads
        self.dims_per_head = self.proj_dim // self.heads
        self.norm_factor = 1 / math.sqrt(self.dims_per_head)                       # 1 / sqrt(proj_dim // head) for normalization
        # build up mapping matrices
        self.key_map = nn.Linear(self.enc_out_dim, self.proj_dim)
        self.value_map = nn.Linear(self.enc_out_dim, self.proj_dim)
        self.query_map = nn.Linear(self.dec_out_dim, self.proj_dim)
        # [optional] additional linear transformation layer
        self.final_map = nn.Linear(self.proj_dim, self.proj_dim)
        # softmax layer for attended value normalization
        self.softmax = nn.Softmax(dim=-1)
        # dropout rate
        self.dropout = dropout


    @staticmethod
    def build_pad_masks(enc_l):
        """
            build up masks to ignore padded sections
            Args:
                enc_l: (batch_size, ) encoder output lengths
        """
        max_len = enc_l.max()
        return (torch.arange(0, max_len, dtype=torch.int64).unsqueeze(0) 
                >= enc_l.unsqueeze(1))                                        # (batch_size, enc_max_len)


    def locked_dropout(self, x):
        """
            keep the same dropout for the entire batch
        """
        if (not self.training) or (not self.dropout):
            return x
        mask = x.new_empty(1, x.size(1)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout).expand_as(x)
        return x * mask
        

    def wrapup_encodings(self, enc_h, enc_l):
        """
            obtain the keys, values for encoded inputs
                Note: computed only once for computation savings (and they stay unchanged)
            Args:
                enc_h: (batch_size, padded_seq_len, h_dim) encoder outputs
                    [1] padded_seq_len is the discounted version
                    [2] h_dim is original hidden dim * 2 (concatenation)
                enc_l: (batch_size, ) encoder output lengths 
                # Note: enc_l may be long int: remember to get back to int
        """
        # [0] dims
        batch_size, enc_max_len, _ = enc_h.size()
        # [1] keys and values 
        self.keys = (self.key_map(enc_h)                                            # (batch_size, seq_len, proj_dim)
                         .view(batch_size, -1, self.heads, self.dims_per_head)      # (batch_size, seq_len, num_heads, proj_dims_per_head)
                         .transpose(1, 2)                                           # (batch_size, num_heads, seq_len, proj_dims_per_head)
                         .transpose(-2, -1))                                        # (batch_size, num_heads, proj_dims_per_head, seq_len)
        self.values = (self.value_map(enc_h)                                        # (batch_size, seq_len, num_heads, proj_dim)
                           .view(batch_size, -1, self.heads, self.dims_per_head)    # (batch_size, seq_len, num_heads, proj_dims_per_head)
                           .transpose(1, 2))                                        # (batch_size, num_heads, seq_len, proj_dims_per_head)
        # [2] masks
        mask = self.build_pad_masks(enc_l)                                          # (batch_size, max_len)
        self.masks = (mask[:, None, None, :]
                      .expand((batch_size, self.heads, 1, enc_max_len))             # (batch_size, num_heads, 1, seq_len)
                      .to(enc_h.device))


    def forward(self, dec_h, return_wgts: bool=False, init_wgts_mask: torch.tensor=None):
        """
            official forward for the decoder
            Args:
                dec_h: (batch_size, dec_lstm_out_dim)
        """
        # [0] dims
        batch_size = dec_h.size(0)                                                  # (batch_size, 1, dec_lstm_out_dim)
        # [1] query
        self.queries = (self.query_map(dec_h)                                       # (batch_size, proj_dim)
                            .view(batch_size, self.heads, self.dims_per_head)       # (batch_size, num_heads, proj_dims_per_head)
                            .unsqueeze(2))                                          # (batch_size, num_heads, 1, proj_dims_per_head)
        # [2] attention weights
        wgts_prenorm = torch.matmul(self.queries, self.keys) / self.norm_factor     # (batch_size, num_heads, 1, seq_len)
        # add initial forcing if existed
        if init_wgts_mask is not None:
            wgts_prenorm *= init_wgts_mask
        min_val = torch.finfo(wgts_prenorm.dtype).min
        wgts_prenorm = wgts_prenorm.masked_fill(self.masks, min_val)                # (batch_size, num_heads, 1, seq_len)
        #     apply softmax & zero out trivial vals
        wgts_normed = (self.softmax(wgts_prenorm)
                           .masked_fill(self.masks, 0.0))                           # (batch_size, num_heads, 1, seq_len)
        # [3] attended values
        att_values = (torch.matmul(wgts_normed, self.values)                        # (batch_size, num_heads, 1, proj_dims_per_head)
                           .squeeze(-2).contiguous()                                # (batch_size, num_heads, proj_dims_per_head)
                           .view(batch_size, -1))                                   # (batch_size, proj_dim)
        # [4] (optional) final linear layer
        att_values = self.final_map(self.locked_dropout(att_values))                # (batch_size, proj_dim)

        return (att_values, wgts_normed) if return_wgts else att_values




class Speller(nn.Module):
    """
        Giant class with unidirectional LSTM layers as the decoder.
    """
    def __init__(
        self,
        # attention (encoder -> attention)
        enc_out_dim: int=512,
        att_proj_dim: int=128,
        att_heads: int=4,
        att_dropout: float=0.2,
        # decoder embeddings
        dec_vocab_size: int=30,
        dec_emb_dim: int=256,
        dec_emb_dropout: float=0.5,
        # decoder lstm cells
        dec_lstm_hid_dim: int=512,
        dec_lstm_out_dim: int=128,
        dec_lstm_dropout: float=0.2,
        # trivials
        CHR_MAX_STEPS: int=600,
        CHR_PAD_IDX: int=29,
        CHR_SOS_IDX: int=0,
        USE_GREEDY: bool=True
    ):
        """
            Args:
                vocab_size: size of characters
                dec_emb_dim: embedding dimensions
                dec_pad_idx: padding indices for the characters
                dec_hid_dim: hidden dimension within uniLSTMs
                dec_out_dim: hidden dimension within uniLSTMs
                dec_emb_dropout: embedding dropout
                dec_mid_dropout: dropout rate for the mid later
        """
        super().__init__()
        # attention
        self.enc_out_dim = enc_out_dim
        self.att_proj_dim = att_proj_dim
        self.att_heads = att_heads
        self.att_dropout = att_dropout
        # embedding
        self.dec_vocab_size = dec_vocab_size
        self.dec_emb_dim = dec_emb_dim
        self.dec_emb_dropout = dec_emb_dropout
        # lstm: 2 cells by default
        self.dec_lstm_hid_dim = dec_lstm_hid_dim
        self.dec_lstm_out_dim = dec_lstm_out_dim
        self.dec_lstm_dropout = dec_lstm_dropout
        # trivials
        self.CHR_MAX_STEPS = CHR_MAX_STEPS
        self.CHR_PAD_IDX = CHR_PAD_IDX
        self.CHR_SOS_IDX = CHR_SOS_IDX
        self.USE_GREEDY = USE_GREEDY

        # attention module
        self.attention = MultiheadCrossAttention(
            enc_out_dim=self.enc_out_dim,
            dec_out_dim=self.dec_lstm_out_dim,
            proj_dim=self.att_proj_dim,
            heads=self.att_heads,
            dropout=self.att_dropout
        )
        # embedding layer
        self.char_emb = nn.Embedding(
            num_embeddings=self.dec_vocab_size,
            embedding_dim=self.dec_emb_dim,
            padding_idx=self.CHR_PAD_IDX
        )
        # LSTMs
        self.lstms = AutoRegDecoderLSTMCell(
            att_proj_dim=self.att_proj_dim,
            dec_emb_dim=self.dec_emb_dim,
            dec_hid_dim=self.dec_lstm_hid_dim,
            dec_out_dim=self.dec_lstm_out_dim,
            dec_mid_dropout=self.dec_lstm_dropout
        )
        self.init_query = nn.Parameter(torch.rand((1, self.dec_lstm_out_dim)), requires_grad=True)
        self.init_hiddens = [(
            nn.Parameter(torch.zeros((1, self.dec_lstm_hid_dim)), requires_grad=True),
            nn.Parameter(torch.zeros((1, self.dec_lstm_hid_dim)), requires_grad=True)
        ), (
            nn.Parameter(torch.zeros((1, self.dec_lstm_out_dim)), requires_grad=True),
            nn.Parameter(torch.zeros((1, self.dec_lstm_out_dim)), requires_grad=True)
        )]
        # classification layers
        self.gap = nn.Linear(self.dec_lstm_out_dim + self.att_proj_dim, self.dec_emb_dim)
        self.act = nn.GELU()
        self.cls = nn.Linear(self.dec_emb_dim, self.dec_vocab_size)
        # weight tying
        self.cls.weight = self.char_emb.weight
    
        
    def locked_dropout_withmask(self, x, p):
        """
            keep the same dropout for the entire batch
        """
        if (not self.training) or (not p):
            return x, None
        mask = x.new_empty(1, x.size(1)).bernoulli_(1 - p).div_(1 - p).expand_as(x)
        return x * mask, mask
        

    def forward(self, enc_h, enc_l, dec_y=None, teacher_forcing_rate: float=1, init_force: bool=False):
        """
            Args:
                enc_h: (batch_size, enc_seq_len, enc_output_dim) encoder outputs
                enc_l: (batch_size, ) encoder output lengths
                dec_y: (batch_size, dec_seq_len) decoder outputs for training
                teacher_forcing_rate: (float) how much of teacher forcing to apply
        """
        batch_size, enc_max_len, enc_dim = enc_h.size()

        # teacher forcing during training
        if self.training:
            steps = dec_y.size(-1)
            gold_label_emb = self.char_emb(dec_y)                                   # (batch_size, dec_seq_len, dec_emb_dim)
        else:
            steps = self.CHR_MAX_STEPS
        
        # all prediction logits saved in one list
        pred_logits = list()

        """
            initiate attention for the encoded inputs
        """
        # attention keys & vals for encoder
        self.attention.wrapup_encodings(enc_h, enc_l)

        if init_force:
            a_side, b_side = enc_max_len // 5 + 1, steps // 5 + 1
            areas = a_side * b_side
            blocks = [torch.ones((a_side, b_side), device=enc_h.device) for _ in range(5)]
            init_wgts = torch.block_diag(*blocks)[:enc_max_len, :steps]

        """
            priors: t = -1
        """
        # first character: <sos>
        char = torch.full((batch_size, ), fill_value=self.CHR_SOS_IDX, 
                           dtype=torch.long, device=enc_h.device)                   # (batch_size, )
        # initial hidden states
        hiddens = [None, None]
        hiddens[0] = [u.expand(batch_size, self.dec_lstm_hid_dim).to(enc_h.device) 
                      for u in self.init_hiddens[0]]
        hiddens[1] = [u.expand(batch_size, self.dec_lstm_out_dim).to(enc_h.device) 
                      for u in self.init_hiddens[1]]
        # initial query & context
        init_query = self.init_query.expand(batch_size, self.dec_lstm_out_dim).to(enc_h.device)
        context, att_wgts = self.attention(init_query, return_wgts=True)
        
        # bookkeeping
        att_wgts_list = [att_wgts[0].detach().cpu()]                                # (num_heads, 1, enc_seq_len)
        # loop
        for t in range(steps):
            # get character embeddings from prev step
            char_emb = self.char_emb(char)                                          # (batch_size, dec_emb_dim)
            # teacher forcing if wanted
            if self.training and t > 0:
                if torch.rand(1).item() <= teacher_forcing_rate:
                    char_emb = gold_label_emb[:, t - 1, :]                          # (batch_size, dec_emb_dim)
                char_emb = char_emb_mask * char_emb
                context = context_mask * context
            if self.training and t == 0:
                # embedding dropout
                char_emb, char_emb_mask = self.locked_dropout_withmask(char_emb, self.dec_emb_dropout)
                # context dropout
                context, context_mask = self.locked_dropout_withmask(context, self.att_dropout)

            # input for decoder: char emb + context, prev step
            hiddens = self.lstms(char_emb, context, hiddens)
            # context
            init_wgts_slice = None
            if init_force:
                init_wgts_slice = init_wgts[:, t].expand(batch_size, self.att_heads, 1, enc_max_len)
            context, att_wgts = self.attention(hiddens[-1][0], return_wgts=True, init_wgts_mask=init_wgts_slice)
            # (batch_size, proj_dim), (batch_size, num_heads, 1, enc_seq_len)
        
            # concatenate last layer hidden states w/ context for char network to make a decision
            projected_queries = self.attention.queries.view(batch_size, -1)
            dec_out = torch.cat((projected_queries, context), dim=-1)               # (batch_size, dec_out_dim + att_proj_dim)
            # char_logits = self.cls(self.act(self.gap(dec_out)))                     
            # (batch_size, dec_out_dim + att_proj_dim) --> (batch_size, dec_emb_dim) --> (batch_size, vocab_size)
            char_logits = self.cls(dec_out)

            # add logits
            pred_logits.append(char_logits)                                         # (batch_size, dec_max_len, vocab_size)
            att_wgts_list.append(att_wgts[0].detach().cpu())                        # (num_heads, 1, enc_seq_len)

            # obtain the char to input for next timestep
            if self.USE_GREEDY:
                char = char_logits.argmax(-1)                                       # (batch_size, )
            else:
                # left for beam search (w/ LM rescoring)
                pass
        
        # concatenate all the predicted char logits across time
        pred_logits = torch.stack(pred_logits, dim=1)                               # (batch_size, dec_max_len, vocab_size)
        # concatenate all the attention maps
        att_wgts_list = torch.cat(att_wgts_list, dim=1).transpose(-2, -1)           # (num_heads, dec_max_len, enc_seq_len)
        return pred_logits, att_wgts_list



class ListenAttendSpell(nn.Module):
    """
        Full LAS model.
    """
    def __init__(
        self, 
        listener_configs: dict,
        speller_configs: dict
    ):
        super().__init__()
        self.listener_configs = listener_configs
        self.speller_configs = speller_configs
        self.speller_configs['enc_out_dim'] = 2 * self.listener_configs['uniform_hid_dim']
        # modules
        self.listen = Listener(**self.listener_configs)
        self.spell = Speller(**self.speller_configs)

    
    def forward(self, x, lx, dec_y=None, teacher_forcing_rate: float=0.0, init_force: bool=False):
        # listen
        enc_h, enc_l = self.listen(x, lx)
        # enc_h: (batch_size, enc_max_len, enc_hid_dim * 2); enc_l: (batch_size, )

        # spell
        pred_logits, att_wgts_list = self.spell(enc_h, enc_l, dec_y, teacher_forcing_rate, init_force)
        # pred_logits: (batch_size, dec_max_len, num_chars); att_wgts_list: (num_heads, out_len, in_len)

        return pred_logits, att_wgts_list 




if __name__ == '__main__':

    """
        test run for model classes
    """
    SEED = 416
    torch.manual_seed(SEED)
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    """
        encoder
    """
    ENC_LSTM_INPUT_DIM = 15
    ENC_LSTM_LAYERS = 1
    ENC_PLSTM_LAYERS = 3
    ENC_BIDIRECTIONAL = True

    ENC_HID_DIM = 256
    ENC_INIT_DROPOUT = 0.2
    ENC_MID_DROPOUT = 0.3
    ENC_FINAL_DROPOUT = 0.3

    """
        attention
    """
    ATT_PROJ_DIM = 128
    ATT_HEADS = 4
    ATT_DROPOUT = 0.2


    """
        decoder
    """
    DEC_EMB_DIM = 256
    DEC_EMB_DROPOUT = 0.2
    DEC_LSTM_HID_DIM = 512
    DEC_LSTM_OUT_DIM = 128
    DEC_LSTM_DROPOUT = 0.2
    VOCAB = [
        '<sos>', 'A', 'B', 'C', 'D', 'E', 
        'F', 'G', 'H', 'I', 'J', 'K', 
        'L', 'M', 'N', 'O', 'P', 'Q', 
        'R', 'S', 'T', 'U', 'V', 'W', 
        'X', 'Y', 'Z', "'", ' ', '<eos>'
    ]
    DEC_VOCAB_SIZE = len(VOCAB)
    CHAR2IDX = {
        '<sos>': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 
        'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 
        'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 
        'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 
        'X': 24, 'Y': 25, 'Z': 26, "'": 27, ' ': 28, '<eos>': 29
    }
    CHR_PAD_IDX = CHAR2IDX['<eos>']          # 29
    CHR_SOS_IDX = CHAR2IDX['<sos>']          # 0
    CHR_MAX_STEPS = 20
    TEACHER_FORCING = 0.9


    testListener = Listener(
        input_dim=ENC_LSTM_INPUT_DIM,
        uniform_hid_dim=ENC_HID_DIM,
        lstm_layers=ENC_LSTM_LAYERS,
        plstm_layers=ENC_PLSTM_LAYERS,
        bidirectional=ENC_BIDIRECTIONAL,
        init_dropout=ENC_INIT_DROPOUT,
        mid_dropout=ENC_MID_DROPOUT,
        final_dropout=ENC_FINAL_DROPOUT
    ).to(device)

    # simulate data
    BATCH_SIZE = 4
    LX = torch.randint(low=800, high=1200, size=(BATCH_SIZE, ), dtype=torch.int32)
    X = [torch.rand(size=(LX[i], ENC_LSTM_INPUT_DIM)) for i in range(BATCH_SIZE)]
    from torch.nn.utils.rnn import pad_sequence
    X = pad_sequence(X, batch_first=True, padding_value=0)
    X = X.to(device)
    
    print(f"\n\nModel summary for the encoder pyramidal BiLSTM is:\n{summary(testListener, X, LX)}\n")
    XX, LXX = testListener(X, LX)
    
    """ 
        Standalone attention test (shape)
            # show masks
            testMHA = MultiheadCrossAttention(
                enc_out_dim=ENCODER_HID_DIM * 2,
                dec_hid_dim=dec_hid_dim,
                proj_dim=ATT_PROJ_DIM,
                heads=ATT_HEADS,
                dropout=ATT_DROPOUT
            )

            testMHA.wrapup_encodings(XX, LXX)
            Y = torch.zeros((BATCH_SIZE, dec_hid_dim))

            att_vals, wgts = testMHA(Y, True)
            # expected shapes: torch.Size([4, 512]) torch.Size([4, 4, 1, 130])

        This section has been commented since it's included in the next Speller module
    """

    # build y
    LY = torch.randint(low=0, high=60, size=(BATCH_SIZE,))
    Y = [torch.randint(low=0, high=DEC_VOCAB_SIZE, size=(ly.int(), )) for ly in LY]
    Y = pad_sequence(Y, batch_first=True, padding_value=CHR_PAD_IDX)
    Y = Y.to(device)
    
    testSpeller = Speller(
        # attention (encoder -> attention)
        enc_out_dim=ENC_HID_DIM * 2,
        att_proj_dim=ATT_PROJ_DIM,
        att_heads=ATT_HEADS,
        att_dropout=ATT_DROPOUT,
        # decoder embeddings
        dec_vocab_size=DEC_VOCAB_SIZE,
        dec_emb_dim=DEC_EMB_DIM,
        dec_emb_dropout=DEC_EMB_DROPOUT,
        # decoder lstm cells
        dec_lstm_hid_dim=DEC_LSTM_HID_DIM,
        dec_lstm_out_dim=DEC_LSTM_OUT_DIM,
        dec_lstm_dropout=DEC_LSTM_DROPOUT,
        # trivials
        CHR_MAX_STEPS=CHR_MAX_STEPS,
        CHR_PAD_IDX=CHR_PAD_IDX,
        CHR_SOS_IDX=CHR_SOS_IDX
    ).to(device)

    print(f"The model architecture of the [Speller] is:\n{summary(testSpeller, XX, LXX, Y, TEACHER_FORCING)}")
    
    # show attention heads
    PY, ATT = testSpeller(XX, LXX, Y, TEACHER_FORCING)
    # expected shape: torch.Size([4, 54, 30]) (4, 55, 130)

    # show multihead attention weights
    import os
    from src.utils import pay_attention_multihead

    img_dir = './imgs/example'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    pay_attention_multihead(ATT, epoch=-1, root_dir=img_dir)
