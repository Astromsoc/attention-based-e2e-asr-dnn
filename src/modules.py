"""
    Module classes in use.
        granularity: single encoder layer or block
"""
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LockedLSTM(nn.Module):
    """
        class for ordinary LSTM w/ locked dropout
            Note:
                [1] assuming every BiLSTM has the same output dim
                [2] assuming batch_first
    """
    def __init__(
            self, 
            lstm_input_dim: int=15,
            uniform_hid_dim: int=256, 
            lstm_layers: int=1,
            bidirectional: bool=True,
            init_dropout: float = 0.2,
            mid_dropout: float = 0.3
        ):
        super().__init__()
        self.lstm_input_dim = lstm_input_dim
        self.uniform_hid_dim = uniform_hid_dim
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.init_dropout = init_dropout
        self.mid_dropout = mid_dropout

        # original LSTM layers
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=(
                    self.lstm_input_dim if i == 0 else 
                    self.uniform_hid_dim * (int(self.bidirectional) + 1)
                ),
                hidden_size=self.uniform_hid_dim,
                num_layers=1,
                batch_first=True,
                dropout=0,
                bidirectional=self.bidirectional
            ) for i in range(self.lstm_layers)
        ])
    

    def locked_dropout(self, x, p: float=0.0):
        """
            functionalized dropout
            Args:
                x: (batch_size, padded_length, hidden_dim)
                p: dropout rate
        """
        if (not self.training) or (not p):
            return x
        # mask
        mask = x.new_empty(
            x.size(0), 1, x.size(2), requires_grad=False
        ).bernoulli_(1 - p).div_(1 - p).expand_as(x)
        return x * mask
    

    def forward(self, x, lx):
        """
            Args:
                x: (batch_size, padded_seq_len, hidden_dim)
                    expected to be unpacked before feeding in
                lx: (batch_size, )
        """
        for i, lstm in enumerate(self.lstms):
            # get dropout probability
            p = self.mid_dropout if i else self.init_dropout
            # pack 
            x = pack_padded_sequence(x, lengths=lx, batch_first=True, enforce_sorted=False)
            # lstm
            x, _ = lstm(x)
            # pad
            x, lx = pad_packed_sequence(x, batch_first=True)
            # dropout
            x = self.locked_dropout(x, p)
        return x, lx

        

class pyramLockedLSTM(nn.Module):
    """
        class for pyramidal LSTM w/ locked dropout
            Note:
                [1] BiLSTM x n --> pyramidalBiLSTM x m
                [2] assuming every BiLSTM or pyramidalBiLSTM layer has the same output dim
                    respectively
                [3] assuming batch_first
    """
    def __init__(
            self, 
            plstm_input_dim: int=512,
            uniform_hid_dim: int=256,
            plstm_layers: int=3,
            bidirectional: bool=True,
            mid_dropout: float = 0.2,
            final_dropout: float = 0.2
        ):
        """
            Args:
                plstm_input_dim: original input dim to 1st plstm layer
                uniform_hid_dim: the same hidden dimensional for all plstm layers
                plstm_layers: number of plstm layers
                dropout: unchanged dropout rates
        """
        super().__init__()
        # logging
        self.plstm_input_dim = plstm_input_dim
        self.uniform_hid_dim = uniform_hid_dim
        self.plstm_layers = plstm_layers
        self.bidirectional = bidirectional
        self.mid_dropout = mid_dropout
        self.final_dropout = final_dropout

        # build up feature dimensions for later use
        self.dims = [2 * self.uniform_hid_dim * (int(self.bidirectional) + 1)
                     for _ in range(self.plstm_layers)]
        self.dims[0] = 2 * self.plstm_input_dim
        # pyramidal LSTM layers
        self.plstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.dims[i],
                hidden_size=self.uniform_hid_dim,
                num_layers=1,
                batch_first=True,
                dropout=0,
                bidirectional=self.bidirectional
            ) for i in range(self.plstm_layers)
        ])
        

    def locked_dropout(self, x, p: float=0.0):
        """
            functionalized dropout
            Args:
                x: (batch_size, padded_seq_len, hidden_dim)
        """
        if (not self.training) or (not p):
            return x
        # mask
        mask = x.new_empty(
            x.size(0), 1, x.size(2), requires_grad=False
        ).bernoulli_(1 - p)
        mask = mask.div_(1 - p)
        # expand along the second axis
        mask = mask.expand_as(x)
        return x * mask
        

    def forward(self, x, lx):
        """
            Args:
                x: (batch_size, padded_seq_len, hidden_dim)
                    expected to be unpacked before feeding in
                    lx: (batch_size, )
        """
        for i, plstm in enumerate(self.plstms):
            # sizes
            batch_size, enc_seq_len, hidden_dim = x.size()
            # dropout specification
            p = self.mid_dropout if i < self.plstm_layers - 1 else self.final_dropout
            # add one extra time frame if odd number
            if x.size(1) % 2 != 0:
                # """
                #     Choice 1: concat a dead vector
                # """
                # # x = torch.cat([
                # #     x, torch.zeros((batch_size, 1, hidden_dim), device=x.device,
                # #                     requires_grad=False)], dim=1)
                """
                    Choice 2: abandon the 1 extra timestep
                """
                x = x[:, :-1, :]
            # lengths change
            lx, enc_seq_len = lx // 2, enc_seq_len // 2
            # reshape
            x = x.reshape(batch_size, enc_seq_len, 2 * hidden_dim)
            # pack
            x = pack_padded_sequence(x, lengths=lx, batch_first=True, enforce_sorted=False)
            # lstm
            x = plstm(x)[0]
            # pad
            x, lx = pad_packed_sequence(x, batch_first=True)
            # dropout
            x = self.locked_dropout(x, p)
        return x, lx



class AutoRegDecoderLSTMCell(nn.Module):
    """
        Autoregressive decoder LSTM cells with cross attention contexts.
            Note: 2 layers version only
    """
    def __init__(
        self,
        att_proj_dim: int=128,
        dec_emb_dim: int=256,
        dec_hid_dim: int=512,
        dec_out_dim: int=128,
        dec_mid_dropout: float=0.2
    ):
        super().__init__()
        self.att_proj_dim = att_proj_dim
        self.dec_emb_dim = dec_emb_dim
        self.dec_hid_dim = dec_hid_dim
        self.dec_out_dim = dec_out_dim
        self.dec_mid_dropout = dec_mid_dropout

        # original lstms
        self.lstms = nn.ModuleList([
            nn.LSTMCell(
                # input: [context, char_embedding]
                input_size=self.att_proj_dim + self.dec_emb_dim,
                hidden_size=self.dec_hid_dim,
            ),
            nn.LSTMCell(
                # input: [hidden_state, cell_state]
                input_size=self.dec_hid_dim * 2,
                hidden_size=self.dec_out_dim
            )
        ])


    def locked_dropout(self, x, p: float=0.0):
        if (not self.training) or (not p):
            return x
        mask = x.new_empty(
            1, x.size(1), requires_grad=False
        ).bernoulli_(1 - p).div_(1 - p).expand_as(x)
        return x * mask


    def forward(self, prev_e, prev_c, prev_h):
        """
            Args:
                prev_e: (batch_size, dec_emb_dim) character embeddings, previous step
                prev_c: (batch_size, proj_dim) context, previous step
                prev_h: List[(batch_size, dec_hid_dim)] hidden states for all layers, previous step
        """
        # original inputs: character emb & context from last time step
        prev_ec = torch.cat([prev_e, prev_c], dim=1)                   
        # (batch_size, dec_emb_dim + proj_dim)

        # iterate
        for i in range(len(self.lstms)):
            prev_h[i] = self.lstms[i](prev_ec, prev_h[i])
            # apply only to the output of the 1st lstm cell
            if i == 0:
                # encode respectivly
                prev_ec = self.locked_dropout(torch.cat(
                    prev_h[i], dim=-1
                ), self.dec_mid_dropout)

        return prev_h
        # List[(h, c)]
        # (batch_size, dec_out_dim), (batch_size, dec_out_dim)
