"""
    Constant variables for output symbols.
    Expected not to be changed.
"""

VOCAB = ['<sos>',   
         'A',   'B',    'C',    'D',    
         'E',   'F',    'G',    'H',    
         'I',   'J',    'K',    'L',       
         'M',   'N',    'O',    'P',    
         'Q',   'R',    'S',    'T', 
         'U',   'V',    'W',    'X', 
         'Y',   'Z',    "'",    ' ', 
         '<eos>']

VOCAB_MAP = {VOCAB[i]: i for i in range(len(VOCAB))}

SOS_IDX = VOCAB_MAP["<sos>"]
EOS_IDX = VOCAB_MAP["<eos>"]