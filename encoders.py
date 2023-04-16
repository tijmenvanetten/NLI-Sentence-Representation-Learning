import torch 
from torch import nn
import numpy as np


class BaselineEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, sentences):
        return torch.mean(sentences, axis=1)
    

class LSTMEncoder(nn.Module):
    def __init__(self, word_embed_dim, enc_n_layers, enc_h_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.word_embed_dim = word_embed_dim
        self.enc_n_layers = enc_n_layers
        self.enc_h_dim = enc_h_dim
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=self.word_embed_dim,
            hidden_size=self.enc_h_dim,
            num_layers=self.enc_n_layers,
            dropout=self.dropout
        )
        
    def forward(self, sentences_input):
        sent, sent_lens_sorted = sentences_input

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_lens_sorted)
        sent_output, _ = self.lstm(sent_packed) 
        sent_output, _ = nn.utils.rnn.pad_packed_sequence(sent_output)

        return sent_output
    

class BiLSTMEncoder(LSTMEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(
            input_size=self.word_embed_dim,
            hidden_size=self.enc_h_dim,
            num_layers=self.enc_n_layers,
            bidirectional=True
        )
    
class BiLSTMMaxPoolEncoder(BiLSTMEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, sentences):
        sentences, sent_lens = sentences
        sent_packed = nn.utils.rnn.pack_padded_sequence(sentences, sent_lens)
        sent_output,_ = self.lstm(sent_packed) 
        sent_output,_ = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)

        # remove padding
        sent_output = [embedding[:pad_idx] for embedding, pad_idx in zip(sent_output, sent_lens)]
        sent_output_max_pool = torch.stack([torch.max(x, 0)[0] for x in sent_output], 0)
        return sent_output_max_pool
    
