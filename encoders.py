import torch 
from torch import nn
import numpy as np


class BaselineEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, sentences):
        sent_unpadded = [embedding[:pad_idx] for embedding, pad_idx in zip(*sentences)]
        return torch.stack([torch.mean(sentence, axis=0) for sentence in sent_unpadded])
    

class LSTMEncoder(nn.Module):
    def __init__(self, word_embed_dim, enc_n_layers, enc_h_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.word_embed_dim = word_embed_dim
        self.enc_n_layers = enc_n_layers
        self.enc_h_dim = enc_h_dim
        self.lstm = nn.LSTM(
            input_size=self.word_embed_dim,
            hidden_size=self.enc_h_dim,
            num_layers=self.enc_n_layers,
        )
        
    def forward(self, sentences_input):
        sent, sent_len = sentences_input

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len, batch_first=True, enforce_sorted=False)
        sent_output, (h_n, c_n) = self.lstm(sent_packed) 
        return h_n.squeeze(0)
    

class BiLSTMEncoder(LSTMEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(
            input_size=self.word_embed_dim,
            hidden_size=self.enc_h_dim,
            num_layers=self.enc_n_layers,
            bidirectional=True
        )
    
    def forward(self, sentences_input):
        sent, sent_len = sentences_input

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len, batch_first=True, enforce_sorted=False)
        sent_output,_ = self.lstm(sent_packed) 
        sent_output,_ = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)

        sent_output = [embedding[:pad_idx][-1] for embedding, pad_idx in zip(sent_output, sent_len)]
        return torch.stack(sent_output)
    
class BiLSTMMaxPoolEncoder(BiLSTMEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, sentences):
        sentences, sent_lens = sentences
        sent_packed = nn.utils.rnn.pack_padded_sequence(sentences, sent_lens, batch_first=True, enforce_sorted=False)
        sent_output,_ = self.lstm(sent_packed) 
        sent_output,_ = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)

        # remove padding for max pooling
        sent_output = [embedding[:pad_idx] for embedding, pad_idx in zip(sent_output, sent_lens)]
        sent_output_max_pool = torch.stack([torch.max(x, 0)[0] for x in sent_output], 0)
        return sent_output_max_pool
    
