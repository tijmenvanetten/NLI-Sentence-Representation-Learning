import torch 
from torch import nn
from encoders import BaselineEncoder, LSTMEncoder, BiLSTMEncoder, BiLSTMMaxPoolEncoder



encoders_dict =  {
    'BaselineEncoder': BaselineEncoder, 
    'LSTMEncoder': LSTMEncoder, 
    'BiLSTMEncoder': BiLSTMEncoder,
    'BiLSTMMaxPoolEncoder': BiLSTMMaxPoolEncoder
    }

class NLIModel(nn.Module):
    def __init__(self, word_embed_dim, fc_h_dim, n_classes, encoder, enc_n_layers, enc_h_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.word_embed_dim = word_embed_dim
        self.fc_h_dim = fc_h_dim
        self.n_classes = n_classes
        self.encoder = encoders_dict[encoder](word_embed_dim, enc_n_layers, enc_h_dim,)

        self.classifier = nn.Sequential(
            nn.Linear(self.word_embed_dim * 4, self.fc_h_dim),
            nn.Linear(self.fc_h_dim, self.fc_h_dim),
            nn.Linear(self.fc_h_dim, self.n_classes)
        )
        self.normalize = nn.Softmax(dim=1)


    def forward(self, premise, hypothesis):
        u, v = self.encoder(premise), self.encoder(hypothesis)
        diff_u_v = torch.abs(u - v)
        cross_u_v = u * v 
        linear_input = torch.concat([u, v, diff_u_v, cross_u_v], axis=1)
        linear_output = self.classifier(linear_input)
        return self.normalize(linear_output)
    