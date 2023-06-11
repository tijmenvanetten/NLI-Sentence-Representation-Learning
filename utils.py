from torchtext.vocab import GloVe
from nltk.tokenize import word_tokenize
import torch

glove = GloVe(name="840B", dim=300)
text_pipeline = lambda x: glove.get_vecs_by_tokens(x)

label_to_class = {
    0 : "Entailment",
    1 : "Contradiction",
    2 : "Neutral"
}

def preprocess(sentence: str) -> str:
    tokens = word_tokenize(sentence)
    return glove.get_vecs_by_tokens(tokens)[None, :], torch.tensor([len(tokens)])


if __name__ == "__main__":

    baseline_transfer = {'MR': {'devacc': 78.03, 'acc': 77.0, 'ndev': 10662, 'ntest': 10662}, 
                         'CR': {'devacc': 80.55, 'acc': 78.62, 'ndev': 3775, 'ntest': 3775}, 
                         'MPQA': {'devacc': 87.86, 'acc': 87.53, 'ndev': 10606, 'ntest': 10606}, 
                         'SUBJ': {'devacc': 91.66, 'acc': 91.45, 'ndev': 10000, 'ntest': 10000}, 
                         'SST2': {'devacc': 79.59, 'acc': 79.68, 'ndev': 872, 'ntest': 1821}, 
                         'TREC': {'devacc': 74.78, 'acc': 82.8, 'ndev': 5452, 'ntest': 500}, 
                         'MRPC': {'devacc': 73.8, 'acc': 73.16, 'f1': 81.69, 'ndev': 4076, 'ntest': 1725}, 
    }
    lstm_transfer = {'MR': {'devacc': 73.24, 'acc': 72.21, 'ndev': 10662, 'ntest': 10662}, 
                     'CR': {'devacc': 78.38, 'acc': 75.26, 'ndev': 3775, 'ntest': 3775}, 
                     'MPQA': {'devacc': 87.36, 'acc': 87.8, 'ndev': 10606, 'ntest': 10606}, 
                     'SUBJ': {'devacc': 85.44, 'acc': 85.28, 'ndev': 10000, 'ntest': 10000}, 
                     'SST2': {'devacc': 78.78, 'acc': 75.84, 'ndev': 872, 'ntest': 1821}, 
                     'TREC': {'devacc': 61.1, 'acc': 71.2, 'ndev': 5452, 'ntest': 500}, 
                     'MRPC': {'devacc': 72.82, 'acc': 72.12, 'f1': 81.86, 'ndev': 4076, 'ntest': 1725},
    } 
