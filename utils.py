import torch 
from torchtext.vocab import GloVe

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

glove = GloVe(name='840B', dim=300)
text_pipeline = lambda x: glove.get_vecs_by_tokens(x, lower_case_backup=True)


def collate_batch(batch):
    premises, hypotheses, labels = [], [], []
    for premise, hypothesis, label in batch:
        premise = text_pipeline(premise)
        hypothesis = text_pipeline(hypothesis)
        hypotheses.append(hypothesis)
        premises.append(premise)
        labels.append(label)
    premises_len = [len(premise) for premise in premises]
    hypotheses_len = [len(hypothesis) for hypothesis in hypotheses]
    premises_padded = torch.nn.utils.rnn.pad_sequence(premises, batch_first=True)
    hypotheses_padded = torch.nn.utils.rnn.pad_sequence(hypotheses, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.int32)
    return (premises_padded.to(device), premises_len), (hypotheses_padded.to(device), hypotheses_len), labels.to(device)