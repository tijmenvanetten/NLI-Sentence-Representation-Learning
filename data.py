from datasets import load_dataset
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
import torch 
from torchtext.vocab import GloVe


glove = GloVe(name='840B', dim=300)
text_pipeline = lambda x: glove.get_vecs_by_tokens(x)

def preprocess(example):
    example['premise'] = word_tokenize(example['premise'].lower())
    example['hypothesis'] = word_tokenize(example['hypothesis'].lower())
    example['total_len'] = len(example['premise']) + len(example['hypothesis'])
    return example

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
    labels = torch.stack(labels)
    return (premises_padded, premises_len), (hypotheses_padded, hypotheses_len), labels

class CustomSNLIDataset(Dataset):
    def __init__(self, split='train', sort=False) -> None:
        data = load_dataset("snli", split=split)
        self.data = data.with_format("torch").filter(lambda example: example['label'] >= 0).map(preprocess)
        if sort:
            self.data.sort('total_len')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['premise'], self.data[idx]['hypothesis'], self.data[idx]['label']
    
if __name__ == "__main__":
    data = CustomSNLIDataset(split='test')
    print(data[0])