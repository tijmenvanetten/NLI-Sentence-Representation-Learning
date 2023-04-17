from datasets import load_dataset
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
import torch


def preprocess(example):
    example['premise'] = word_tokenize(example['premise'].lower())
    example['hypothesis'] = word_tokenize(example['hypothesis'].lower())
    example['premise_len'] = len(example['premise'])
    example['hypothesis_len'] = len(example['hypothesis'])
    example['total_len'] = example['premise_len'] + example['hypothesis_len']
    return example

class CustomSNLIDataset(Dataset):
    def __init__(self, split='train', sort=False) -> None:
        data = load_dataset("snli", split=split)
        self.data = data.with_format("torch").filter(lambda example: example['label'] >= 0).map(preprocess)
        if sort:
            self.sort(sort)

    def sort(self, label):
        self.data = self.data.sort(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['premise'], self.data[idx]['hypothesis'], self.data[idx]['label']