from datasets import load_dataset
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import torch
from torchtext.vocab import GloVe

glove = GloVe(name="840B", dim=300)
text_pipeline = lambda x: glove.get_vecs_by_tokens(x)


def preprocess(example):
    example["premise"] = text_pipeline(word_tokenize(example["premise"].lower()))
    example["premise_len"] = len(example["premise"])
    example["hypothesis"] = text_pipeline(word_tokenize(example["hypothesis"].lower()))
    example["hypothesis_len"] = len(example["hypothesis"])
    example["total_len"] = len(example["premise"]) + len(example["hypothesis"])
    return example


def collate_batch(batch):
    premises, hypotheses, labels = [], [], []
    premises_len, hypotheses_len = [], []
    for (premise, premise_len), (hypothesis, hypothesis_len), label in batch:
        hypotheses.append(hypothesis)
        hypotheses_len.append(hypothesis_len)

        premises_len.append(premise_len)
        premises.append(premise)

        labels.append(label)

    premises_padded = torch.nn.utils.rnn.pad_sequence(premises, batch_first=True)
    hypotheses_padded = torch.nn.utils.rnn.pad_sequence(hypotheses, batch_first=True)
    labels = torch.stack(labels)
    premises_len = torch.stack(premises_len)
    hypotheses_len = torch.stack(hypotheses_len)
    return (premises_padded, premises_len), (hypotheses_padded, hypotheses_len), labels


class CustomSNLIDataset(Dataset):
    def __init__(self, split="train", sort=False) -> None:
        if sort:
            self.data = load_dataset("snli", split=split).with_format("torch") \
            .filter(lambda example: example["label"] >= 0) \
            .map(preprocess).sort("total_len")
        else:
            self.data = load_dataset("snli", split=split).with_format("torch") \
            .filter(lambda example: example["label"] >= 0) \
            .map(preprocess)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            (self.data[idx]["premise"], self.data[idx]["premise_len"]),
            (self.data[idx]["hypothesis"], self.data[idx]["hypothesis_len"]),
            self.data[idx]["label"],
        )


if __name__ == "__main__":
    data = CustomSNLIDataset(split="test")
    dataloader = DataLoader(data, collate_fn=collate_batch, batch_size=64)
    for (premise, premise_len), (hypothesis, hypothesis_len), label in dataloader:
        print(premise.shape)
        break
