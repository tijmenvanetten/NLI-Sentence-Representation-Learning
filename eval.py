import torch 
from data import CustomSNLIDataset, collate_batch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataloader, criterion=None):
    model.eval()
    with torch.no_grad():
        acc_total, loss_total, no_batches = 0, 0, 0
        for (premise, premise_len), (hypothesis, hypothesis_len), label in dataloader:
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            label = label.to(device)
            output = model((premise, premise_len), (hypothesis, hypothesis_len))

            loss = criterion(output, label) if criterion else 0

            predicted_label = torch.argmax(output, axis=1)
            accuracy = torch.mean((predicted_label == label).float())

            acc_total += accuracy 
            loss_total += loss 
            no_batches += 1
        return loss_total / no_batches, acc_total / no_batches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser")
    # Data options
    parser.add_argument("--model", type=str, default="models/LSTMEncoder_model.pt", 
                        help="Set Model name",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=0)

    args = parser.parse_args()

    dev = CustomSNLIDataset(split="validation")
    dev_dataloader = DataLoader(
        dev,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    test = CustomSNLIDataset(split="test")
    test_dataloader = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.model, map_location=device)

    print("Dev Accuracy:", evaluate(model, dev_dataloader)[1])
    print("Test Accuracy:", evaluate(model, test_dataloader)[1])