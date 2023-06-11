import torch
from torch import nn
import argparse
from torch.utils.tensorboard import SummaryWriter
from eval import evaluate
from data import CustomSNLIDataset, collate_batch
from models import NLIModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, optimizer, criterion, train_loader):
    total_loss, total_count = 0, 0
    model.train()
    for (premise, premise_len), (hypothesis, hypothesis_len), label in train_loader:
        premise = premise.to(device)
        hypothesis = hypothesis.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        predicted_label = model((premise, premise_len), (hypothesis, hypothesis_len))
        loss = criterion(predicted_label, label)
        loss.backward()
        optimizer.step()

        total_loss += loss
        total_count += 1
    return total_loss / total_count


def train_model(model, epochs, lr, min_lr, weight_decay, max_norm, train_loader, val_loader, encoder):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    writer = SummaryWriter(f"runs/{current_time}_{encoder}")

    print("using device:", device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    scheduler1 = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=weight_decay
    )
    scheduler2 = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=1/max_norm
    )

    val_acc, prev_val_acc = 0, 0
    for epoch in range(epochs):
        print("Training Epoch:", epoch)
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion=criterion)
        writer.add_scalar("Loss/val", val_loss, epoch)

        if val_acc < prev_val_acc:
            scheduler2.step()
        else:
            scheduler1.step()
        prev_val_acc = val_acc
        print(f"Finished Training Epoch: {epoch}, Train/Loss: {train_loss}, Val/Loss: {val_loss}")
        print(f"Validation Accuracy: {val_acc}")
        if optimizer.param_groups[0]["lr"] <= min_lr:
            break
    writer.close()
    torch.save(model, f"models/final_{encoder}_model.pt")

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Argument Parser")
    # Data options
    parser.add_argument("--sort_data", type=bool, default=False, 
                        help="Sort training data based on sentence length",
    )

    # Training options
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--max_norm", type=int, default=5)

    # Model options
    parser.add_argument("--word_embed_dim", type=int, default=300)
    parser.add_argument("--encoder", type=str, default='BaselineEncoder')
    parser.add_argument("--enc_h_dim", type=int, default=2048)
    parser.add_argument("--enc_n_layers", type=int, default=1)
    parser.add_argument("--fc_h_dim", type=int, default=512)
    parser.add_argument("--n_classes", type=int, default=3)

    args = parser.parse_args()

    print("Loading Training Data...")
    train, val = (
        CustomSNLIDataset(split="train"),
        CustomSNLIDataset(split="validation"),
    )
    print("Finished Loading Training Data...")
    train_dataloader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.n_workers,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.n_workers,
        pin_memory=True,
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Initialising Model...")
    model = NLIModel(
        args.word_embed_dim,
        args.fc_h_dim,
        args.n_classes,
        args.encoder,
        args.enc_n_layers,
        args.enc_h_dim,
    ).to(device)

    print("Training Model...")
    train_model(
        model=model, 
        epochs=args.n_epochs, 
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.decay,
        max_norm=args.max_norm,
        train_loader=train_dataloader, 
        val_loader=valid_dataloader,
        encoder=args.encoder,
    )
    
    # test_acc = evaluate(model, test_dataloader)
    # print("Final Test Accuracy:", test_acc)
