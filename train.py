import torch
from torch import nn
import argparse
from torch.utils.tensorboard import SummaryWriter
from eval import evaluate
from data import CustomSNLIDataset, collate_batch
from models import NLIModel
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model, optimizer, criterion, train_loader, writer):
    total_loss, total_count = 0, 0
    for premise, hypothesis, label in tqdm(train_loader):
        optimizer.zero_grad()
        predicted_label = model(premise, hypothesis)

        loss = criterion(predicted_label, label)
        loss.backward()
        optimizer.step()

        total_loss += loss
        total_count += 1
    return total_loss / total_count


def train_model(model, epochs, lr, min_lr, weight_decay, max_norm, train_loader, val_loader, writer, encoder):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    scheduler1 = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=weight_decay
    )
    scheduler2 = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=1/max_norm
    )

    val_acc, prev_val_acc = 0, 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, writer)
        writer.add_scalar("Loss/train", train_loss, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion=criterion)
        writer.add_scalar("Loss/val", val_loss, epoch)
        # divide by 5 if dev accuracy decreases
        if val_acc < prev_val_acc:
            scheduler2.step()
        else:
            scheduler1.step()
        prev_val_acc = val_acc
        if optimizer.param_groups[0]["lr"] <= min_lr:
            break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "loss": val_loss,
        },
       f"{encoder}_model.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser")
    # Data options
    parser.add_argument(
        "--sort_data",
        type=bool,
        default=False,
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

    train, val, test = (
        CustomSNLIDataset(split="train", sort=args.sort_data),
        CustomSNLIDataset(split="validation"),
        CustomSNLIDataset(split="test"),
    )
    train_dataloader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.n_workers,
    )
    valid_dataloader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.n_workers,
    )
    test_dataloader = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.n_workers,
    )

    model = NLIModel(
        args.word_embed_dim,
        args.fc_h_dim,
        args.n_classes,
        args.encoder,
        args.enc_n_layers,
        args.enc_h_dim,
    )

    writer = SummaryWriter()

    train_model(
        model=model, 
        epochs=args.n_epochs, 
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.decay,
        max_norm=args.max_norm,
        train_loader=train_dataloader, 
        val_loader=valid_dataloader,
        writer=writer,
        encoder=args.encoder,
    )
    val_acc = evaluate(model, test_dataloader)
    print(val_acc)
