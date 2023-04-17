import torch 
from torch import nn 
import argparse
from torch.utils.tensorboard import SummaryWriter
from eval import evaluate
from data import CustomSNLIDataset
from models import NLIModel
from torch.utils.data import DataLoader
from utils import collate_batch
from tqdm import tqdm

writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, optimizer, criterion, train_loader):
    total_loss, total_count = 0, 0
    for premise, hypothesis, label in tqdm(train_loader):
        optimizer.zero_grad()
        predicted_label = model(premise, hypothesis)

        loss = criterion(predicted_label, label)
        loss.backward()
        optimizer.step()

        total_loss += loss
        total_count += 1
    writer.add_scalar("Loss/train", total_loss/total_count)

def train_model(model, train_loader, dev_loader):

    weight = torch.FloatTensor(model.n_classes).fill_(1)
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

    prev_val_acc = 0
    epochs = 1
    while optimizer.param_groups[0]['lr'] >= 1e-4:
        train_epoch(model, optimizer, criterion, train_loader)
        print("Epochs:", epochs)
        epochs +=1 
        val_loss, val_acc = evaluate(model, dev_loader)
        writer.add_scalar("Loss/val", val_loss, epoch=epochs)
        # divide by 5 if dev accuracy decreases
        if val_acc < prev_val_acc:
            scheduler2.step()
        else:
            scheduler1.step()
        prev_val_acc = val_acc
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            }, "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser")

    # Training options
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="optimizer for training, e.g. adam or sgd with learning rate")
    parser.add_argument("--decay", type=float, default=0.99, help="learning rate decay")
    parser.add_argument("--minlr", type=float, default=1e-5, help="minimum learning rate")
    parser.add_argument("--max_norm", type=float, default=5., help="maximum gradient norm (for gradient clipping)")

    # Model options
    parser.add_argument("--word_embed_dim", type=int, default=300, help="dimension of word embedding space")
    parser.add_argument("--encoder", type=str, default='BaselineEncoder', help="type of encoder to use: ")
    parser.add_argument("--enc_h_dim", type=int, default=2048, help="dimension of encoder hidden states")
    parser.add_argument("--enc_n_layers", type=int, default=1, help="number of encoder layers")
    parser.add_argument("--fc_h_dim", type=int, default=512, help="dimension of fully connected layers")
    parser.add_argument("--n_classes", type=int, default=3, help="number of classes for classification (entailment/neutral/contradiction)")

    args = parser.parse_args()

    train, val, test= CustomSNLIDataset(split='train'), CustomSNLIDataset(split='validation'), CustomSNLIDataset(split='test'), 
    train_dataloader = DataLoader(train, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(val, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_batch)
    
    model = NLIModel(
        args.word_embed_dim,
        args.fc_h_dim,
        args.n_classes,
        args.encoder,
        args.enc_n_layers,
        args.enc_h_dim,
        ).to(device)

    train_model(model, train_dataloader, valid_dataloader)
    val_acc = evaluate(model, test_dataloader)
    print(val_acc)


