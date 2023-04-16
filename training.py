import torch 
from torch import nn 
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate
import tqdm
writer = SummaryWriter()

def train_epoch(model, optimizer, criterion, train_loader):
    total_loss, total_count = 0, 0
    for premise, hypothesis, label, *_ in train_loader:
        optimizer.zero_grad()
        predicted_label = model(premise, hypothesis)
        loss = criterion(predicted_label, torch.abs(label))
        loss.backward()
        optimizer.step()

        total_loss += loss
        total_count += 1
    writer.add_scalar("Loss/train", total_loss/total_count)

def train_model(model, train_loader, dev_loader):

    weight = torch.FloatTensor(model.n_classes).fill_(1)
    criterion = nn.CrossEntropyLoss(weight=weight)
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
        writer.add_scalar("Loss/val", val_loss)
        # divide by 5 if dev accuracy decreases
        if val_acc < prev_val_acc:
            scheduler2.step()
        else:
            scheduler1.step()
        prev_val_acc = val_acc


