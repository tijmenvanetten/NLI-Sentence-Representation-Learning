import torch 

def evaluate(model, dataloader, criterion=None):
    model.eval()
    with torch.no_grad():
        for premise, hypothesis, label in enumerate(dataloader):
            predicted_label = model(premise, hypothesis)
            accuracy = torch.sum(predicted_label == label) / len(predicted_label)
    if criterion:
        loss = criterion(predicted_label, label)
        return loss, accuracy
    return accuracy