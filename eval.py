import torch 
from models import NLIModel

def evaluate(model, dataloader, criterion=None):
    model.eval()
    with torch.no_grad():
        for premise, hypothesis, label in dataloader:
            predicted_label = model(premise, hypothesis)
            accuracy = torch.sum(torch.argmax(predicted_label, axis=1) == label) / len(predicted_label)
    if criterion:
        loss = criterion(predicted_label, label)
        return loss, accuracy
    return accuracy

if __name__ == "__main__":
    model = NLIModel()
    model.load_state_dict("model.pt")
    print(evaluate(model))