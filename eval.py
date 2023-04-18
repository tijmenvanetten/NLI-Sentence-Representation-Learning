import torch 
from models import NLIModel

def evaluate(model, dataloader, criterion=None):
    with torch.no_grad():
        for premise, hypothesis, label in dataloader:
            output = model(premise, hypothesis)
            predicted_label = torch.argmax(output, axis=1)
            accuracy = torch.mean(predicted_label == label)
        if criterion:
            loss = criterion(output, label)
            return loss, accuracy
        return accuracy

if __name__ == "__main__":
    model = NLIModel()
    model.load_state_dict("model.pt")
    print(evaluate(model))