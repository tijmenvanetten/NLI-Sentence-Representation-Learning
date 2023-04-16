import argparse
from models import NLIModel
from data import CustomSNLIDataset
from torch.utils.data import DataLoader
from utils import collate_batch
from training import train_model
from evaluation import evaluate

def main(args):
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
        )

    train_model(model, train_dataloader, valid_dataloader)
    val_acc = evaluate(model, test_dataloader)
    print(val_acc)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser")

    # Training options
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fully connected layers")
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="optimizer for training, e.g. adam or sgd with learning rate")
    parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for SGD optimizer")
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

    encoders = ['BaselineEncoder', 'LSTMEncoder', 'BiLSTMEncoder','BiLSTMMaxPoolEncoder']
    assert args.encoder in encoders, "encoder must be one of " + str(encoders)

    main(args)