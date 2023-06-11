# Advanced Topics in Computational Semantics - Practical 1: SNLI Task

This is an attempt to reproduce the results from the "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data"  (https://arxiv.org/abs/1705.02364). 
It uses [SentEval](https://github.com/facebookresearch/SentEval) to automatically evaluate the quality of the sentence representations on a wide range of tasks. 

## Dependencies

The required dependencies can be installed via anaconda

```bash
conda env create -f environment_gpu.yml
conda activate atcs
```

```bash
git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/
python setup.py install
cd data/downstream/
./get_transfer_data.bash
```

trained models can be found here: https://drive.google.com/file/d/19lHYk0uRKvkyZyYv1RfUn4XXdB_Y3Xxj/view?usp=sharing

## Setup

.  
├── models/             # Folder for the models trained on the SNLI dataset  
├── runs/               # Tensorboard logs of each training run  
├── environment_gpu.yml # Conda environment  
├── data.py             # Contains the CustomSNLIDataset used to load the SNLI data into a PyTorch Dataloader  
├── encoders.py         # Implementation of the different encoders used for the   
├── model.py            # Contains the PyTorch implementation of the NLIModel  
├── train.py            # Training function used to train the NLIModel on the SNLIDataset  
├── eval.py             # Evaluate the accuracy of the pretrained models on the SNLI training task  
├── senteval.py         # Evaluates the accuracy of the pretrained models on the SentEval/ task  
├── utils.py            # Contains utility function for inference preprocessing  
├── Results.ipynb       # Shows final results and error analysis  
└── README.md  


## Glove Embedding

The torchtext package was to load the GloVe Embeddings. This package comes with a built in dictionary and can directly map tokens to word embeddings. This allows us to still use embeddings for some of the tokens present in the test and validation set without having seen them in the training data, allowing for more generalizing capability leading to possibly better results. Be sure to change the SentEval glove path to the .vector_cache location to prevent unnecessary downloads.

### Training on the SNLI dataset

```bash
usage: python train.py [-h] [--sort_data SORT_DATA] [--n_epochs N_EPOCHS] [--n_workers N_WORKERS] [--batch_size BATCH_SIZE] [--decay DECAY] [--lr LR]
                [--min_lr MIN_LR] [--max_norm MAX_NORM] [--word_embed_dim WORD_EMBED_DIM] [--encoder ENCODER] [--enc_h_dim ENC_H_DIM]
                [--enc_n_layers ENC_N_LAYERS] [--fc_h_dim FC_H_DIM] [--n_classes N_CLASSES]
```

### Evaluation of SNLI dataset
```bash
usage: python eval.py [-h] [--model MODEL] [--batch_size BATCH_SIZE] [--n_workers N_WORKERS]
```

### Evaluation of SentEval
```bash
usage: python senteval.py [-h] [--model MODEL]
```

We obtained the following results:

## Original Paper
(https://arxiv.org/abs/1705.02364)
| **Model**  |         | **NLI** |          | **Transfer** |           |
|------------|---------|:-------:|----------|:------------:|-----------|
|            | **dim** | **dev** | **test** | **micro**    | **macro** |
| Baseline   | 300     | -       | -        | -            | -         |
| LSTM       | 2048    | 81.9    | 80.7     | 79.5         | 78.6      |
| BiLSTM-Last| 4096    | -       | -        | -            | -         |
| BiLSTM-Max | 4096    | 85      | 84.5     | 85.2         | 83.7      |

## Reproduction implementation

| **Model**  |         | **NLI** |          | **Transfer** |           |
|------------|---------|:-------:|----------|:------------:|-----------|
|            | **dim** | **dev** | **test** | **micro**    | **macro** |
| Baseline   | 300     | 65.3    | 65.3     | 78.3         | 76.7      |
| LSTM       | 2048    | 79.3    | 80.0     |   82.8       |  81.0     |
| BiLSTM-Last| 4096    | 80.9    | 80.6     | 78.8         | 77.2      |
| BiLSTM-Max | 4096    | 83.5    | 83.5     |   83.6       |    82.0   |

We demonstrate similar results to the original with only small margins of deviation paper, indicating strong reproducibility 

## Curriculum Learning

Curriculum learning (CL) is a powerful technique that has proven effective in improving the performance of NLP models. This approach involves training models on a curriculum, gradually increasing the complexity of the tasks they are asked to perform. 

The way complexity is defined in this setting is by the total sentence length (number of tokens in the premise + number of tokens in the entailment). This assumes that longer sentences are harder to predict entailment for than for shorter sentences.

```bash
usage: python train.py [-h] [--encoder ENCODER] --sort_data=True
```

#### Results
| **Model**  | **NLI** |          | **Transfer** |           |
|------------|:-------:|----------|:------------:|-----------|
|            | **dev** | **test** | **micro**    | **macro** |
| LSTM       | 79.3    | 80.0     |   82.8       |  81.0     |
| LSTM+CL    | 79.9    | 79.8     | 79.5| 77.9
| BiLSTM-Last| 80.9    | 80.6     | 78.8         | 77.2      |
| BiLSTM-Last+CL  | 80.5    | 80.2 |78.5|77.1
| BiLSTM-Max | 83.5    | 83.5     |   83.6       |    82.0   |
| BiLSTM-Max+CL       | 83.6    | 83.2     |   83.5       |  82.1    |

The results show no indication that this type of curriculum learning improved the sentence representations in any way. While, accuracy has improved for the LSTM and BiLSTM + Max-Pooling on the NLI dev set, this is not the case for the other tests. The contrary actually happens for the LSTM on the transfer tasks. However, because the differences are not significant, they could also be due to randomnes in the initialisation.

### Supervised Learning of Universal Sentence Representations from Natural Language Inference Data (EMNLP 2017, Outstanding Paper Award)

[1] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, A. Bordes, [*Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*](https://arxiv.org/abs/1705.02364)

```
@article{conneau2017supervised,
  title={Supervised Learning of Universal Sentence Representations from Natural Language Inference Data},
  author={Conneau, Alexis and Kiela, Douwe and Schwenk, Holger and Barrault, Loic and Bordes, Antoine},
  journal={arXiv preprint arXiv:1705.02364},
  year={2017}
}
```
