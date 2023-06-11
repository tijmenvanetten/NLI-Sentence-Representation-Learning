# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
# import sklearn
from pathlib import Path
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set PATHs
# path to senteval
PATH_TO_SENTEVAL = 'SentEval/'
# path to the NLP datasets 
PATH_TO_DATA = 'SentEval/data/'
# path to glove embeddings
PATH_TO_VEC = '.vector_cache/glove.840B.300d.txt'


# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

from senteval.bow import create_dictionary, get_wordvec


def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
    _, params.word2id = create_dictionary(samples)
    # load glove/word2vec format 
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    # dimensionality of glove embeddings
    params.wvec_dim = 300
    return

def batcher_model(params, batch):
    return batcher(params, model, batch)

def batcher(params, model, batch):
    """
    In this example we use an encoder model for sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        # the format of a sentence is a lists of words (tokenized and lowercased)
        for word in sent:
            if word in params.word_vec:
                # [number of words, embedding dimensionality]
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            # [number of words, embedding dimensionality]
            sentvec.append(vec)
    
        sentvec = torch.tensor(np.array(sentvec),  dtype=torch.float32)[None, :]
        sent_len = [len(sent) for sent in sentvec]
        sent_input = (sentvec.to(device), sent_len)

        sentvec = model.encoder(sent_input).detach().cpu()
        embeddings.append(sentvec)
    # [batch size, embedding dimensionality]
    embeddings = torch.vstack(embeddings)
    return embeddings


# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
# for example the path to a dictionary of indices, of hyper parameters
# this dictionary is passed to the batched and the prepare fucntions
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False 
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")

    # Specify Model
    parser.add_argument("--model", type=str, default="models/BiLSTMMaxPoolEncoder_model.pt")

    args = parser.parse_args()

    # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                    'MRPC', 'SICKEntailment', 'STS14']
    transfer_tasks_report = transfer_tasks[:-2]
    print(transfer_tasks_report)
    # senteval prints the results and returns a dictionary with the scores

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model)
    model = torch.load(model_path, map_location=device)

    se = senteval.engine.SE(params_senteval, batcher_model, prepare)
    
    results = se.eval(transfer_tasks)
    print(results)
    micro, macro = 0, 0
    for task in transfer_tasks_report:
        micro += results[task]['devacc']/results[task]['ndev']
        macro += results[task]['devacc']
    print("Micro:", micro, "- Macro:", macro/len(transfer_tasks_report))