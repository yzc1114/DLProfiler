import argparse

import torch
import torch.nn as nn
import torch.onnx
import pathlib
from torch.nn.parallel import DistributedDataParallel as DDP

from model_factory.LSTM.word_language_model import data
from profiler_utils import ProfileIterator


def load_model():
    from model_factory.LSTM.word_language_model.model import RNNModel
    args_data = pathlib.Path(__file__).parent / "data" / "wikitext-2"
    args_data = str(args_data)
    corpus = data.Corpus(args_data)
    args_emsize = 200
    args_nhid = 200
    args_nlayers = 2
    args_dropout = 0.2
    args_tied = True
    ntokens = len(corpus.dictionary)
    model = RNNModel("LSTM", ntokens, args_emsize, args_nhid, args_nlayers, args_dropout, args_tied)
    return model

def do_profile(batch_size=32, duration_sec=10, device="0", is_train=True):
    args_data = pathlib.Path(__file__).parent / "data" / "wikitext-2"
    args_data = str(args_data)

    args_seed = 1111
    args_bptt = 35

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args_seed)

    corpus = data.Corpus(args_data)

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    train_data = batchify(corpus.train, batch_size)
    model = load_model()
    hidden = model.init_hidden(batch_size)

    def gen_hidden(hidden_):
        ls = []
        for e in hidden_:
            ls.append(torch.rand_like(e, device=device))
        return tuple(ls)

    hidden = gen_hidden(hidden)
    model = model.to(device)
    model = DDP(model)

    criterion = nn.NLLLoss()

    def get_batch(source, i):
        seq_len = min(args_bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    def train():
        nonlocal hidden
        model.train()
        iterator = ProfileIterator(enumerate(range(0, train_data.size(0) - 1, args_bptt)), duration_sec)
        for batch, i in iterator:
            data, targets = get_batch(train_data, i)
            model.zero_grad()
            output, _ = model(data, hidden)
            loss = criterion(output, targets)
            loss.backward()
        return model, iterator

    def inference():
        with torch.no_grad():
            nonlocal hidden
            model.train(False)
            iterator = ProfileIterator(enumerate(range(0, train_data.size(0) - 1, args_bptt)), duration_sec)
            for batch, i in iterator:
                data, targets = get_batch(train_data, i)
                model(data, hidden)
            return model, iterator

    # At any point you can hit Ctrl + C to break out of training early.
    if is_train:
        return train()
    else:
        return inference()
