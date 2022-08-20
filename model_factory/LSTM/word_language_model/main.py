import argparse

import torch
import torch.nn as nn
import torch.onnx
from torch.nn.parallel import DistributedDataParallel as DDP

from model_factory.LSTM.word_language_model import data
from profiler_objects import ProfileIterator


def do_profile(batch_size=32, duration_sec=10, device="0", is_train=True):
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--data', type=str, default='./word_language_model/data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=batch_size, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--dry-run', action='store_true',
                        help='verify the code and the model')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    corpus = data.Corpus(args.data)

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    train_data = batchify(corpus.train, args.batch_size)

    ntokens = len(corpus.dictionary)
    from model_factory.LSTM.word_language_model.model import RNNModel
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
    hidden = model.init_hidden(args.batch_size)

    def gen_hidden(hidden_):
        ls = []
        for e in hidden_:
            ls.append(torch.rand_like(e, device=device))
        return tuple(ls)

    hidden = gen_hidden(hidden)
    model = DDP(model)
    model = model.to(device)

    criterion = nn.NLLLoss()

    def get_batch(source, i):
        seq_len = min(args.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    def train():
        nonlocal hidden
        model.train()
        iterator = ProfileIterator(enumerate(range(0, train_data.size(0) - 1, args.bptt)), duration_sec)
        for batch, i in iterator:
            data, targets = get_batch(train_data, i)
            model.zero_grad()
            output, _ = model(data, hidden)
            loss = criterion(output, targets)
            loss.backward()
        return iterator

    def inference():
        with torch.no_grad():
            nonlocal hidden
            model.train(False)
            iterator = ProfileIterator(enumerate(range(0, train_data.size(0) - 1, args.bptt)), duration_sec)
            for batch, i in iterator:
                data, targets = get_batch(train_data, i)
                model(data, hidden)
            return iterator

    # At any point you can hit Ctrl + C to break out of training early.
    if is_train:
        return train()
    else:
        return inference()
