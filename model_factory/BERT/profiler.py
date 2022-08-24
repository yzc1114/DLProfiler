import itertools
import pathlib

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from common import Config
from model_factory.BERT.models import load_bert_base
from profiler_utils import Profileable, ProfileIterator

tokenizer_path = str(pathlib.Path(__file__).parent.parent / "repos" / "bert-base-uncased")


def bert_rand_input_decorator(wrapped):
    tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer_path)

    def rand_input_impl(batch_size, tokenizer=None):
        return wrapped(batch_size, tokenizer=tokenizer_obj if tokenizer is None else tokenizer)

    return rand_input_impl


@bert_rand_input_decorator
def bert_rand_input(batch_size: int, tokenizer=None):
    fix_sentence_input = "Some fixed sentence input for relative stable execution of each iteration."
    fix_sentence_input *= 5
    inputs = tokenizer([fix_sentence_input] * batch_size, return_tensors="pt")
    inputs.to(Config().local_rank)
    return inputs


def bert_rand_output(batch_size: int):
    return torch.tensor([1] * batch_size, device=Config().local_rank)


def bert_train_template(model: torch.nn.Module, batch_size: int, duration_sec: int, rand_input, rand_output):
    model = model.to(Config().local_rank)
    model = DDP(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    iterator = ProfileIterator(itertools.count(0), duration_sec)
    for _ in iterator:
        optimizer.zero_grad()
        (X, y) = (rand_input(batch_size), rand_output(batch_size))
        pred = model(**X, labels=y)
        loss = pred.loss
        loss.backward()
        optimizer.step()
    return iterator


def bert_inference_template(model: torch.nn.Module, batch_size: int, duration_sec: int, rand_input):
    with torch.no_grad():
        model.to(Config().local_rank)
        model.train(False)
        iterator = ProfileIterator(itertools.count(0), duration_sec)
        for _ in iterator:
            X = rand_input(batch_size)
            model(**X)
        return iterator


class BERTTrain(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        bert_model = load_bert_base()
        return bert_train_template(bert_model, batch_size, duration_sec, bert_rand_input, bert_rand_output)


class BERTInference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        bert_model = load_bert_base()
        return bert_inference_template(bert_model, batch_size, duration_sec, bert_rand_input)


def do_test():
    from common import process_group
    process_group.setup(0, 1)
    profile = BERTTrain()
    iterator = profile.profile(batch_size=64, duration_sec=10)
    print(iterator.to_dict())
    torch.cuda.empty_cache()
    profile = BERTInference()
    iterator = profile.profile(batch_size=64, duration_sec=10)
    print(iterator.to_dict())


if __name__ == '__main__':
    do_test()
