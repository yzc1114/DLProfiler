import torch

from common import Config
from model_factory.LSTM.word_language_model import main as LSTMMain
from profiler_objects import Profileable, ProfileIterator


class LSTMTrain(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        return LSTMMain.do_profile(batch_size, duration_sec, device=Config().local_rank, is_train=True)


class LSTMInference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        return LSTMMain.do_profile(batch_size, duration_sec, device=Config().local_rank, is_train=False)


def do_test():
    from common import process_group
    process_group.setup(0, 1)
    profile = LSTMTrain()
    iterator = profile.profile(batch_size=50, duration_sec=10)
    print(iterator.to_dict())
    torch.cuda.empty_cache()
    profile = LSTMInference()
    iterator = profile.profile(batch_size=50, duration_sec=10)
    print(iterator.to_dict())


if __name__ == '__main__':
    do_test()
