import torch

from common import Config
from model_factory.LSTM.word_language_model import main as LSTMMain
from model_factory.hub_common_profile import common_checkpoint_template
from profiler_utils import Profileable, ProfileIterator


class LSTMTrain(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        _, iterator = LSTMMain.do_profile(batch_size, duration_sec, device=Config().device, is_train=True)
        return iterator


class LSTMInference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        _, iterator = LSTMMain.do_profile(batch_size, duration_sec, device=Config().device, is_train=False)
        return iterator


class LSTMCheckpoint(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model, _ = LSTMMain.do_profile(batch_size, 5, device=Config().device, is_train=True)
        return common_checkpoint_template(model, duration_sec)

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
