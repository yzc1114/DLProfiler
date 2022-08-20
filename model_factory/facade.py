from model_factory.BERT.profiler import BERTInference, BERTTrain
from model_factory.LSTM.profiler import LSTMTrain, LSTMInference
from model_factory.Mobile.profiler import MobileNetV2Inference, MobileNetV2Train
from model_factory.ResNet.profiler import ResNet50Inference, ResNet50Train, ResNet18Inference, ResNet18Train
from model_factory.VGG.profiler import VGG16Inference, VGG16Train
from model_factory.Yolo.profiler import YoloV5sInference, YoloV5sTrain
from profiler_objects import ModelDescriptions

DDP_train_profilers = dict(
    {
        ModelDescriptions.YOLO_V5S.value.name: YoloV5sTrain,
        ModelDescriptions.VGG_16.value.name: VGG16Train,
        ModelDescriptions.RESNET_18.value.name: ResNet18Train,
        ModelDescriptions.RESNET_50.value.name: ResNet50Train,
        ModelDescriptions.LSTM.value.name: LSTMTrain,
        ModelDescriptions.BERT_BASE.value.name: BERTTrain,
        ModelDescriptions.MOBILE_NET.value.name: MobileNetV2Train,
    }
)

inference_profilers = dict(
    {
        ModelDescriptions.YOLO_V5S.value.name: YoloV5sInference,
        ModelDescriptions.VGG_16.value.name: VGG16Inference,
        ModelDescriptions.RESNET_18.value.name: ResNet18Inference,
        ModelDescriptions.RESNET_50.value.name: ResNet50Inference,
        ModelDescriptions.LSTM.value.name: LSTMInference,
        ModelDescriptions.BERT_BASE.value.name: BERTInference,
        ModelDescriptions.MOBILE_NET.value.name: MobileNetV2Inference,
    }
)


def generate_model_profiler(model_name: str, is_train: bool, is_DDP: bool = True):
    assert is_DDP, "Model Parallel is currently not supported"
    if is_train and is_DDP:
        profiler_cls = DDP_train_profilers[model_name]
    else:
        profiler_cls = inference_profilers[model_name]
    profiler_obj = profiler_cls()
    return profiler_obj


def do_test():
    for md in ModelDescriptions:
        generate_model_profiler(md, True)
        generate_model_profiler(md, False)


if __name__ == '__main__':
    do_test()
