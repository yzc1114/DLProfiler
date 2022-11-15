from model_factory.BERT.profiler import BERTInference, BERTTrain, BERTCheckpoint
from model_factory.LSTM.profiler import LSTMTrain, LSTMInference, LSTMCheckpoint
from model_factory.Mobile.profiler import MobileNetV2Inference, MobileNetV2Train, MobileNetV2Checkpoint
from model_factory.ResNet.profiler import ResNet50Inference, ResNet50Train, ResNet18Inference, ResNet18Train, ResNet18Checkpoint, ResNet50Checkpoint
from model_factory.VGG.profiler import VGG16Inference, VGG16Train, VGG19Train, VGG19Inference, VGG16Checkpoint, VGG19Checkpoint
from model_factory.Yolo.profiler import YoloV5sInference, YoloV5sTrain, YoloV5sCheckpoint
from model_factory.Inception.profiler import InceptionV3Train, InceptionV3Inference, InceptionV3Checkpoint
from model_factory.EfficientNet.profiler import EfficientNetTrain, EfficientNetInference, EfficientNetCheckpoint
from objects import ModelDescriptions

DDP_train_profilers = dict(
    {
        ModelDescriptions.YOLO_V5S.value.name: YoloV5sTrain,
        ModelDescriptions.VGG_16.value.name: VGG16Train,
        ModelDescriptions.Inception_V3.value.name: InceptionV3Train,
        ModelDescriptions.RESNET_18.value.name: ResNet18Train,
        ModelDescriptions.RESNET_50.value.name: ResNet50Train,
        ModelDescriptions.LSTM.value.name: LSTMTrain,
        ModelDescriptions.BERT_BASE.value.name: BERTTrain,
        ModelDescriptions.MOBILE_NET.value.name: MobileNetV2Train,
        ModelDescriptions.EfficientNet.value.name: EfficientNetTrain,
    }
)

checkpoint_profilers = dict(
    {
        ModelDescriptions.YOLO_V5S.value.name: YoloV5sCheckpoint,
        ModelDescriptions.VGG_16.value.name: VGG16Checkpoint,
        ModelDescriptions.Inception_V3.value.name: InceptionV3Checkpoint,
        ModelDescriptions.RESNET_18.value.name: ResNet18Checkpoint,
        ModelDescriptions.RESNET_50.value.name: ResNet50Checkpoint,
        ModelDescriptions.LSTM.value.name: LSTMCheckpoint,
        ModelDescriptions.BERT_BASE.value.name: BERTCheckpoint,
        ModelDescriptions.MOBILE_NET.value.name: MobileNetV2Checkpoint,
        ModelDescriptions.EfficientNet.value.name: EfficientNetCheckpoint,
    }
)

inference_profilers = dict(
    {
        ModelDescriptions.YOLO_V5S.value.name: YoloV5sInference,
        ModelDescriptions.VGG_16.value.name: VGG16Inference,
        ModelDescriptions.Inception_V3.value.name: InceptionV3Inference,
        ModelDescriptions.RESNET_18.value.name: ResNet18Inference,
        ModelDescriptions.RESNET_50.value.name: ResNet50Inference,
        ModelDescriptions.LSTM.value.name: LSTMInference,
        ModelDescriptions.BERT_BASE.value.name: BERTInference,
        ModelDescriptions.MOBILE_NET.value.name: MobileNetV2Inference,
        ModelDescriptions.EfficientNet.value.name: EfficientNetInference,
    }
)


def generate_model_profiler(model_name: str, mode: str, is_DDP: bool = True):
    assert is_DDP, "Model Parallel is currently not supported"
    if mode == "train" and is_DDP:
        profiler_cls = DDP_train_profilers[model_name]
    elif mode == "inference":
        profiler_cls = inference_profilers[model_name]
    elif mode == "checkpoint":
        profiler_cls = checkpoint_profilers[model_name]
    else:
        assert False, f"mode {mode} not supported"
    profiler_obj = profiler_cls()
    return profiler_obj


def do_test():
    for md in ModelDescriptions:
        generate_model_profiler(md, True)
        generate_model_profiler(md, False)


if __name__ == '__main__':
    do_test()
