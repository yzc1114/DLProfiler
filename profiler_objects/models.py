from collections import namedtuple
from enum import Enum
from pathlib import Path
# from transformers import AutoModelForSequenceClassification, BertConfig
import torch
import os.path

torch.cuda.mem_get_info()

home = str(Path.home())
cache_home = os.path.join(home, ".cache")
print(f"cache_home = {cache_home}.")
torch_hub_cache_home = os.path.join(cache_home, "torch/hub")
print(f"torch_hub_cache_home = {torch_hub_cache_home}.")


def get_mono_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device


ModelDescriptionNT = namedtuple(typename="ModelDescription", field_names=["name"])


class ModelDescriptions(Enum):
    # model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrined=False)
    YOLO_V5S = ModelDescriptionNT(name="YoloV5S")

    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
    RESNET_18 = ModelDescriptionNT(name="ResNet18")

    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=False)
    RESNET_50 = ModelDescriptionNT(name="ResNet50")

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    MOBILE_NET = ModelDescriptionNT(name="MobileNet")

    BERT_BASE = ModelDescriptionNT(name="BertBase")

    LSTM = ModelDescriptionNT(name="LSTM")


ModelSpecNT = namedtuple("ModelSpec", field_names=[
    "model_description", "batch_size"
])

def generate_model_spec(model_description: ModelDescriptions, batch_size: int):
    return ModelSpecNT(model_description=model_description, batch_size=batch_size)

def do_test():
    # ModelDescriptions.load(ModelDescriptions.YOLO_V5S)
    # ModelDescriptions.load(ModelDescriptions.MOBILE_NET)
    for md in ModelDescriptions:
        ModelDescriptions.load(md)


def main():
    model = ModelDescriptions.load(ModelDescriptions.YOLO_V5S)
    rand_input = torch.rand((1,) + ModelDescriptions.RESNET_50.value.input_shape)
    print(f'input_shape: {rand_input}')
    output = model(rand_input)
    if isinstance(output, torch.Tensor):
        print(f'output_shape: {output.shape}')
    else:
        print(f"output {output}")


if __name__ == '__main__':
    # print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))
    # main()
    do_test()
