from collections import namedtuple
from enum import Enum

ModelDescriptionNT = namedtuple(typename="ModelDescription", field_names=["name"])


class ModelDescriptions(Enum):
    # model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrined=False)
    YOLO_V5S = ModelDescriptionNT(name="YoloV5S")

    # model = torch.hub.load("pytorch/vision:v0.10.0", "vgg16", pretrained=False)
    VGG_16 = ModelDescriptionNT(name="VGG16")

    Inception_V3 = ModelDescriptionNT(name="InceptionV3")

    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
    RESNET_18 = ModelDescriptionNT(name="ResNet18")

    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=False)
    RESNET_50 = ModelDescriptionNT(name="ResNet50")

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    MOBILE_NET = ModelDescriptionNT(name="MobileNet")

    BERT_BASE = ModelDescriptionNT(name="BertBase")

    LSTM = ModelDescriptionNT(name="LSTM")


def do_test():
    pass


if __name__ == '__main__':
    # print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))
    # main()
    do_test()
