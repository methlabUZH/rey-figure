import torchvision
from .model import TDRG

model_dict = {'TDRG': TDRG}


def get_model(num_classes, args):
    res101 = torchvision.models.resnet101(pretrained=True)
    return model_dict[args.model_name](res101, num_classes)
