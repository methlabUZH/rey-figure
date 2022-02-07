from constants import *
from src.models.reyclassifier import rey_classifier_3, rey_classifier_4
from src.models.resnet_classifier import wide_resnet50_2


def get_classifier(arch, num_classes: int = 2):
    if arch == WIDE_RESNET50_2:
        return wide_resnet50_2(num_outputs=num_classes)

    if arch == REYCLASSIFIER_3:
        return rey_classifier_3(num_classes=num_classes)

    if arch == REYCLASSIFIER_4:
        return rey_classifier_4(num_classes=num_classes)

    raise ValueError(f'unknown arch {arch}')
