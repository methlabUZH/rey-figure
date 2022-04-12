from constants import *
from src.models.rey_regressor import reyregressor
from src.models.rey_regressor_v2 import reyregressor_v2
from src.models.rey_classifier import rey_classifier_3, rey_classifier_4
from src.models.rey_multilabel_classifier import rey_multiclassifier


def get_classifier(arch, num_classes: int = 2):
    if arch == REYCLASSIFIER_3:
        return rey_classifier_3(num_classes=num_classes)

    if arch == REYCLASSIFIER_4:
        return rey_classifier_4(num_classes=num_classes)

    if arch == REYMULTICLASSIFIER:
        return rey_multiclassifier(num_classes=num_classes)

    raise ValueError(f'unknown arch {arch}')


def get_regressor():
    return reyregressor()


def get_regressor_v2():
    return reyregressor_v2()
