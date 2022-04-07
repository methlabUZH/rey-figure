"""
Training and evaluation settings 
"""

from constants import REYMULTICLASSIFIER, REYREGRESSOR

config = dict()

"""
dirs with specific multilabel classification models 
"""
config[REYMULTICLASSIFIER] = {
    'non-aug': {
        '78 100': '../spaceml-results/data_78x100-seed_1/final/rey-multilabel-classifier',
        '116 150': '../spaceml-results/data_116x150-seed_1/final/rey-multilabel-classifier',
        '232 300': '../spaceml-results/data_232x300-seed_1/final/rey-multilabel-classifier',
        '348 450': '../spaceml-results/data_348x450-seed_1/final/rey-multilabel-classifier'
    },
    'aug': {
        '78 100': '../spaceml-results/data_78x100-seed_1/final-aug/rey-multilabel-classifier',
        '116 150': '../spaceml-results/data_116x150-seed_1/final-aug/rey-multilabel-classifier',
        '232 300': '../spaceml-results/data_232x300-seed_1/final-aug/rey-multilabel-classifier',
        '348 450': '../spaceml-results/data_348x450-seed_1/final-aug/rey-multilabel-classifier'},
    'angles': [-2, -1, 0, 1, 2]
}

"""
dirs with specific regression models 
"""
config[REYREGRESSOR] = {
    'aug': {
        '78 100': '',
        '116 150': '',
        '232 300': '',
        '348 450': ''
    },
    'non-aug': {
        '78 100': '',
        '116 150': '',
        '232 300': '',
        '348 450': ''
    },
    'angles': [-2, -1, 0, 1, 2]
}
