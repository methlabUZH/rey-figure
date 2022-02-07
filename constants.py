import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results/')

FOTO_FOLDERS = ['Typeform', 'USZ_fotos']

# architectures
REYCLASSIFIER_3 = 'rey-classifier-3'
REYCLASSIFIER_4 = 'rey-classifier-4'
WIDE_RESNET50_2 = 'wide-resnet50-2'
REYREGRESSOR = 'rey-regressor'


# data
DEBUG_DATADIR = '/Users/maurice/phd/src/rey-figure/data/serialized-data/debug-116x150-pp0'
N_ITEMS = 18
DEFAULT_CANVAS_SIZE = (116, 150)
AUGM_CANVAS_SIZE = (464, 600)
DEFAULT_SEED = 762

BIN_LOCATIONS1 = [
    (0, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    (22, 24),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (32, 34),
    (34, 36),
    (36, 37)
]

BIN_LOCATIONS2 = [
    (0, 6),
    (6, 12),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    (22, 24),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (32, 34),
    (34, 36),
    (36, 37)
]

BIN_LOCATIONS_DENSE = [
    (0, 7),
    (7, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
    (27, 28),
    (28, 29),
    (29, 30),
    (30, 31),
    (31, 32),
    (32, 33),
    (33, 34),
    (34, 35),
    (35, 36),
    (36, 37)
]

# # hyperparam configs
# RESNET18_CONFIG = os.path.join(ROOT_DIR, 'old_code/configs/resnet18-hyperparams.json')
# RESNET50_CONFIG = os.path.join(ROOT_DIR, 'old_code/configs/resnet50-hyperparams.json')
# RESNET101_CONFIG = os.path.join(ROOT_DIR, 'old_code/configs/resnet101-hyperparams.json')
# RESNEXT50_CONFIG = os.path.join(ROOT_DIR, 'old_code/configs/resnext50-32x4d-hyperparams.json')
# EFFICIENTNET_B0 = os.path.join(ROOT_DIR, 'old_code/configs/efficientnet-b0.json')
# EFFICIENTNET_L2 = os.path.join(ROOT_DIR, 'old_code/configs/efficientnet-l2.json')
