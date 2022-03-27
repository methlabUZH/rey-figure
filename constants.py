import os as _os

# project structure
ROOT_DIR = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_DIR = _os.path.join(ROOT_DIR, 'results/')
DATA_DIR = _os.path.join(ROOT_DIR, 'data/')
RESOURCES_DIR = _os.path.join(ROOT_DIR, 'resources/')

USER_RATING_DATA_DIR = 'UserRatingData'
MAIN_LABEL_FILENAME = '{split}-labels.csv'

FOTO_FOLDERS = ['Typeform', 'USZ_fotos']

# architectures
REYCLASSIFIER_3 = 'rey-classifier-3'
REYCLASSIFIER_4 = 'rey-classifier-4'
REYMULTICLASSIFIER = 'rey-multilabel-classifier'
WIDE_RESNET50_2 = 'wide-resnet50-2'
REYREGRESSOR = 'rey-regressor'

# data
TEST_FRACTION = 0.2
DEBUG_DATADIR_SMALL = '/Users/maurice/phd/src/rey-figure/data/serialized-data/debug-116x150-pp0'
DEBUG_DATADIR_BIG = '/Users/maurice/phd/src/rey-figure/data/serialized-data/debug-232x300-pp0'
N_ITEMS = 18
DEFAULT_CANVAS_SIZE = (116, 150)
DEFAULT_CANVAS_SIZE_BIG = (232, 300)
AUGM_CANVAS_SIZE = (464, 600)
CLASSIFICATION_LABELS = 'classification_labels'
REGRESSION_LABELS = 'regression_labels'

BIN_LOCATIONS1_V2 = [-1, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
BIN_LOCATIONS2_V2 = [-1, 6, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
BIN_LOCATIONS3_V2 = [-1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]

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
