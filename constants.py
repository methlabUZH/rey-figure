import os as _os

# project structure
ROOT_DIR = _os.path.dirname(_os.path.abspath(__file__))
RESULTS_DIR = _os.path.join(ROOT_DIR, 'results/')
DATA_DIR = _os.environ.get("REY_FIGURE_DATA")
RESOURCES_DIR = _os.path.join(ROOT_DIR, 'resources/')

# architectures
REYCLASSIFIER_3 = 'rey-classifier-3'
REYCLASSIFIER_4 = 'rey-classifier-4'
REYMULTICLASSIFIER = 'rey-multilabel-classifier'
WIDE_RESNET50_2 = 'wide-resnet50-2'
REYREGRESSOR = 'rey-regressor'

# data
USER_RATING_DATA_DIR = 'UserRatingData'
MAIN_LABEL_FILENAME = '{split}-labels.csv'
FOTO_FOLDERS = ['Typeform', 'USZ_fotos']
TEST_FRACTION = 0.2
DATADIR_SMALL = './data/serialized-data/data_116x150-seed_1'
DATADIR_BIG = './data/serialized-data/data_232x300-seed_1'
N_ITEMS = 18
DEFAULT_CANVAS_SIZE = (116, 150)
DEFAULT_CANVAS_SIZE_BIG = (232, 300)
AUGM_CANVAS_SIZE = (464, 600)
CLASSIFICATION_LABELS = 'classification_labels'
REGRESSION_LABELS = 'regression_labels'

# semantic transformations
TF_ROTATION = 'rotation'
TF_PERSPECTIVE = 'perspective'
TF_BRIGHTNESS = 'brightness'
TF_CONTRAST = 'contrast'

# results
SCORE_COLUMNS = [f'score_item_{i + 1}' for i in range(N_ITEMS)]
CLASS_COLUMNS = [f'class_item_{i + 1}' for i in range(N_ITEMS)]
ABSOLUTE_ERROR = 'absolute_error'
SQUARED_ERROR = 'squared_error'
NUM_MISCLASS = 'num_misclassified'
R_SQUARED = 'r_squared'
ERR_LEVEL_TOTAL_SCORE = 'total_score'
ERR_LEVEL_ITEM_SCORE = 'item_score'

# score bins
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
