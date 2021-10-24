import os

ROOT_DIR = '/Users/maurice/phd/src/psychology/main'

# Data
RAW_DATA_DIR = '/Users/maurice/phd/src/data_preprocessing/psychology/raw/'
DATA_DIR = os.path.join(ROOT_DIR, 'data_preprocessing/')
FILENAME_LABELS = 'Data07112018.csv'
LABEL_FORMAT = 'one-per-item'

# training
DATA_AUGMENTATION = False
TRAINING_RESULTS_DIR = os.path.join(ROOT_DIR, 'results_data/training/')
RANDOM_SEED = 786

if DATA_AUGMENTATION:
    CANVAS_SIZE = (464, 600)  # height, width
else:
    CANVAS_SIZE = (116, 150)

augmented_CANVAS_SIZE = (116, 150)

BIN_LOCATIONS = [
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
