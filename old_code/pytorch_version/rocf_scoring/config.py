import os
import argparse
import yaml

# command line arguments:
# --local: 1 (run locally) or 0 (run on spaceml)
# --GPU: [0-7] specifies the gpu to run it on, check with gpustat which one is available
# --config: path to the config.yaml file
# --runname: optional run name argument

DATA_DIR = ''
GPU = -1
RUN_NAME = ''

arg_parser = argparse.ArgumentParser(description="Read in configuration")

arg_parser.add_argument("-C", "--config", help="config file", required=True)
arg_parser.add_argument("-L", "--local", help="local flag", required=True)
arg_parser.add_argument("--GPU", help="GPU number on spaceml")
arg_parser.add_argument("-R", "--runname", help="optional runname argument")


args = arg_parser.parse_args()

print("os.getcwd(): ", os.getcwd())

# Path to config
base_path = "/Users/felixasanger/Desktop/pytorch_rey-figure/rocf_scoring/config"
config_path = os.path.join(base_path, args.config)

f = open(config_path, 'r')
# RUN_PARAMETERS: dictionary containing flags, can later be used to add further run information
RUN_PARAMETERS = yaml.load(f)

if args.local == "1":
    LOCAL = True
else:
    LOCAL = False

if not LOCAL:
    if args.GPU is None:
        print("Must provide GPU number. Exiting...")
        exit(1)
    else:
        try:
            GPU = int(args.GPU) # id of gpu to use
            if GPU < 0 or GPU > 7:
                raise ValueError('The GPU must be between 0 and 7')
        except ValueError:
            print('The GPU provided is invalid, exiting...')
            exit(1)

if args.runname is not None:
    RUN_NAME = args.runname
    print("Using RUN_NAME " + RUN_NAME)


# use new preprocessing (crowdsourcing)
try:
    NEW_DATA = RUN_PARAMETERS['NEW_DATA']
except KeyError:
    print("NEW_DATA not provided, by default set to True...")
    NEW_DATA = True
assert NEW_DATA in [True, False], "Invalid NEW_DATA"

# Mode to run code in: True for Regressor, False for Classifier
try:
    REGRESSOR_MODE = RUN_PARAMETERS['REGRESSOR_MODE']
except KeyError:
    print("REGRESSOR_MODE not provided, by default set to True...")
    REGRESSOR_MODE = True
assert REGRESSOR_MODE in [True, False], "Invalid REGRESSOR_MODE"


if LOCAL:
    if NEW_DATA:
        DATA_DIR = "../new_data/"
        print("Running file locally on new preprocessing...")
    else:
        DATA_DIR = "../preprocessing/"
        print("Running file locally on old preprocessing...")

    ROOT_DIR = "/Users/felixasanger/Desktop/pytorch_rey-figure"
    TRAINING_RESULTS_DIR = os.path.join(ROOT_DIR, "rocf_scoring", "training_results")
    TRAINED_MODELS = os.path.join(ROOT_DIR, "rocf_scoring", "trained_models")
else:
    if NEW_DATA:
        DATA_DIR = "/mnt/ds3lab-scratch/stmuelle/data_new/"
        print("Running on new preprocessing on spaceml on GPU " + str(GPU))
    else:
        DATA_DIR = "/mnt/ds3lab-scratch/stmuelle/preprocessing/"
        print("Running on old preprocessing on spaceml on GPU "+str(GPU))

    # set up gpu information for spaceml (see slack)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)  # select ID of GPU that shall be used

# do preprocessing augmentation
try:
    DATA_AUGMENTATION = RUN_PARAMETERS['DATA_AUGMENTATION']
except KeyError:
    print("DATA_AUGMENTATION not provided, by default set to False...")
    DATA_AUGMENTATION = False
assert DATA_AUGMENTATION in [True, False], "Invalid DATA_AUGMENTATION"

# format of label
# 'one' for one score per image
# 'one-per-item' for an 19-dimensional label (one score per item + complete score)
# 'three-per-item' for an 55-dimensional label (three scores per item + complete score)
try:
    LABEL_FORMAT = RUN_PARAMETERS['LABEL_FORMAT']
except KeyError:
    print("LABEL_FORMAT not provided, by default set to 'one'...")
    LABEL_FORMAT = 'one'
assert LABEL_FORMAT in ['one', 'one-per-item', 'three-per-item'], "Invalid LABEL_FORMAT"


try:
    BINNING = RUN_PARAMETERS['BINNING']
except KeyError:
    print("BINNING not provided, by default set to 'none'...")
    BINNING = 'none'

try:
    VAL_BINNING = RUN_PARAMETERS['VAL_BINNING']
except KeyError:
    print("VAL_BINNING not provided, by default set to False (validation same as training)...")
    VAL_BINNING = False
assert VAL_BINNING in [True, False], "Invalid VAL_BINNING"


try:
    REDO_PREPROCESSING_LABELS = RUN_PARAMETERS['REDO_PREPROCESSING_LABELS']
except KeyError:
    print("REDO_PREPROCESSING_LABELS not provided, by default set to False...")
    REDO_PREPROCESSING_LABELS = False
assert REDO_PREPROCESSING_LABELS in [True, False], "Invalid REDO_PREPROCESSING_LABELS"

try:
    REDO_PREPROCESSING_IMAGES = RUN_PARAMETERS['REDO_PREPROCESSING_IMAGES']
except KeyError:
    print("REDO_PREPROCESSING_IMAGES not provided, by default set to False...")
    REDO_PREPROCESSING_IMAGES = False
assert REDO_PREPROCESSING_IMAGES in [True, False], "Invalid REDO_PREPROCESSING_IMAGES"


try:
    REDO_PREPROCESSING = RUN_PARAMETERS['REDO_PREPROCESSING']
except KeyError:
    print("REDO_PREPROCESSING not provided, by default set to False...")
    REDO_PREPROCESSING = False
assert REDO_PREPROCESSING in [True, False], "Invalid REDO_PREPROCESSING"




try:
    LOAD_ONLY_FEW = RUN_PARAMETERS['LOAD_ONLY_FEW']
except KeyError:
    print("LOAD_ONLY_FEW not provided, by default set to False...")
    LOAD_ONLY_FEW = False
assert LOAD_ONLY_FEW in [True, False], "Invalid LOAD_ONLY_FEW"

try:
    NUMBER_LOAD_ONLY_FEW = RUN_PARAMETERS['NUMBER_LOAD_ONLY_FEW']
except KeyError:
    print("NUMBER_LOAD_ONLY_FEW not provided, by default set to 50 ...")
    NUMBER_LOAD_ONLY_FEW = 50


# define folds you want to skip, if run before failed some time
try:
    SKIP_FOLDS = RUN_PARAMETERS['SKIP_FOLDS']
except KeyError:
    print("SKIP_FOLDS not provided, by default set to the empty list (no skipping)...")
    SKIP_FOLDS = []
assert isinstance(SKIP_FOLDS, (list,)), "Invalid SKIP_FOLDS"


# shows debugging prints
try:
    DEBUG = RUN_PARAMETERS['DEBUG']
except KeyError:
    print("DEBUG not provided, by default set to True...")
    DEBUG = True
assert DEBUG in [True, False], "Invalid DEBUG"


# classification encoding: 'one-hot' (0,0,1,0,0), 'weighted' (0.05,0.1,0.7,0.1,0.05), 'ordinal' (1, 1, 1, 0, 0)
try:
    CLASSIFICATION_ENCODER = RUN_PARAMETERS['CLASSIFICATION_ENCODER']
except KeyError:
    print("CLASSIFICATION_ENCODER not provided, by default set to 'one-hot'...")
    CLASSIFICATION_ENCODER = 'one-hot'
assert CLASSIFICATION_ENCODER in ['one-hot','weighted','ordinal'], "Invalid CLASSIFICATION_ENCODER"



# specify number of convolutional layers (currently 2, 3 or 4)
try:
    CONV_LAYERS = RUN_PARAMETERS['CONV_LAYERS']
except KeyError:
    print("CONV_LAYERS not provided, by default set to 3...")
    CONV_LAYERS = 3
assert CONV_LAYERS in [2,3,4], "Invalid CONV_LAYERS"


try:
    DROPOUT = RUN_PARAMETERS['DROPOUT']
except KeyError:
    print("DROPOUT not provided, by default set to 0...")
    DROPOUT = 0
assert 0 <= float(DROPOUT) <= 1, "Invalid DROPOUT"

try:
    MODEL_PATH = RUN_PARAMETERS['MODEL_PATH']
except KeyError:
    print("MODEL_PATH not provided, by default set to None...")
    MODEL_PATH = None

try:
    INTERMEDIATES = RUN_PARAMETERS['INTERMEDIATES']
except KeyError:
    print("INTERMEDIATES not provided, by default set to False...")
    INTERMEDIATES = False

try:
    CONVERGENCE = RUN_PARAMETERS['CONVERGENCE']
except KeyError:
    print("CONVERGENCE not provided, by default set to None...")
    CONVERGENCE = 0

try:
    DIRECTORY_TO_WATCH = RUN_PARAMETERS['DIRECTORY_TO_WATCH']
except KeyError:
    print("DIRECTORY_TO_WATCH not provided, by default set to None...")
    DIRECTORY_TO_WATCH = None

try:
    DEST_PATH = RUN_PARAMETERS['DEST_PATH']
except KeyError:
    print("DEST_PATH not provided, by default set to None...")
    DEST_PATH = None

# TEST can be False (cross-validation on training preprocessing) or True (train on all training, test on test preprocessing)
try:
    TEST = RUN_PARAMETERS['TEST']
except KeyError:
    print("TEST not provided, by default set to false (i.e. cross-validation)...")
    TEST = False


try:
    DROPOUT_MC = RUN_PARAMETERS['DROPOUT_MC']
except KeyError:
    print("DROPOUT_C not provided, by default set to false ...")
    DROPOUT_MC = False

try:
    DROPOUT_MC_RATE = RUN_PARAMETERS['DROPOUT_MC_RATE']
except KeyError:
    print("DROPOUT_MC_RATE not provided, by default set to 0 ...")
    DROPOUT_MC_RATE = 0

try:
    RESUME_CHECKPOINT = RUN_PARAMETERS['RESUME_CHECKPOINT']
except KeyError:
    print("RESUME_CHECKPOINT not provided, by default set to False ...")
    RESUME_CHECKPOINT = False

try:
    PATH_CHECKPOINT = RUN_PARAMETERS['PATH_CHECKPOINT']
except KeyError:
    print("PATH_CHECKPOINT not provided, by default set to None ...")
    PATH_CHECKPOINT = None


try:
    EVERY_N_EPOCHS = RUN_PARAMETERS['EVERY_N_EPOCHS']
except KeyError:
    print("EVERY_N_EPOCHS not provided, by default set to 10 ...")
    EVERY_N_EPOCHS = 10

try:
    SAVE_MODEL_CHECKPOINTS = RUN_PARAMETERS['SAVE_MODEL_CHECKPOINTS']
except KeyError:
    print("SAVE_MODEL_CHECKPOINTS not provided, by default set to False ...")
    SAVE_MODEL_CHECKPOINTS = False


print("CONFIG: {}".format(RUN_PARAMETERS))
