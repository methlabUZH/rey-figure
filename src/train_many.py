"""
this script allows you to run multiple experiments sequentially
it reads all *.yaml files from the directory config/ and trains the model with each
it ignores config_example.yaml
the run name of the individual runs will be the concatenation of the
runname specified as command line argument and the name of the config file
"""

import os
import argparse
import time


# command line arguments:
# --local: 1 (run locally) or 0 (run on spaceml)
# --GPU: [0-7] specifies the gpu to run it on, check with gpustat which one is available
# --runname: optional run name argument

DATA_DIR = ''
GPU = -1
RUN_NAME = ''

arg_parser = argparse.ArgumentParser(description="Read in configuration")

arg_parser.add_argument("-L", "--local", help="local flag", required=True)
arg_parser.add_argument("--GPU", help="GPU number on spaceml")
arg_parser.add_argument("-R", "--runname", help="optional runname argument")
args = arg_parser.parse_args()
# args.local
# args.GPU
# args.runname




config_dir = "../config"
output_directory = "../output"

# directory to store terminal output to
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for file in os.listdir(config_dir):
    if file.endswith(".yaml"):
        if file == 'config_example.yaml': continue

        config_path = os.path.join(config_dir, file)
        run_name = ''
        if args.runname:
            run_name = args.runname + "--" + file.replace(".yaml", "")
        else:
            run_name = file.replace(".yaml", "")
        output_file = os.path.join(output_directory, run_name+".txt")

        print("\n\n\nTraining with config file " + config_path + " on GPU " + args.GPU + "\n\n\n")
        start_training = time.perf_counter()

        # run command
        os.system('~/miniconda2/envs/dslab_spaceml/bin/python -u train.py --config ' + config_path + ' --local ' + args.local + ' --GPU ' + args.GPU + ' --runname ' + run_name + ' > ' + output_file)

        end_training = time.perf_counter()
        print("\nTraining took {}min".format((end_training-start_training)/60))

