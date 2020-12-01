# Automated Scoring of the ROCF test
This repo contains code for the automated scoring of the Rey-Osterrieth 
Complex Figure (ROCF) test. In this test, examinees are asked to reproduce 
a complicated line drawing, which is then used to make a neuropsychological 
assessment.


## Methodology
### Regression
Performance, architecture, advantages and disadvantes, etc.


### Classification
Performance, architecture, advantages and disadvantes, etc.

### Uncertainty Quantification
TODO

## How to use this code
Under construction, refactoring.



# BELOW IS OUTDATED AND MIGHT NOT WORK AS DESCRIBED
## Data Science Lab Pyschology Project

### Run on spaceml1
Login to spaceml from eth network: `ssh user@spaceml1`

#### On first login
The first time you need to set some environment variables and install conda environment to work with. Later this doesn't have to be done anymore.

Set environment variables to access internet:  
`echo -e "\nexport http_proxy='http://proxy.ethz.ch:3128'" >> .bash_profile`  
`echo -e "export https_proxy='https://proxy.ethz.ch:3128'" >> .bash_profile`  
Then logout and login again to make these changes work

Install conda to your local folder:  
`wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh`  
`bash Miniconda2-latest-Linux-x86_64.sh`  
This installs conda into your local directory
Then logout and login again to make these changes work

Set up the environment  
`conda env create -f /mnt/ds3lab-scratch/stmuelle/environment_spaceml.yml`

In your home directory, clone the git repository  
`git clone https://github.com/jheitz/dslab.git`

#### Every time
Pull the newest version from git  
`cd dslab/src`  
`git pull`

Activate environment
`source activate dslab_spaceml`

Check which GPU to use 
`gpustat`

Run model  
`python train.py --config filepath --local 0 --GPU GPU --runname RUNNAME` 
where you replace `GPU` by the index of the GPU and `RUNNAME` by a name you give to this run and filepath by the path to your config.yaml file or
`nohup python train.py --config filepath --local 0 --GPU GPU --runname RUNNAME &`  
to run it in the background (doesn't stop when ssh connection is broken)

You can also sequentially train multiple times with different config files using `train_many.py`. 
To do this, create multiple `.yaml` files in the directory `config` and specify the run parameters. 
Then run `nohup python train_many.py --local 0 --GPU GPU --runname RUNNAME &`.
The run name of the individual runs will be the concatenation of `RUNNAME` and the name of the config file.

## Minimal working example to run locally 
1. Download data from [here](https://polybox.ethz.ch/index.php/apps/files/?dir=/Shared/rey_figure_data/Data/ReyFigures/uploadFinal&fileid=2076554047). 
If the link does not work, go to polybox -> rey_figure_data -> Data -> ReyFigures and download the uploadFinal folder. 
2. Place the downloaded files in the rey-figure project cloned from git. Specifically, in the 
`new_data/Data_train folder. 
3. Download the corresponding Data07112018.csv file which contains the labels for the data from  [here](https://polybox.ethz.ch/index.php/apps/files/?dir=/Shared/rey_figure_data/Data/UserRatingData&fileid=2076554026).
Again if the link does not work, please go to rey_figure_data -> Data -> UserRatingData and download the file from there. 
4. Place the .csv file in the new_data/` folder. 
5. Go to the route of the rey-figure project and type either `conda env create -f environment.yml` or `conda env create -f environment_from_history.yml`.
This will install al the required packages you need to train a model. The latter one should be usable across platforms. 
6. Go to the `src/` folder and execute `python train.py --config filepath --local 0 --GPU GPU --runname RUNNAME`