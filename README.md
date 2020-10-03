# Data Science Lab Pyschology Project

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