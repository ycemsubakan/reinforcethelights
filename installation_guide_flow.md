## Building the container (if you want to keep eveyrything inside the container) 

* *Note:*  The approach below installs the conda environment within the container. This would work if you have sudo permission in the cluster, as the flow environment try to write on the installation path. 

* This will be done on a place where we have sudo. 
* We first pull the container with `singularity pull docker://pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime`. (I picked this from https://docs.mila.quebec/singularity/index.html - It comes with Ubuntu 16.04 and cuda stuff installed ) 
* We then build a sandbox container `sudo singularity build --sandbox pytorch pytorch-1.0.1-cuda10.0-cudnn7-runtime.simg`
* You then log inside the container `singularity shell --writable -H pytorch` . 

## updating linux stuff
* apt update && apt full-upgrade
* apt install wget


## Installing conda
* `wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh`
* I installed anaconda on `/usr/local/home` inside the container. 

## Installing flow
* Follow the steps here `https://flow.readthedocs.io/en/latest/flow_setup.html`
* When installing sumo we need to remove sudo commands from `scripts/setup_sumo_ubuntu1604.sh`. - the problem with the script above is that, it will try to install the files in your home, which is usually set as the ~/root. We don't want this since you cannot access ~/root if you are working on the cluster. Therefore, either modify the installation script so that the files will be installed on /usr/local/home, or you can basically move the files manually. 
* you also need to add the path for the sumo files to your path. What I do is, I keep a .bashrc file on /usr/local/home, and source that. That .bashrc file should contain the correct paths for the sumo _ binaries.  
* see if this runs: `python examples/sumo/sugiyama.py`.  -- I got an error regarding not being able to load opencv. It seems this is an issue with containers. I did the following:
```
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python`
```
And it solved my problem. I found the solution here: `https://github.com/NVIDIA/nvidia-docker/issues/864`

## install ray 

* pip install -U ray

## Install stable baselines
* https://stable-baselines.readthedocs.io/en/master/guide/install.html, (do 'sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev')


* /usr/local/home/anaconda3/envs/flow/bin/python examples/sumo/sugiyama.py 

## Hybrid installation 

* Here, we install conda outside the container. We cannot use the conda of the modules available on the Mila cluster because inside the cluster, we cannot see the /ai/ path, which in which the conda of the module is loaded. 
* I therefore installed a conda on my home path, and I then source the anaconda at my home path within the container. I also added the path of the sumo binaries (that is `/usr/local/home` in the bashrc, so when I source it, the path of of the sumo path is added to `$PATH`)

