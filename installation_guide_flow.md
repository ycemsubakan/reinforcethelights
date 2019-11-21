## Building the container 

* This will be done on a place where we have sudo. 
* We first pull the container with `singularity pull docker://pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime`. (I picked this from https://docs.mila.quebec/singularity/index.html - It comes with Ubuntu 16.04 and cuda stuff installed ) 
* We then build a sandbox container `sudo singularity build --sandbox pytorch pytorch-1.0.1-cuda10.0-cudnn7-runtime.simg`
* You then log inside the container `singularity shell --writable -H pytorch` . 

## Installing conda
* I installed anaconda on `/root` inside the container. 
* Installing 

## Installing flow
* Follow the steps here `https://flow.readthedocs.io/en/latest/flow_setup.html`
* see if this runs: `python examples/sumo/sugiyama.py`.  -- I got an error regarding not being able to load opencv. It seems this is an issue with containers. I did the following:
```
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python`
```
And it solved my problem. I found the solution here: `https://github.com/NVIDIA/nvidia-docker/issues/864`

## Install stable baselines
* https://stable-baselines.readthedocs.io/en/master/guide/install.html



