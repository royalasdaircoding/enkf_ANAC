This project uses a Conda environment defined in environment.yml. Follow the steps below to install and activate it.

## Requirements:

Miniconda
 or Anaconda
 installed

Python (version specified in environment.yml)

## Installation Steps

1. Create the Conda Environment
conda env create -f environment.yml

If you'd like to use a different environment name:

conda env create -f environment.yml -n your-env-name

2. Activate the Environment
conda activate your-env-name

3. You will also require the local copy of the ```filterpy``` code.


## Basic useage
A basic example is provided in the ```EnkF_ANAC_github.ipynb``` notebook. 
