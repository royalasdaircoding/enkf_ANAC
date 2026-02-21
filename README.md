# EnKF Data Assimilation Software
This respository contains code which is adapted from https://github.com/rlabbe/filterpy. For the most part this code works in the same way, however there are some additions which have been made, in particular  the inclusion of various inflation factors. 

If you are unsure on how Kalman Filters work, then I highly recommend reading the guide https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/ for a hands-on discussion of using data assimilation in Python. `filterpy` has many other data assimilation routines that I have not implemented here. 

This project uses a Conda environment defined in environment.yml. Follow the steps below to install and activate it.

## Pre-requisites
If you are new to Python, you need to know how to set-up environments using either `pip` or `conda`. 

## Requirements:

Miniconda
 or Anaconda
 or mamba
 installed

Python (version specified in ```environment.yml```)

## Installation Steps

1. Create the Conda Environment using Anaconda or mamba 
```conda env create -f environment.yml```

If you'd like to use a different environment name:

```conda env create -f environment.yml -n your-env-name```

2. Activate the Environment
conda activate your-env-name

3. You will also require the local copy of the ```filterpy``` code.


## Basic usage
A basic example is provided in the ```EnkF_ANAC_github.ipynb``` notebook. 
