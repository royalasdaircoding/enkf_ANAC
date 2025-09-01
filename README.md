This project uses a Conda environment defined in environment.yml. Follow the steps below to install and activate it.

## Requirements:

Miniconda
 or Anaconda
 installed

Python (version specified in environment.yml)

## Installation Steps
1. Clone the Repository (if needed)
git clone https://github.com/your-username/your-repo.git
cd your-repo

2. Create the Conda Environment
conda env create -f environment.yml

If you'd like to use a different environment name:

conda env create -f environment.yml -n your-env-name

3. Activate the Environment
conda activate your-env-name
