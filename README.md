# Correlated-Q Learning

## Setting up the environment

To create the environment

    conda env create -f environment.yml

To activate the environment

    conda activate ceq

## Running the code

To run Q-learning on the RoboCup environment

    python main.py path/to/config.yaml
    
This script `run.sh` generates Q-values for the following learners:

- Q-learner
- Foe-Q learner
- Friend-Q learner
- Correlated-Q learner
