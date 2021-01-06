# Correlated-Q Learning

Learn equilibrium strategies for the two-player game RoboCub with the following reinforcement learning algorithms:

- Q-learning
- Foe-Q learning
- Friend-Q learning
- Correlated-Q learning

## Setting up the environment

To create the environment

    conda env create -f environment.yml

To activate the environment

    conda activate ceq

## Running the code

Set Q-learning parameters in the YAML files found in the `configs` directory.

To generate Q-values for all learners

    ./run.sh
