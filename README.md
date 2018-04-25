# Kuka_Pybullet
Using Pybullet simulation environment to collect data as described in "Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping"

# Prerequisites:
	Pybullet library in Python 3.6 environment(havn't tested it in other environments yet).

#  Build training data:
python build_data.py --X_input_dir "sim_backSubed" --X_output_file "simsubed_22_1006.tfrecords"


# Usage
## Run original Kuka environment:
python -m pybullet_envs.examples.kukaGymEnvTest
## Step 1:
Download procedurally generated random geometric shapes on https://sites.google.com/site/brainrobotdata/home/models.

## Step 2:
Extract the zip file and put the random_urdfs folder right below this repo folder.

## Step 3:
Run the simulation with: python simulation_model.py
	


