# AeroRL-Aerodynamic-Reinforcement-Learning
![Screen_Shot_2024-06-11_at_10 07 20-removebg-preview](https://github.com/ScriptServants/AeroRL-Aerodynamic-Reinforcement-Learning-/assets/172327968/08ac8c97-ee26-4f9d-af53-0fb62ee9623f)

Our primary focus is on creating a robust drone model capable of navigating through challenging conditions such as fog, rain, and other environmental variables, all while ensuring collision avoidance.

## AirSim Drone Simulation Setup and Training RL Agent

### Prerequisites

### Steps
1. **Install Conda**
   - wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O getCondaInstaller.sh
   - bash getCondaInstaller.sh
   
2. **Run Setup Script**
   - Run the setup.sh script to set up the environment. This script will include the steps for setting up Conda and installing required     
   packages.

3. **Activate Conda Environment**
   - conda activate airsim

4. **Run the AirSim Environment and Python Code**
   
   - In one terminal run run runEnv.sh to start the AirSim environment.
         - ./runEnv.sh
   - In another terminal unPython.sh to run the Python code for the RL agent
         - ./runPython.sh
