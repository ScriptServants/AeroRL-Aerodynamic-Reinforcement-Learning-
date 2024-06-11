CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda create -n airsim python=3.10
conda init
conda activate airsim
pip install numpy
pip install msgpack-rpc-python
pip install airsim
pip install torch
pip install pillow
pip install tensorboard

wget https://github.com/microsoft/AirSim/releases/download/v1.8.1/AirSimNH.zip
unzip ./AirSimNH.zip
rm -f AirSimNH.zip
sed '5s/$/ -graphicsadapter=1/' ./AirSimNH/LinuxNoEditor/AirSimNH.sh > file.tmp
chmod 740 file.tmp
mv file.tmp ./AirSimNH/LinuxNoEditor/AirSimNH.sh

git clone https://github.com/batuhan3526/AirSim-PyTorch-Drone-DDQN-Agent.git
