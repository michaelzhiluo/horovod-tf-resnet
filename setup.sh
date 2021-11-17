# NCCL 2
cd ~/
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
sudo apt install libnccl2=2.6.4-1+cuda10.0 libnccl-dev=2.6.4-1+cuda10.0
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> ~/.bashrc
source ~/.bashrc

# OpenMPI
cd ~/
sudo mv /usr/bin/mpirun /usr/bin/bk_mpirun
sudo mv /usr/bin/mpirun.openmpi /usr/bin/bk_mpirun.openmpi
sudo mv /usr/bin/orted /usr/bin/bk_orted
sudo mv /opt/amazon/openmpi/ /opt/amazon/b_openmpi/
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
tar -xvf openmpi-4.0.1.tar.gz
cd ./openmpi-4.0.1
./configure --prefix=$HOME/openmpi
sudo make -j 8 all
sudo make install
sudo ln -s ~/openmpi/bin/orted /usr/bin/orted
echo 'export LD_LIBRARY_PATH=~/openmpi/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=~/openmpi/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Setup Python Environment
sudo apt --assume-yes install g++-4.8
echo alias python=python3 >> ~/.bashrc
echo alias pip=pip3 >> ~/.bashrc
source ~/.bashrc
pip3 install tensorflow-gpu==1.14.0 keras==2.3.0
HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip3 install --no-cache-dir horovod==0.18.0
