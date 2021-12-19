#!/bin/sh

echo "Setting up container.."
sudo apt-get update

# https://serverfault.com/questions/949991/how-to-install-tzdata-on-a-ubuntu-docker-image
sudo DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

sudo apt-get --assume-yes install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get --assume-yes install python3.9-venv

echo "Cloning project repo.."
git clone https://github.com/jakobdybdahl/msc-project.git
echo "Project cloned!"

cd msc-project

echo "Setting up venv.."
python3.9 -m venv venv
source venv/bin/activate

echo "Setting up project"
pip install -e .

echo "Install PyTorch"
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

echo "Installing project requirements"
pip install -r requirements.txt