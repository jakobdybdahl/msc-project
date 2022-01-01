# msc-project

This project is developed by [RasmusThorsen](https://github.com/RasmusThorsen) and [jakobdybdahl](https://github.com/jakobdybdahl) during our master thesis at [Aarhus University](https://www.au.dk/).

# Prerequisites

Environment with python 3.9.x (what we have used and only version tested)

# Installation

Install requirements through PiPy:

```bash
pip install -r requirements.txt
```

'Install' project. Makes it avaiable at python path:

```bash
pip install -e .
```

### PyTorch (only one of the two following commands)

With CUDA enabled (requires Windows and Nvidia graphic card to utilize):

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl cu113/torch_stable.html
```

or without CUDA:

```bash
pip install torch torchvision torchaudio
```

### PyTorch Geometric (only one of the two following commands)

With Cuda enabled (requires Windows and Nvidia graphic card to utilize)

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

or without CUDA:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
```

# Quickly get it running

## Train DDPG agent (learning after 10 minutes / 60 episodes)

Run the following script in project folder

```bash
python ./msc_project/scripts/train/train_tjc.py --algorithm_name ddpg --render True --num_eval_episodes 1
```

## Load pretrained DDPG model (with success rate of 98%)

Run the following script in project folder

```bash
python ./msc_project/scripts/train/train_tjc.py --algorithm_name ddpg --render True --model_dir "./msc_project/models/ddpg/seed_9_ep_1260"
```

Other models can be found under the folder `models`.

# Reproduce experiment results

To reproduce any of the results in the report run the scripts:

| Agent    | Script                                               |
| -------- | ---------------------------------------------------- |
| Baseline | `python ./msc_project/baselines/traffic-light-agent.py` |
| DDPG     | `source msc_project/scripts/train_tjc_ddpg.sh`       |
| MADDPG   | `source msc_project/scripts/train_tjc_maddpg.sh`     |
| DDPG+GAT | `source msc_project/scripts/train_tjc_gat.sh`        |

The script for DDPG will only run five experiments in parralel. To run more, open the script and adjust the seeds. The same apply for MADDPG and DDPG+GAT.

Notice the scripts will be started in the background.
