#!/bin/sh
env="tjc_gym:TrafficJunctionContinuous6-v0"
algo="ddpg"
experiment="debug"
episode_length=500
actor_train_interval_step=1
tau=0.001
num_env_steps=1000000
batch_size=128
lr=0.001
buffer_size=1000000
seed=1

echo "Env is ${env} and algo is ${algo}"

python ./train/train_tjc.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${experiment} --seed ${seed} --episode_length ${episode_length} --actor_train_interval_step ${actor_train_interval_step} --tau ${tau} --lr ${lr} --num_env_steps ${num_env_steps} --batch_size ${batch_size} --buffer_size ${buffer_size} 