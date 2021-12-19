#!/bin/sh
algo="maddpg"
experiment="pure_maddpg"
env="tjc_gym:TrafficJunctionContinuous4-v1"
# render=0
buffer_size=1000000
hidden_size1=400
hidden_size2=300
lr_actor=0.0001
lr_critic=0.001
weight_decay=0
tau=0.001
batch_size=128
gamma=0.99
num_random_episodes=1
act_noise_std_start=0.3
act_noise_std_min=0.0
act_noise_decay_end_step=200000
max_episode_length=500
actor_train_interval_step=1
train_interval=2
save_interval=100
running_avg_size=50
step_cost_factor=-0.01
collision_cost=-100
arrive_prob=0.05
fov_radius=3

echo "Env is ${env} and algo is ${algo}"

seeds=(1 2 3 4 5)
for seed in "${seeds[@]}"
do
  echo "Running experiment with seed = ${seed}"

  python3 ./train/train_tjc.py \
  --algorithm_name ${algo} \
  --experiment_name ${experiment} \
  --env_name ${env} \
  --seed ${seed} \
  ${render:+--render "$render"} \
  --buffer_size ${buffer_size} \
  --hidden_size1 ${hidden_size1} \
  --hidden_size2 ${hidden_size2} \
  --lr_actor ${lr_actor} \
  --lr_critic ${lr_critic} \
  --weight_decay ${weight_decay} \
  --tau ${tau} \
  --batch_size ${batch_size} \
  --gamma ${gamma} \
  --num_random_episodes ${num_random_episodes} \
  --act_noise_std_start ${act_noise_std_start} \
  --act_noise_std_min ${act_noise_std_min} \
  --act_noise_decay_end_step ${act_noise_decay_end_step} \
  --max_episode_length ${max_episode_length} \
  --actor_train_interval_step ${actor_train_interval_step} \
  --train_interval ${train_interval} \
  --save_interval ${save_interval} \
  --running_avg_size ${running_avg_size} \
  --step_cost_factor ${step_cost_factor} \
  --collision_cost ${collision_cost} \
  --arrive_prob ${arrive_prob} \
  --fov_radius ${fov_radius} &

  sleep 2
done

