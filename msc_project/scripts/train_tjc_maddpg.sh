#!/bin/sh
algo="maddpg"
experiment="pure_maddpg_v0"
env="tjc_gym:TrafficJunctionContinuous6-v0"
# render=True
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
act_noise_std_min=0.01
act_noise_decay_end_step=300000
max_agent_episode_steps=500
actor_train_interval_step=1
train_interval=1
episodes_per_epoch=20
epochs=100
num_eval_episodes=100
save_interval=10
step_cost_factor=-0.01
collision_cost=-10
arrive_prob=0.05
fov_radius=3

echo "Env is ${env} and algo is ${algo}"

seeds=(1 2 3 4 5)
for seed in "${seeds[@]}"
do
  echo "Running experiment with seed = ${seed}"

  python ./train/train_tjc.py \
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
  --step_cost_factor ${step_cost_factor} \
  --collision_cost ${collision_cost} \
  --arrive_prob ${arrive_prob} \
  --fov_radius ${fov_radius} &

  sleep 2
done

