#!/bin/sh
algo="maddpg"
experiment="pure_maddpg_v0"
env="tjc_gym:TrafficJunctionContinuous6-v0"
# render=True
buffer_size=500000
hidden_size1=400
hidden_size2=300
lr_actor=0.01
lr_critic=0.01
weight_decay=0
tau=0.01
batch_size=128
gamma=0.95
num_random_episodes=1
act_noise_std_start=0.3
act_noise_std_min=0.01
act_noise_decay_end_step=2500000
max_agent_episode_steps=500
actor_train_interval_step=10
train_interval=10
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
  --seed ${seed} \
  --env_name ${env} \
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
  --max_agent_episode_steps ${max_agent_episode_steps} \
  --actor_train_interval_step ${actor_train_interval_step} \
  --train_interval ${train_interval} \
  --episodes_per_epoch ${episodes_per_epoch} \
  --epochs ${epochs} \
  --num_eval_episodes ${num_eval_episodes} \
  --save_interval ${save_interval} \
  --step_cost_factor ${step_cost_factor} \
  --collision_cost ${collision_cost} \
  --arrive_prob ${arrive_prob} \
  --fov_radius ${fov_radius} &

  sleep 2
done

