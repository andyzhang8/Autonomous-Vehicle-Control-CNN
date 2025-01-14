import gymnasium as gym
import torch
import gym_ppo
import carla
import torch
import math
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import signal


def make_env(rank, params):
  def _init():
      env = gym.make('ppo-v0', params=params)
      # Custom close method to ensure CARLA processes are properly terminated
      def close_env():
        env.clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'vehicle.*', 'controller.ai.walker', 'walker.*'])
        print(f"Environment {rank} closed")
      
      # Ensure cleanup on exit
      signal.signal(signal.SIGINT, lambda *args: close_env())
      signal.signal(signal.SIGTERM, lambda *args: close_env())

      return env
  return _init
  
def train_agent(params, total_timesteps=500000):
  # Create vectorized environment
  num_envs = 1
  env = SubprocVecEnv([make_env(i, params) for i in range(num_envs)])
  
  # Add normalization
  env = VecNormalize(
      env,
      norm_obs=False,
  )
  def lr_schedule(initial_value: float, end_value: float, rate: float):
    """
    Learning rate schedule:
        Exponential decay by factors of 10 from initial_value to end_value.

    :param initial_value: Initial learning rate.
    :param rate: Exponential rate of decay. High values mean fast early drop in LR
    :param end_value: The final value of the learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: A float value between 0 and 1 that represents the remaining progress.
        :return: The current learning rate.
        """
        if progress_remaining <= 0:
            return end_value

        return end_value + (initial_value - end_value) * (10 ** (rate * math.log10(progress_remaining)))

    func.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"
    lr_schedule.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"

    return func

  model = PPO(
      "MultiInputPolicy",
      env,
      verbose=1,
      learning_rate=lr_schedule(1e-4, 1e-6, 2),
      gamma=0.98,
      gae_lambda=0.95,
      clip_range=0.2,
      ent_coef=0.05,
      n_epochs=15,
      n_steps=1024,
      policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                          net_arch=dict(pi=[500, 300], vf=[500, 300])),
      tensorboard_log="./carla_tensorboard/",
      device='auto'
  )

  
  model.learn(
      total_timesteps=total_timesteps,
      log_interval=1,
      progress_bar=True
  )
  
  return model


def main():
  # parameters for the gym_carla environment
  params = {
    'target_speed': 8, # target speed (m/s)
    'dt': 0.01,  # time interval between two frames
    'ego_vehicle_filter': 'vehicle.ford.mustang',  # filter for defining ego vehicle
    'steering_limits': [-1, 1],
    'velocity_limits': [-1, 1],
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'max_time_episode': 5000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane 
    'desired_speed': 25,  # desired speed (m/s)
    'max_speed': 40,
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'spec_dist': 10,
    'obs_size_x': 256,
    'obs_size_y': 256,
    'show_sensor': False,
    'wayp_sampling_resolution': 10, # waypoint sampling resolution
    'wayp_angle_bias': 5, # how much the actor can deviate from the waypoint
    'use_vae': False
  }




  model = train_agent(params=params, total_timesteps=500000)
  model.save("carla_ppo_agent_final")

  print("TRAIN DONE")
  num_envs = 1
  env = SubprocVecEnv([make_env(i, params) for i in range(num_envs)])
  # Add normalization
  env = VecNormalize(
      env,
      norm_obs=False,
      norm_reward=True
  )

  model = PPO.load("carla_ppo_agent_final", env=env, device="cuda")
  obs = env.reset()
  while True:
    steer = model.predict(obs)[0]
    env.step(steer)
    obs, reward, _, done = env.step(steer)



if __name__ == '__main__':
  main()

