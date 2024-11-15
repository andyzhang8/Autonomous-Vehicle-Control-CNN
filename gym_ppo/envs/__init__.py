from gym.envs.registration import register
from gym_ppo.envs.carla_env import PPOEnv

# Register the environment with Gym
register(
    id='CarlaEnv-v0',  # Unique identifier for the CARLA environment
    entry_point='gym_ppo.envs.carla_env:PPOEnv',  # Full path to the environment class
    max_episode_steps=3000,
)
