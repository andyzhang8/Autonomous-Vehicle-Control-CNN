from gymnasium.envs.registration import register

register(
    id='ppo-v0',
    entry_point='gym_ppo.envs:PPOEnv',
)
