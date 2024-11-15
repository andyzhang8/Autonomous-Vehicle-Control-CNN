import torch
from carla_env import CarlaEnv
from model import Actor, Critic
from replay_buffer import ReplayBuffer
import signal

# Hyperparameters
REPLAY_BUFFER_CAPACITY = 10000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 1e-4
TOTAL_TIMESTEPS = 300000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
actor = Actor().to(device)
critic = Critic().to(device)
target_actor = Actor().to(device)
target_critic = Critic().to(device)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

# Optimizers
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

# Replay Buffer
replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)


def soft_update(target, source, tau):
    """Soft update of the target network parameters."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def compute_td_error(states, actions, rewards, next_states, dones):
    """Compute TD error for the Critic."""
    with torch.no_grad():
        next_actions = target_actor(next_states)
        target_q_values = rewards + (1 - dones) * GAMMA * target_critic(next_states, next_actions)

    q_values = critic(states, actions)
    td_error = torch.mean((q_values - target_q_values) ** 2)
    return td_error


def train_actor_critic():
    """Train the Actor and Critic networks using experiences from the ReplayBuffer."""
    if len(replay_buffer) < BATCH_SIZE:
        return

    # Sample a batch of experiences
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Critic loss
    td_error = compute_td_error(states, actions, rewards, next_states, dones)
    critic_optimizer.zero_grad()
    td_error.backward()
    critic_optimizer.step()

    # Actor loss
    predicted_actions = actor(states)
    actor_loss = -critic(states, predicted_actions).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update target networks
    soft_update(target_actor, actor, TAU)
    soft_update(target_critic, critic, TAU)


def train_agent(params):
    env = CarlaEnv(params)
    obs = env.reset()

    for timestep in range(TOTAL_TIMESTEPS):
        obs_tensor = torch.FloatTensor([obs['semantic_image']]).to(device)
        steer, throttle = actor(obs_tensor).detach().cpu().numpy()[0]
        action = [throttle, steer]

        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(
            obs['semantic_image'],
            action,
            reward,
            next_obs['semantic_image'],
            done
        )

        train_actor_critic()

        if done:
            obs = env.reset()
        else:
            obs = next_obs

        if timestep % 1000 == 0:
            print(f"Timestep: {timestep}")

    return actor


def main():
    params = {
        'target_speed': 8,
        'number_of_vehicles': 1,
        'number_of_walkers': 0,
        'dt': 0.01,
        'ego_vehicle_filter': 'vehicle.ford.mustang',
        'steering_limits': [-1, 1],
        'velocity_limits': [-1, 1],
        'port': 2000,
        'town': 'Town01',
        'max_time_episode': 3000,
        'max_waypt': 12,
        'd_behind': 12,
        'out_lane_thres': 2.0,
        'desired_speed': 8,
        'max_ego_spawn_times': 200,
        'spec_dist': 10,
        'obs_size_x': 256,
        'obs_size_y': 256,
        'show_sensor': False,
        'wayp_sampling_resolution': 10,
        'wayp_angle_bias': 5,
    }

    trained_actor = train_agent(params)
    torch.save(trained_actor.state_dict(), "carla_actor.pth")
    print("Training Complete and Model Saved!")


if __name__ == '__main__':
    main()
