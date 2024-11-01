# train.py
import torch
import torch.optim as optim
from carla_env import CarlaEnv
from replay_buffer import ReplayBuffer
from model import Actor, Critic

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

gamma = 0.99
batch_size = 64
tau = 0.001

env = CarlaEnv()
actor = Actor().to(device)
critic = Critic().to(device)
actor_target = Actor().to(device)
critic_target = Critic().to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer()

for episode in range(1000):
    state = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    episode_reward = 0
    done = False

    while not done:
        # Actor forward pass
        steering, throttle = actor(state_tensor)
        action = torch.cat([steering, throttle], dim=1).detach().cpu().numpy().squeeze()  # Move to CPU for env step
        
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)

        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = (
                states.to(device),
                actions.to(device),
                rewards.to(device),
                next_states.to(device),
                dones.to(device)
            )

            # Critic forward pass and loss calculation
            next_steering, next_throttle = actor_target(next_states)
            next_actions = torch.cat([next_steering, next_throttle], dim=1)
            target_q_values = rewards + (1 - dones) * gamma * critic_target(next_states, next_actions)

            critic_loss = torch.nn.MSELoss()(critic(states, actions), target_q_values)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(states, torch.cat(actor(states), dim=1)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft update of target networks
            for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        episode_reward += reward
        state = next_state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    print(f"Episode {episode}, Reward: {episode_reward}")
