This project applies Actor-Critic reinforcement learning to train an autonomous driving agent in the CARLA simulator. The system integrates convolutional neural networks (CNNs) for image processing, a replay buffer for efficient training, and a custom Gym-compatible CARLA environment to navigate complex simulation scenarios. The model is designed to enhance vehicle safety by enabling real-time decision-making and dynamic control in complex driving environments.

# What This Does
## Reinforcement Learning with Real-Time Visual Input:
The agent learns to control steering and throttle from high-dimensional camera images.

## Convolutional Neural Networks:
Processes visual observations (semantic_image) with CNNs for feature extraction.
## Custom Gym Environment:
Built around CARLA's API to simulate realistic driving scenarios with vehicles, pedestrians, and waypoints.
## Replay Buffer and Stable Training:
Implements efficient experience replay and target network updates for reinforcement learning stability.

# Setup
### Clone this repository:
##
	git clone https://github.com/your-username/carla-actor-critic.git
##
	cd carla-actor-critic

### Install the dependencies:
##
	pip install -r requirements.txt

### Download the CARLA simulator if you don’t have it already
https://carla.readthedocs.io/en/latest/start_quickstart/

### Start the CARLA server:
##
	./CarlaUE4.sh

### Train the agent:
##
	python run.py
# How it Works
### Image Processing:
The agent gets a semantic_image (RGB camera feed from CARLA) as input.
A CNN in both the Actor and Critic models extracts features from these images.

### Reinforcement Learning:
The Actor predicts the best action (steering and throttle).
The Critic evaluates how good the action is for the current state.

### Replay Buffer:
Experiences (state, action, reward, next state, done) are stored and sampled randomly during training to improve stability.
Soft Updates:
Gradually updates the target networks for smoother training.

### Custom CARLA Environment:
Wraps CARLA’s API to provide observations, rewards, and step logic in a Gym-compatible format.


