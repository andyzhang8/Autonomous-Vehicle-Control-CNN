# Problem Formulation and Methodology
- Model to be used: [proximal policy optimization](#Proximal-Policy-Optimization)

## RL Setup
### Environment
- Learning agent with other dynamic actors
  - Dynamic actors will be driven by CARLA's autdriving algorithm
### Input
- One top-down semantically segmented image from CARLA's sensor library
  - CNN-based auto encoder to reduce dimensionality of the SS image. Use the bottleneck as the input to the agent's policy network
- Trajectory waypoints to guide nagivation. Waypoints are passed as a funcion ```f(p, w1, w2, ..., wn)```, where ```p``` is the position of the vehicle and ```w1-wn``` are the waypoints
  - f is defined as the average angle between the vehicle and the next 5 waypoints
### Output
- The policy network outputs the control actions ```(s, v)```. 
  - ```s``` is the predicted steer and ```v``` is the predicted velocity
  - NOTE: ```v``` is mapped to throttle and brake ```(t, b)``` using a PID controller.

### Reward Function
$R=R_s + R_d + I(c) * R_c$: where $R_s$ is the agent speed, $R_d$ is the perpendicular distance from the nominal trajectory, and $R_c$ will be applied if the actor collides with other agents or goes outside the road, determined by the indicator function $I(c)$


### Proximal Policy Optimization
- Reinforcement learning algorithm
- The agent takes a sample from the probabilty distribution. Actions with higher rewards have a higher chance of being chosen
