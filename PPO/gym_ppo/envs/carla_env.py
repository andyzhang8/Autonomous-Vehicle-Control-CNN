from gymnasium import spaces
import gymnasium as gym
import carla

from gym_ppo.envs.route_planner import RoutePlanner
from gym_ppo.envs.curriculum_manager import Curriculum

from gym_ppo.envs.misc import *


import gc
import random
import time
import numpy as np
import math
import cv2


class PPOEnv(gym.Env):
  """PPO gymnasium environment for CARLA simulator."""
  def __init__(self, params):
    # env parameters
    self.desired_speed = params['target_speed']
    self.max_speed = params['max_speed']
    self.dt = params['dt']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.d_behind = params['d_behind']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.out_lane_thres = params['out_lane_thres']
    self.obs_size_x = params['obs_size_x']
    self.obs_size_y = params['obs_size_y']
    self.show_sensor = params['show_sensor']
    self.wayp_sampling_resolution = params['wayp_sampling_resolution']
    self.wayp_angle_bias = params['wayp_angle_bias']
    self.enable_vae = params['use_vae']


    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(4000.0)
    self.world = client.load_world(params['town'])
    print('Carla server connected!')

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    buildings = self.world.get_environment_objects(carla.CityObjectLabel.Buildings)
    terrain = self.world.get_environment_objects(carla.CityObjectLabel.Terrain)
    vegetation = self.world.get_environment_objects(carla.CityObjectLabel.Vegetation)
    lights = self.world.get_environment_objects(carla.CityObjectLabel.TrafficLight)
    ground = self.world.get_environment_objects(carla.CityObjectLabel.Ground)
    bridge = self.world.get_environment_objects(carla.CityObjectLabel.Bridge)
    fence = self.world.get_environment_objects(carla.CityObjectLabel.Fences)
    pole = self.world.get_environment_objects(carla.CityObjectLabel.Poles)
    wall = self.world.get_environment_objects(carla.CityObjectLabel.Walls)
    sky = self.world.get_environment_objects(carla.CityObjectLabel.Sky)
    other = self.world.get_environment_objects(carla.CityObjectLabel.Other)




    all = [*buildings, *terrain, *vegetation, *lights, *ground, *bridge, *fence, *pole, *wall, *sky, *other]
    all = [i.id for i in all]
    self.world.enable_environment_objects(all, False)
    # Actor blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')


    # Vehicle spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())


    # Set spectator to follow the car
    self.spectator = self.world.get_spectator()
    self.spectator.set_transform(carla.Transform(carla.Location(x=125,y=175,z=200), carla.Rotation(pitch=-90)))

    self.dist_m = params['spec_dist'] # how far behind the specator should be

    
    self.curriculum = Curriculum()

    
    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.synchronous_mode = True
    self.settings.fixed_delta_seconds = self.dt

    # Semantic segmentation sensor
    self.semantic_cc = carla.ColorConverter.CityScapesPalette
    self.semantic_img = np.zeros((self.obs_size_x, self.obs_size_y, 3), dtype=np.float32)
    self.semantic_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    self.semantic_bp.set_attribute('image_size_x', str(self.obs_size_x))
    self.semantic_bp.set_attribute('image_size_y', str(self.obs_size_y))
    self.semantic_trans = carla.Transform(carla.Location(x=3, z=1.5), carla.Rotation(pitch=15))
    self.semantic_bp.set_attribute('fov', str(150.0))

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length 
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')


    # action space
    self.action_space = spaces.Box(low=np.array([params['velocity_limits'][0], params['steering_limits'][0]]),
                                   high=np.array([params['velocity_limits'][1], params['steering_limits'][1]]), 
                                   dtype=np.float32)  # acc, steer

    self.speed = np.float32(0.0)
    
    self.vae = None
    if self.enable_vae:
      self.vae = load_vae('../../vae/best.tar', VAE.LSIZE)
      self.observation_space = spaces.Dict({
        'velocity': spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),
        'image': spaces.Box(low=-4, high=4, shape=(VAE.LSIZE,), dtype=np.float32),
        'waypoint_distance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        'waypoint_angle': spaces.Box(low=-360, high=360, shape=(1,), dtype=np.float32)
      })
    else:
      self.observation_space = spaces.Dict({
        'velocity': spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),
        'semantic_image': spaces.Box(low=0, high=255, shape=(self.obs_size_x-150, self.obs_size_y, 3), dtype=np.uint8),
        'waypoint_distance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        'waypoint_angle': spaces.Box(low=-360, high=360, shape=(1,), dtype=np.float32)
      })

  
  
    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0

    self.semantic_sensor = None
    self.collision_sensor = None

    self.waypoints = None


  def reset(self, seed=None, options=None):
    gc.collect()
    # Enable sync mode
    self._set_synchronous_mode(False)
    # # Clear sensor objects
    # Delete sensors, vehicles and walkers
    self.clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

    

    # Get vehicle collisions
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)


    # # Spawn the ego vehicle
    # ego_spawn_times = 0
    # while True:
    #   if ego_spawn_times > self.max_ego_spawn_times:
    #     self.clear_all_actors([])
    #     print("RESPAWN RESET TRIGGERED")
    #     self.reset()

    #   transform = random.choice(self.vehicle_spawn_points)

    #   if self._try_spawn_ego_vehicle_at(transform):
    #     break
    #   else:
    #     ego_spawn_times += 1
    #     time.sleep(0.1)
    self._try_spawn_ego_vehicle_at(self.curriculum.plan[0][0])
  


    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
    self.collision_sensor.listen(lambda event: get_collision_hist(event))
    def get_collision_hist(event):
      impulse = event.normal_impulse
      intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
      self.collision_hist.append(intensity)
      if len(self.collision_hist)>self.collision_hist_l:
        self.collision_hist.pop(0)
    self.collision_hist = []

    def _get_semantic_img(data):
      # Convert raw image data to a numpy array
      array = np.frombuffer(data.raw_data, dtype=np.uint8)
      array = np.reshape(array, (data.height, data.width, 4))
      array = array[150:, :, :3]
      # array = array[:, :, ::-1]
      self.semantic_img = array

    self.semantic_sensor = self.world.spawn_actor(self.semantic_bp, self.semantic_trans, attach_to=self.ego)
    self.semantic_sensor.listen(lambda data: _get_semantic_img(data))

    # Set spectator
    ego_pos = self.ego.get_transform()
    y, x = calcluate_sides(self.dist_m, ego_pos.rotation.yaw)
    self.spectator_pos = carla.Transform(ego_pos.location + carla.Location(x=-x, y=-y, z=5),
                                         carla.Rotation(yaw=ego_pos.rotation.yaw, pitch=-25))
    self.spectator.set_transform(self.spectator_pos)

    self.planner = RoutePlanner(self.ego, 5, self.curriculum.plan[0][1])
    self.waypoints, _, self.vehicle_front = self.planner.run_step()


    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)


    self.time_step=0

    return self._get_obs(), self._get_info()
  
  def step(self, action):

    velo = action[0]
    steer = action[1]
    throttle, brake = 0, 0


    if velo > 0:
      throttle = velo
    else:
      brake = velo


    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
    self.ego.apply_control(act)

    # ego_pos = self.ego.get_transform()
    # y, x = self._calcluate_sides(self.dist_m, ego_pos.rotation.yaw)
    # self.spectator_pos = carla.Transform(ego_pos.location + carla.Location(x=-x, y=-y, z=5),
    #                                      carla.Rotation(yaw=ego_pos.rotation.yaw, pitch=-25))
    # self.spectator.set_transform(self.spectator_pos)


    if self.show_sensor:
      self._render_semantic()

    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    if not hasattr(self, 'speed_history'):
        self.speed_history = []
    self.speed_history.append(speed)


    

    self.waypoints, _, self.vehicle_front = self.planner.run_step()

    reward = self._get_reward()

    dead = self._terminal()
    done = self._truncated()

    
    if done:
      reward += 5
    
    
    self.time_step += 1
    self.world.tick()

    return self._get_obs(), reward, done, dead, self._get_info()
  



  #================AUX FUNCTIONS================

  def _get_reward(self):
    # Get the vehicle's velocity vector
    velocity = self.ego.get_velocity()

    # Calculate forward speed (project velocity onto the forward vector of the vehicle)
    transform = self.ego.get_transform()
    forward_vector = transform.get_forward_vector()
    self.speed = velocity.x * forward_vector.x + velocity.y * forward_vector.y

    # Reward for speed (interpolated similar to reward_fn5)
    min_speed = 20.0  # Minimum speed for positive reward (replace with your CONFIG values)
    max_speed = 35.0  # Maximum allowable speed (replace with your CONFIG values)

    if self.speed < min_speed:
        speed_reward = self.speed / min_speed  # Interpolate in [0, min_speed]
    elif self.speed > self.desired_speed:
        speed_reward = max(1.0 - (self.speed - self.desired_speed) / (max_speed - self.desired_speed), 0.0)
    else:
        speed_reward = 1.0  # Reward is max when within [min_speed, target_speed]

    # Lane deviation penalty (similar to centering factor in reward_fn5)
    ego_x, ego_y = get_pos(self.ego)
    dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    max_distance = 2.0  # Maximum distance from center before heavy penalties (replace with your CONFIG value)
    centering_factor = max(1.0 - abs(dis) / max_distance, 0.0)

    # Alignment with the road (angle factor from reward_fn5)
    angle = self.planner.get_wayp_vec()[1]
    max_angle_center_lane = 90.0  # Maximum angle deviation in degrees
    angle_factor = max(1.0 - abs(angle) / max_angle_center_lane, 0.0)

    # Combine factors into a reward
    reward = speed_reward * centering_factor * angle_factor

    # Terminal penalties
    if abs(dis) > max_distance:
        reward = -10  # Heavy penalty for being out of bounds

    return reward


  def _get_info(self):
    info = {
      "speed": self.speed
    }

    return info
  
  def _get_obs(self):
    wayp_vec = self.planner.get_wayp_vec()
    if self.enable_vae:
      with torch.no_grad():
        frame = preprocess_frame(self.semantic_img)
        mu, logvar = self.vae.encode(frame)
        vae_latent = self.vae.reparameterize(mu, logvar)[0].cpu().detach().numpy().squeeze()
      observation = {
        'velocity': np.array([self.speed], dtype=np.float32),
        'image': vae_latent,
        'waypoint_distance': np.array([wayp_vec[0]], dtype=np.float32),
        'waypoint_angle': np.array([wayp_vec[1]], dtype=np.float32)
      }
    else:
      observation = {
        'velocity': np.array([self.speed], dtype=np.float32),
        'semantic_image':self.semantic_img,
        'waypoint_distance': np.array([wayp_vec[0]], dtype=np.float32),
        'waypoint_angle': np.array([wayp_vec[1]], dtype=np.float32)
      }

    return observation
  
  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    # If collides
    if len(self.collision_hist)>0:
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      return True
    
    if self.waypoints:

      # If out of lane
      dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
      if abs(dis) > self.out_lane_thres:
        return True
      
    if self.planner.route_done:
      return True
    
    
    return False
  
  def _truncated(self):
    return self.planner.route_done

  def _render_semantic(self):
    cv2.imshow("SS_cam", cv2.cvtColor(self.semantic_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp
  
  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      actor_poly_dict[actor.id]=poly
    return actor_poly_dict
  
  
  def _try_spawn_ego_vehicle_at(self, transform):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = None
    overlap = False
    for idx, poly in self.vehicle_polygons[-1].items():
      poly_center = np.mean(poly, axis=0)
      ego_center = np.array([transform.location.x, transform.location.y])
      dis = np.linalg.norm(poly_center - ego_center)
      if dis > 8:
        continue
      else:
        overlap = True
        break

    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

    if vehicle is not None:
      self.ego=vehicle
      return True
      
    return False
  
  def clear_all_actors(self, actor_filters):

    if hasattr(self, "ego") and self.ego != None:
      self.ego.destroy()
      self.ego = None
    if self.semantic_sensor != None:
      self.semantic_sensor.stop()
      self.semantic_sensor.destroy()
    self.semantic_sensor = None
    if self.collision_sensor != None:
      self.collision_sensor.stop()
      self.collision_sensor.destroy()
    self.collision_sensor = None
    



