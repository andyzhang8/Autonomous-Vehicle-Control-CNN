from gymnasium import spaces
import gymnasium as gym
import carla

from gym_ppo.envs.curriculum_manager import CurriculumRoutePlanner, CurriculumManager
from gym_ppo.envs.route_planner import RoutePlanner

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
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
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


    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(4000.0)
    self.world = client.load_world(params['town'])
    print('Carla server connected!')


    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)


    # Actor blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')


    # Vehicle spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())


    # Set spectator to follow the car
    self.spectator = self.world.get_spectator()
    self.dist_m = params['spec_dist'] # how far behind the specator should be

    
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
    self.semantic_trans = carla.Transform(carla.Location(x=1.5, z=1.5))

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length 
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')


    # action space
    self.action_space = spaces.Box(low=np.array([params['steering_limits'][0],params['velocity_limits'][0]]),
                                   high=np.array([params['steering_limits'][1],params['velocity_limits'][1]]), 
                                   dtype=np.float32)  # acc, steer

    self.observation_space = spaces.Dict({
      'semantic_image': spaces.Box(low=0, high=255, shape=(self.obs_size_x, self.obs_size_y, 3), dtype=np.float32),
      'waypoint_distance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
      'waypoint_angle': spaces.Box(low=-360, high=360, shape=(1,), dtype=np.float32)
    })

  
    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0

    self.semantic_sensor = None
    self.collision_sensor = None


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


    # Spawn the ego vehicle
    ego_spawn_times = 0
    self.ego_init_trans = None
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        self.reset()

      transform = random.choice(self.vehicle_spawn_points)

      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        ego_spawn_times += 1
        time.sleep(0.1)


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
      data.convert(self.semantic_cc)
      array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
      
      # Reshape array into the image (height, width, 4) because it contains 4 channels
      array = np.reshape(array, (data.height, data.width, 4))
      
      # Drop the alpha channel
      array = array[:, :, :3]
      array = array[:, :, ::-1]

      array = array.astype(np.float32) / 255.0
    
      # Store the image in the class attribute
      self.semantic_img = array

    self.semantic_sensor = self.world.spawn_actor(self.semantic_bp, self.semantic_trans, attach_to=self.ego)
    self.semantic_sensor.listen(lambda data: _get_semantic_img(data))

    # Set spectator
    ego_pos = self.ego.get_transform()
    y, x = self._calcluate_sides(self.dist_m, ego_pos.rotation.yaw)
    self.spectator_pos = carla.Transform(ego_pos.location + carla.Location(x=-x, y=-y, z=5),
                                         carla.Rotation(yaw=ego_pos.rotation.yaw, pitch=-25))
    self.spectator.set_transform(self.spectator_pos)

    self.planner = RoutePlanner(self.ego, 5)


    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)


    self.time_step=0

    return self._get_obs(), {}
  
  def step(self, action):

    velo = action[0]
    steer = action[1]
    throttle, brake = 0, 0

    if velo > 0:
      throttle = velo
    else:
      brake = velo

    throttle = 0.3
    brake = 0


    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
    self.ego.apply_control(act)

    ego_pos = self.ego.get_transform()
    y, x = self._calcluate_sides(self.dist_m, ego_pos.rotation.yaw)
    self.spectator_pos = carla.Transform(ego_pos.location + carla.Location(x=-x, y=-y, z=5),
                                         carla.Rotation(yaw=ego_pos.rotation.yaw, pitch=-25))
    self.spectator.set_transform(self.spectator_pos)


    if self.show_sensor:
      self._render_semantic()

    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    if not hasattr(self, 'speed_history'):
        self.speed_history = []
    self.speed_history.append(speed)

    done = self._terminal()

    if done:
      self.reset()

    self.planner.run_step()
    

    self.time_step += 1
    self.world.tick()

    return self._get_obs(), self._get_reward(), self._truncated(), done, self._get_info()
  




  
  #================AUX FUNCTIONS================

  def _get_reward(self):
    # Calculate car speed
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    
    # reward for maintaining speed

    # higher penalty for overspeed
    # Reward for maintaining speed near the desired speed
    # if abs(speed - self.desired_speed) < 0.5:  # Increased threshold for "near"
    #     r_speed = 1.0
    # else:
    #     r_speed = -min(2, abs(speed - self.desired_speed))  # Reduced penalty
    # # Give a bonus when within 1 m/s of the target speed
    # if abs(speed - self.desired_speed) < 1.0:
    #     r_speed = 2

    r_speed = 0


    # reward for steering:
    r_steer = 0
    delta_steer = 0
    if hasattr(self, 'prev_steer'):
        delta_steer = abs(self.ego.get_control().steer - self.prev_steer)
    # Reward smoother steering
     # reward for smoother steering
    if delta_steer < 0.05:  # Increased threshold for "smooth"
        r_steer = 1.0
    else:
        r_steer -= min(delta_steer / 10, 2)  # Softer penalty
    self.prev_steer = self.ego.get_control().steer
    
    
    # waypoint reward
    wayp_vec = self.planner.get_wayp_vec()
    wayp_dis = wayp_vec[0]

    # negative reward for being far away from the target
    wayp_dis_sat = max(0, 20 / (1 + wayp_dis))  # Saturation component
    r_wayp_dis = -wayp_dis_sat

    if speed < 0.2 and r_wayp_dis > 15:
      r_wayp_dis = -5

    # reward for being at the same angle of the target waypoint
    wayp_angle = wayp_vec[1]
    r_wayp_angle = 0.05 * -abs(wayp_angle)


    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -1

    reward = r_speed + r_steer + r_wayp_dis + r_wayp_angle + 20  * r_collision
    return reward

  
  # determines the type of collision the actor has
  #TODO determine how the car collided and apply reward as such
  def _indicator():
    pass



  def _get_info(self):
    info = {
      # 'curr_waypoint': self.planner.route[self.planner.curr_wayp][0],
      # 'curr_waypoint_distance': copy.deepcopy(self.planner.get_wayp_vector()[0]), # the issue is HERE! cannot serialize carla.Waypoint?
      # 'curr_waypoint_angle': copy.deepcopy(self.planner.get_wayp_vector()[1]),
    }

    return info
  
  def _get_obs(self):
    wayp_vec = self.planner.get_wayp_vec()
    observation = {
      'semantic_image':self.semantic_img,
      'waypoint_distance': np.array([wayp_vec[0]], dtype=np.float32),
      'waypoint_angle': np.array([wayp_vec[1]], dtype=np.float32)
    }



    return observation
  
  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    # ego_x, ego_y = self._get_pos(self.ego)

    # If collides
    if len(self.collision_hist)>0:
      self.clear_all_actors([])
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      self.clear_all_actors([])
      return True
    
    
    return False
  def _truncated(self):
    return False
    # return self.planner.route_complete()

  def _render_semantic(self):
    cv2.imshow("SS_cam", self.semantic_img)
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
    # Check if ego position overlaps with surrounding vehicles
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
      self.semantic_sensor.stop() # deleting this line removes the invalid stream id , but also causes seg fault
      self.semantic_sensor.destroy()
    self.semantic_sensor = None
    if self.collision_sensor != None:
      self.collision_sensor.stop()
      self.collision_sensor.destroy()
    self.collision_sensor = None
    

      
      

  def _calcluate_sides(self, hyp, angle):
    angle_radians = math.radians(angle)
    opp = hyp * math.sin(angle_radians)
    adj = hyp * math.cos(angle_radians)

    return opp, adj
  
  def _get_pos(self, vehicle):
    """
    Get the position of a vehicle
    :param vehicle: the vehicle whose position is to get
    :return: speed as a float in Kmh
    """
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    return x, y
  
  def _get_speed(self, vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
  
  
  def _check_success(self):
    """Define what constitutes a successful episode"""
    if len(self.collision_hist) > 0:
        return False
        
    params = self.curriculum.get_current_params()

    # # Check if enough waypoints were completed
    # waypoints_completed = self.planner.curr_wayp / len(self.planner.route)
    # if waypoints_completed < params['min_waypoints_complete']:
    #     return False
        
    # Check if average speed was maintained
    # (You'll need to track speed during episode)
    if hasattr(self, 'speed_history'):
        avg_speed = sum(self.speed_history) / len(self.speed_history)
        if avg_speed < params['target_speed'] * 0.7:  # 70% of target speed
            return False
        
