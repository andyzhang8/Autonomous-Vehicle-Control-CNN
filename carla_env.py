import carla
import numpy as np
import random
import cv2
import time
import math
import gymnasium as gym
from gymnasium import spaces
import gc

class CarlaEnv(gym.Env):
    def __init__(self, params):
        # init
        self.client = carla.Client('localhost', params['port'])
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(params['town'])
        
        self.desired_speed = params['target_speed']
        self.max_time_episode = params['max_time_episode']
        self.obs_size_x = params['obs_size_x']
        self.obs_size_y = params['obs_size_y']
        self.dt = params['dt']
        self.show_sensor = params.get('show_sensor', False)
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.image = None
        self.collision_occurred = False
        self.speed_history = []
        self.time_step = 0
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([params['steering_limits'][0], params['velocity_limits'][0]]),
                                       high=np.array([params['steering_limits'][1], params['velocity_limits'][1]]),
                                       dtype=np.float32)
        self.observation_space = spaces.Dict({
            'semantic_image': spaces.Box(low=0, high=255, shape=(self.obs_size_x, self.obs_size_y, 3), dtype=np.float32),
            'waypoint_distance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'waypoint_angle': spaces.Box(low=-360, high=360, shape=(1,), dtype=np.float32)
        })

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)

        self.spectator = self.world.get_spectator()
        self.dist_m = params.get('spec_dist', 10.0)

        self.image = None
        
    def reset(self):
        self.clear_all_actors()
        self._set_synchronous_mode(False)
        
        self.time_step = 0
        
        vehicle_bp = self.world.get_blueprint_library().filter('model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        def process_image(data):
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = np.reshape(array, (data.height, data.width, 4))
            self.image = array[:, :, :3]  # Remove alpha channel

        # camera sensor
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.obs_size_x))
        camera_bp.set_attribute('image_size_y', str(self.obs_size_y))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda data: process_image(data))
        
        # collision sensor
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, camera_transform, attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.process_collision(event))

        self.collision_occurred = False
        self.speed_history = []

        self.settings.synchronous_mode=True
        self.world.apply_settings(self.settings)




        return self._get_obs()

    

    def process_collision(self, event):
        self.collision_occurred = True

    def step(self, action):
        self.time_step += 1
        throttle, steer = action
        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer))
        self.vehicle.apply_control(control)

        self._update_spectator()
        
        reward = self.get_reward()
        done = self.check_done()
        
        self.world.tick()
        return self._get_obs(), reward, done, {}

    def get_reward(self):
        # Calculate speed
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        r_speed = 1.0 if abs(speed - self.desired_speed) < 1.0 else -0.5 * abs(speed - self.desired_speed)
        
        # Collision penalty
        r_collision = -10 if self.collision_occurred else 0
        return r_speed + r_collision

    def check_done(self):
        if self.collision_occurred:
            self.clear_all_actors()
            return True
        if self.time_step > self.max_time_episode:
            self.clear_all_actors()
            return True
        return False

    def render(self, steering=None, throttle=None):
        if self.image is not None:
            frame = self.image.copy()
            if steering is not None and throttle is not None:
                cv2.putText(frame, f"Steering: {steering:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Throttle: {throttle:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("CARLA", frame)
            cv2.waitKey(1)

    def close(self):
        self.clear_all_actors()
        cv2.destroyAllWindows()

    def _get_obs(self):
        wayp_vec = [0, 0]  # Placeholder for waypoint vector
        return {
            'semantic_image': self.image,
            'waypoint_distance': np.array([wayp_vec[0]], dtype=np.float32),
            'waypoint_angle': np.array([wayp_vec[1]], dtype=np.float32)
        }

    def _update_spectator(self):
        ego_pos = self.vehicle.get_transform()
        y, x = self._calculate_sides(self.dist_m, ego_pos.rotation.yaw)
        self.spectator_pos = carla.Transform(ego_pos.location + carla.Location(x=-x, y=-y, z=5),
                                             carla.Rotation(yaw=ego_pos.rotation.yaw, pitch=-25))
        self.spectator.set_transform(self.spectator_pos)

    def clear_all_actors(self):
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        if self.camera:
            self.camera.destroy()
            self.camera = None
        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None

    def _calculate_sides(self, hyp, angle):
        angle_radians = math.radians(angle)
        return hyp * math.sin(angle_radians), hyp * math.cos(angle_radians)
    
    def _set_synchronous_mode(self, synchronous = True):
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)
