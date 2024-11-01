# carla_env.py
import carla
import numpy as np
import random
import cv2

class CarlaEnv:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town03')
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.image = None
        self.collision_occurred = False

    def reset(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()

        # Spawn vehicle
        vehicle_bp = self.blueprint_library.filter('model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Attach camera sensor
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda data: self.process_image(data))

        # Attach collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, camera_transform, attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.process_collision(event))

        self.image = None
        self.collision_occurred = False
        while self.image is None:
            pass
        return self.image

    def process_image(self, data):
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))
        self.image = array[:, :, :3]  # Remove alpha channel

    def process_collision(self, event):
        self.collision_occurred = True  # Set collision flag

    def step(self, action):
        throttle, steering = action
        control = carla.VehicleControl(throttle=float(throttle), steer=float(steering))
        self.vehicle.apply_control(control)

        # Calculate reward and done status
        reward = self.get_reward()
        done = self.check_done()
        return self.image, reward, done, {}

    def get_reward(self):
        # Give a positive reward for speed and penalize for collision
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        reward = speed if not self.collision_occurred else -10
        return reward

    def check_done(self):
        # End episode if collision has occurred
        return self.collision_occurred

    def render(self):
        if self.image is not None:
            cv2.imshow("CARLA", self.image)
            cv2.waitKey(1)

    def close(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        cv2.destroyAllWindows()
