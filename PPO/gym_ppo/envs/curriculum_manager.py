import carla
import random

class Curriculum:

    straight = {
        "Town03" : [
            [carla.Transform(carla.Location(x=-65.0, y=-136.0, z=1), carla.Rotation(pitch=0.0, yaw=2.0, roll=0.0)), 
             carla.Transform(carla.Location(x=-16, y=-135.0, z=1.0),carla.Rotation(pitch=0.0, yaw=2.0, roll=0.0))],
            [(-1.0, -124.0), (-1.0, -53.0)],
            [(33.0, 7.0), (217.0, 9.0)],
            [(-10.0, 43.0), (-10, 118)],
            [(-73.0, 120.0), (-74.0, 17.0)]
        ]
    }
    

    def __init__(self, current_plan=None, town=None):
        self.town = "Town03"
        if town:
            self.town = town

        self.plan = Curriculum.straight[self.town]
        if current_plan:
            self.plan = current_plan


    def get_random_start(self):
        return random.choice(self.plan)