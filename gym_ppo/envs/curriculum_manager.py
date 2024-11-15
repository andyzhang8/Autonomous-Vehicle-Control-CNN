from gym_ppo.envs.route_planner import RoutePlanner

import random
import numpy as np
import math
import carla

class CurriculumRoutePlanner(RoutePlanner):
    def __init__(self, vehicle, curriculum_manager):
        super().__init__(vehicle, 15)
        self.curriculum = curriculum_manager
        
    def generate_route(self):
        """Generate route based on current curriculum level"""
        params = self.curriculum.get_current_params()
        target_route_length = params['route_length']
        max_curve_angle = params['max_curve_angle']
        
        
        # Draw route
        
        
        return self._waypoints_queue
    
    def _get_max_route_angle(self, route):
        """Calculate maximum angle between consecutive waypoints"""
        max_angle = 0
        for i in range(len(route) - 2):
            wp1 = route[i][0]
            wp2 = route[i+1][0]
            wp3 = route[i+2][0]
            
            # Get vectors between waypoints
            v1 = np.array([wp2.transform.location.x - wp1.transform.location.x,
                          wp2.transform.location.y - wp1.transform.location.y])
            v2 = np.array([wp3.transform.location.x - wp2.transform.location.x,
                          wp3.transform.location.y - wp2.transform.location.y])
            
            # Normalize vectors
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Calculate angle
            angle = abs(math.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))))
            max_angle = max(max_angle, angle)
        
        return max_angle
    
    def _get_curriculum_color(self):
        """Return color based on curriculum level for visualization"""
        colors = {
            0: carla.Color(r=0, g=255, b=0),    # Green for easiest
            1: carla.Color(r=255, g=255, b=0),  # Yellow
            2: carla.Color(r=255, g=165, b=0),  # Orange
            3: carla.Color(r=255, g=0, b=0),    # Red
            4: carla.Color(r=255, g=0, b=255)   # Purple for hardest
        }
        return colors[self.curriculum.current_level]

class CurriculumManager:
    def __init__(self):
        self.current_level = 0
        self.max_level = 4
        self.success_threshold = 0.7
        self.window_size = 50
        self.episode_outcomes = []
        
        self.level_params = {
            0: {  # Straight roads, slow speed
                'route_length': 10,
                'max_curve_angle': 10,
                'target_speed': 20,
                'min_waypoints_complete': 0.6  # 60% of route
            },
            1: {  # Gentle curves
                'route_length': 15,
                'max_curve_angle': 30,
                'target_speed': 30,
                'min_waypoints_complete': 0.7
            },
            2: {  # Moderate curves
                'route_length': 20,
                'max_curve_angle': 45,
                'target_speed': 40,
                'min_waypoints_complete': 0.8
            },
            3: {  # Sharp turns
                'route_length': 25,
                'max_curve_angle': 60,
                'target_speed': 45,
                'min_waypoints_complete': 0.8
            },
            4: {  # Full complexity
                'route_length': 30,
                'max_curve_angle': 90,
                'target_speed': 50,
                'min_waypoints_complete': 0.9
            }
        }
    
    def get_current_params(self):
        return self.level_params[self.current_level]
    
    def update_progress(self, success):
        self.episode_outcomes.append(float(success))
        if len(self.episode_outcomes) > self.window_size:
            self.episode_outcomes.pop(0)
            
        if self._check_advancement():
            self._advance_level()
    
    def _check_advancement(self):
        if len(self.episode_outcomes) < self.window_size:
            return False
        
        success_rate = sum(self.episode_outcomes) / len(self.episode_outcomes)
        return success_rate >= self.success_threshold
    
    def _advance_level(self):
        if self.current_level < self.max_level:
            self.current_level += 1
            self.episode_outcomes = []
            print(f"Advancing to curriculum level {self.current_level}")