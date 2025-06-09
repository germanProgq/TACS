#!/usr/bin/env python3
"""
Generate synthetic traffic dataset for TACSNet training.
Creates realistic annotated images with cars, pedestrians, and cyclists.
Designed for recursive training with consistent patterns.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import argparse
import hashlib
from datetime import datetime

class SyntheticTrafficGenerator:
    def __init__(self, output_dir, num_train=1000, num_val=200, seed=None):
        self.output_dir = output_dir
        self.num_train = num_train
        self.num_val = num_val
        self.image_size = (416, 416)
        
        # Set seed for reproducibility across runs
        if seed is None:
            seed = int(hashlib.md5(output_dir.encode()).hexdigest()[:8], 16)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Class definitions with realistic properties
        self.classes = {
            0: {
                'name': 'car',
                'color_variants': [(200, 50, 50), (50, 50, 200), (150, 150, 150), (50, 50, 50)],
                'size_range': (60, 120),  # Realistic car sizes
                'aspect_ratio': (1.5, 2.0),  # Width/height ratio
                'speed_range': (0.5, 2.0),  # Pixels per frame
                'lane_preference': 'road'
            },
            1: {
                'name': 'pedestrian', 
                'color_variants': [(100, 180, 100), (180, 100, 100), (100, 100, 180)],
                'size_range': (20, 35),  # Human-sized
                'aspect_ratio': (0.3, 0.5),
                'speed_range': (0.1, 0.5),
                'lane_preference': 'sidewalk'
            },
            2: {
                'name': 'cyclist',
                'color_variants': [(50, 100, 200), (200, 100, 50), (100, 200, 50)],
                'size_range': (35, 55),  # Bike + rider
                'aspect_ratio': (0.8, 1.2),
                'speed_range': (0.3, 1.0),
                'lane_preference': 'bike_lane'
            }
        }
        
        # Traffic scenario templates for consistent behavior
        self.scenario_templates = [
            'intersection',
            'straight_road',
            'pedestrian_crossing',
            'bike_lane',
            'parking_area'
        ]
        
        # Time of day effects
        self.time_conditions = [
            {'name': 'day', 'brightness': 1.0, 'contrast': 1.0},
            {'name': 'dusk', 'brightness': 0.7, 'contrast': 0.9},
            {'name': 'night', 'brightness': 0.3, 'contrast': 0.7},
            {'name': 'dawn', 'brightness': 0.6, 'contrast': 0.8}
        ]
        
        # Weather conditions
        self.weather_conditions = [
            {'name': 'clear', 'visibility': 1.0, 'noise': 0.0},
            {'name': 'rain', 'visibility': 0.7, 'noise': 0.2},
            {'name': 'fog', 'visibility': 0.5, 'noise': 0.1},
            {'name': 'overcast', 'visibility': 0.9, 'noise': 0.05}
        ]
        
        # Create directory structure
        os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)
        
        # Initialize persistent scene elements
        self.initialize_scene_database()
    
    def initialize_scene_database(self):
        """Initialize persistent scene elements for consistency"""
        self.scene_db = {
            'road_layouts': self.generate_road_layouts(),
            'traffic_patterns': self.generate_traffic_patterns(),
            'object_behaviors': self.generate_object_behaviors()
        }
    
    def generate_road_layouts(self):
        """Generate consistent road layout templates"""
        layouts = {}
        
        # Intersection layout
        layouts['intersection'] = {
            'roads': [
                {'start': (0, 208), 'end': (416, 208), 'width': 80},
                {'start': (208, 0), 'end': (208, 416), 'width': 80}
            ],
            'sidewalks': [
                {'start': (0, 168), 'end': (416, 168), 'width': 10},
                {'start': (0, 248), 'end': (416, 248), 'width': 10},
                {'start': (168, 0), 'end': (168, 416), 'width': 10},
                {'start': (248, 0), 'end': (248, 416), 'width': 10}
            ],
            'crosswalks': [
                {'x': 208, 'y': 168, 'width': 80, 'height': 10},
                {'x': 168, 'y': 208, 'width': 10, 'height': 80}
            ]
        }
        
        # Straight road layout
        layouts['straight_road'] = {
            'roads': [
                {'start': (0, 208), 'end': (416, 208), 'width': 120}
            ],
            'sidewalks': [
                {'start': (0, 148), 'end': (416, 148), 'width': 15},
                {'start': (0, 268), 'end': (416, 268), 'width': 15}
            ],
            'bike_lanes': [
                {'start': (0, 163), 'end': (416, 163), 'width': 10},
                {'start': (0, 253), 'end': (416, 253), 'width': 10}
            ]
        }
        
        return layouts
    
    def generate_traffic_patterns(self):
        """Generate realistic traffic flow patterns"""
        patterns = {
            'rush_hour': {
                'car_density': 0.7,
                'pedestrian_density': 0.5,
                'cyclist_density': 0.3,
                'speed_modifier': 0.5
            },
            'normal': {
                'car_density': 0.4,
                'pedestrian_density': 0.3,
                'cyclist_density': 0.2,
                'speed_modifier': 1.0
            },
            'light': {
                'car_density': 0.2,
                'pedestrian_density': 0.1,
                'cyclist_density': 0.1,
                'speed_modifier': 1.2
            }
        }
        return patterns
    
    def generate_object_behaviors(self):
        """Generate consistent object movement behaviors"""
        behaviors = {
            'car': {
                'straight': {'velocity': (2.0, 0), 'variance': 0.1},
                'turning_left': {'velocity': (1.0, -1.0), 'variance': 0.2},
                'turning_right': {'velocity': (1.0, 1.0), 'variance': 0.2},
                'stopped': {'velocity': (0, 0), 'variance': 0.0}
            },
            'pedestrian': {
                'walking': {'velocity': (0.3, 0), 'variance': 0.2},
                'crossing': {'velocity': (0, 0.3), 'variance': 0.1},
                'standing': {'velocity': (0, 0), 'variance': 0.0}
            },
            'cyclist': {
                'riding': {'velocity': (1.0, 0), 'variance': 0.15},
                'slow': {'velocity': (0.5, 0), 'variance': 0.1},
                'stopped': {'velocity': (0, 0), 'variance': 0.0}
            }
        }
        return behaviors
    
    def generate_background(self, scenario='straight_road', time_condition='day', weather='clear'):
        """Generate realistic road background based on scenario"""
        # Base road color depends on time and weather
        time_data = next(t for t in self.time_conditions if t['name'] == time_condition)
        weather_data = next(w for w in self.weather_conditions if w['name'] == weather)
        
        base_color = int(80 * time_data['brightness'])
        img = Image.new('RGB', self.image_size, color=(base_color, base_color, base_color))
        draw = ImageDraw.Draw(img)
        
        # Get road layout
        layout = self.scene_db['road_layouts'].get(scenario, self.scene_db['road_layouts']['straight_road'])
        
        # Draw roads
        for road in layout.get('roads', []):
            self.draw_road_segment(draw, road, time_data['brightness'])
        
        # Draw sidewalks
        for sidewalk in layout.get('sidewalks', []):
            self.draw_sidewalk(draw, sidewalk, time_data['brightness'])
        
        # Draw bike lanes
        for bike_lane in layout.get('bike_lanes', []):
            self.draw_bike_lane(draw, bike_lane, time_data['brightness'])
        
        # Draw crosswalks
        for crosswalk in layout.get('crosswalks', []):
            self.draw_crosswalk(draw, crosswalk, time_data['brightness'])
        
        # Apply weather effects
        img = self.apply_weather_effects(img, weather_data)
        
        # Add realistic texture and noise
        pixels = np.array(img)
        
        # Add road texture
        texture = np.random.normal(0, 5 * weather_data['noise'] + 2, pixels.shape)
        pixels = np.clip(pixels + texture, 0, 255).astype(np.uint8)
        
        # Add tire marks and oil stains for realism
        img = Image.fromarray(pixels)
        self.add_road_details(img, draw)
        
        return img
    
    def draw_road_segment(self, draw, road, brightness):
        """Draw a road segment with lane markings"""
        road_color = int(60 * brightness)
        draw.rectangle([road['start'][0], road['start'][1] - road['width']//2,
                       road['end'][0], road['end'][1] + road['width']//2],
                      fill=(road_color, road_color, road_color))
        
        # Lane markings
        marking_color = int(200 * brightness)
        if road['start'][1] == road['end'][1]:  # Horizontal road
            y = road['start'][1]
            for x in range(0, self.image_size[0], 40):
                draw.rectangle([x, y-2, x+20, y+2], fill=(marking_color, marking_color, marking_color))
        else:  # Vertical road
            x = road['start'][0]
            for y in range(0, self.image_size[1], 40):
                draw.rectangle([x-2, y, x+2, y+20], fill=(marking_color, marking_color, marking_color))
    
    def draw_sidewalk(self, draw, sidewalk, brightness):
        """Draw sidewalk with texture"""
        sidewalk_color = int(100 * brightness)
        draw.rectangle([sidewalk['start'][0], sidewalk['start'][1] - sidewalk['width']//2,
                       sidewalk['end'][0], sidewalk['end'][1] + sidewalk['width']//2],
                      fill=(sidewalk_color, sidewalk_color, sidewalk_color))
    
    def draw_bike_lane(self, draw, bike_lane, brightness):
        """Draw bike lane with markings"""
        lane_color = int(70 * brightness)
        draw.rectangle([bike_lane['start'][0], bike_lane['start'][1] - bike_lane['width']//2,
                       bike_lane['end'][0], bike_lane['end'][1] + bike_lane['width']//2],
                      fill=(lane_color, lane_color + 10, lane_color))
    
    def draw_crosswalk(self, draw, crosswalk, brightness):
        """Draw pedestrian crosswalk"""
        stripe_color = int(180 * brightness)
        for i in range(0, crosswalk['width'], 10):
            draw.rectangle([crosswalk['x'] + i, crosswalk['y'],
                           crosswalk['x'] + i + 5, crosswalk['y'] + crosswalk['height']],
                          fill=(stripe_color, stripe_color, stripe_color))
    
    def apply_weather_effects(self, img, weather_data):
        """Apply weather-specific visual effects"""
        if weather_data['name'] == 'rain':
            # Add rain streaks
            draw = ImageDraw.Draw(img)
            for _ in range(100):
                x = random.randint(0, self.image_size[0])
                y = random.randint(0, self.image_size[1])
                draw.line([x, y, x+2, y+10], fill=(150, 150, 180), width=1)
        
        elif weather_data['name'] == 'fog':
            # Apply fog effect
            fog_layer = Image.new('RGBA', self.image_size, (200, 200, 200, 100))
            img = Image.alpha_composite(img.convert('RGBA'), fog_layer).convert('RGB')
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        return img
    
    def add_road_details(self, img, draw):
        """Add realistic road details like tire marks and stains"""
        # Tire marks
        for _ in range(random.randint(2, 5)):
            x_start = random.randint(0, self.image_size[0])
            y_start = random.randint(150, 250)
            length = random.randint(50, 200)
            draw.line([x_start, y_start, x_start + length, y_start + random.randint(-5, 5)],
                     fill=(40, 40, 40), width=3)
        
        # Oil stains
        for _ in range(random.randint(1, 3)):
            x = random.randint(100, 300)
            y = random.randint(180, 230)
            radius = random.randint(10, 25)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                        fill=(30, 30, 35))
    
    def generate_object(self, class_id, object_state=None):
        """Generate realistic object representation with state"""
        class_info = self.classes[class_id]
        
        # Determine size with aspect ratio
        base_size = random.randint(*class_info['size_range'])
        aspect_ratio = random.uniform(*class_info['aspect_ratio'])
        w = int(base_size * aspect_ratio)
        h = base_size
        
        # Select color variant
        color = random.choice(class_info['color_variants'])
        
        # Create object image
        obj_img = Image.new('RGBA', (w, h), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(obj_img)
        
        if class_id == 0:  # Car
            self.draw_realistic_car(draw, w, h, color, object_state)
        elif class_id == 1:  # Pedestrian
            self.draw_realistic_pedestrian(draw, w, h, color, object_state)
        elif class_id == 2:  # Cyclist
            self.draw_realistic_cyclist(draw, w, h, color, object_state)
        
        # Apply slight rotation for realism
        if object_state and 'rotation' in object_state:
            obj_img = obj_img.rotate(object_state['rotation'], expand=True)
        
        return obj_img
    
    def draw_realistic_car(self, draw, w, h, color, state):
        """Draw a realistic car with details"""
        # Car body
        body_points = [
            (w*0.1, h*0.4),  # Front bumper
            (w*0.15, h*0.3),  # Hood start
            (w*0.35, h*0.2),  # Windshield bottom
            (w*0.45, h*0.15), # Roof front
            (w*0.65, h*0.15), # Roof back
            (w*0.75, h*0.2),  # Rear windshield
            (w*0.9, h*0.35),  # Trunk
            (w*0.95, h*0.45), # Rear bumper
            (w*0.95, h*0.7),  # Bottom rear
            (w*0.85, h*0.75), # Rear wheel well
            (w*0.7, h*0.75),  # Between wheels
            (w*0.3, h*0.75),  # Between wheels
            (w*0.15, h*0.75), # Front wheel well
            (w*0.05, h*0.7),  # Bottom front
            (w*0.05, h*0.45)  # Close shape
        ]
        draw.polygon(body_points, fill=color + (220,))
        
        # Windows
        window_color = (50, 50, 70, 180)
        # Windshield
        draw.polygon([(w*0.35, h*0.25), (w*0.45, h*0.2), (w*0.47, h*0.35), (w*0.37, h*0.38)], 
                    fill=window_color)
        # Side windows
        draw.polygon([(w*0.48, h*0.2), (w*0.62, h*0.2), (w*0.62, h*0.35), (w*0.48, h*0.35)], 
                    fill=window_color)
        # Rear window
        draw.polygon([(w*0.63, h*0.22), (w*0.72, h*0.25), (w*0.7, h*0.35), (w*0.63, h*0.35)], 
                    fill=window_color)
        
        # Wheels
        wheel_color = (30, 30, 30, 255)
        draw.ellipse([w*0.15, h*0.65, w*0.3, h*0.85], fill=wheel_color)
        draw.ellipse([w*0.7, h*0.65, w*0.85, h*0.85], fill=wheel_color)
        
        # Headlights/taillights
        if state and state.get('lights_on', False):
            draw.ellipse([w*0.05, h*0.4, w*0.1, h*0.5], fill=(255, 255, 200, 200))
            draw.ellipse([w*0.9, h*0.4, w*0.95, h*0.5], fill=(255, 50, 50, 200))
    
    def draw_realistic_pedestrian(self, draw, w, h, color, state):
        """Draw a realistic pedestrian with walking pose"""
        # Head
        head_size = h * 0.15
        draw.ellipse([w*0.35, 0, w*0.65, head_size], fill=color + (220,))
        
        # Body
        torso_color = tuple(int(c * 0.8) for c in color) + (220,)
        draw.rectangle([w*0.3, head_size, w*0.7, h*0.6], fill=torso_color)
        
        # Arms
        arm_angle = 0
        if state and state.get('walking', False):
            # Animate arms based on walking cycle
            arm_angle = state.get('walk_phase', 0) * 20
        
        # Left arm
        draw.polygon([(w*0.3, h*0.2), (w*0.25, h*0.2), (w*0.2, h*0.5), (w*0.25, h*0.5)], 
                    fill=torso_color)
        # Right arm
        draw.polygon([(w*0.7, h*0.2), (w*0.75, h*0.2), (w*0.8, h*0.5), (w*0.75, h*0.5)], 
                    fill=torso_color)
        
        # Legs
        leg_color = tuple(int(c * 0.6) for c in color) + (220,)
        if state and state.get('walking', False):
            # Walking pose
            walk_phase = state.get('walk_phase', 0)
            left_offset = np.sin(walk_phase) * w * 0.1
            right_offset = -np.sin(walk_phase) * w * 0.1
            
            # Left leg
            draw.polygon([(w*0.35, h*0.6), (w*0.4, h*0.6), 
                         (w*0.35 + left_offset, h), (w*0.3 + left_offset, h)], 
                        fill=leg_color)
            # Right leg
            draw.polygon([(w*0.6, h*0.6), (w*0.65, h*0.6), 
                         (w*0.7 + right_offset, h), (w*0.65 + right_offset, h)], 
                        fill=leg_color)
        else:
            # Standing pose
            draw.rectangle([w*0.3, h*0.6, w*0.45, h], fill=leg_color)
            draw.rectangle([w*0.55, h*0.6, w*0.7, h], fill=leg_color)
    
    def draw_realistic_cyclist(self, draw, w, h, color, state):
        """Draw a realistic cyclist on bicycle"""
        # Bicycle frame
        frame_color = (100, 100, 100, 220)
        
        # Main triangle
        draw.polygon([(w*0.3, h*0.7), (w*0.5, h*0.5), (w*0.7, h*0.7)], 
                    outline=frame_color, width=3)
        
        # Seat post and handlebars
        draw.line([(w*0.3, h*0.7), (w*0.3, h*0.4)], fill=frame_color, width=3)
        draw.line([(w*0.7, h*0.7), (w*0.7, h*0.45)], fill=frame_color, width=3)
        draw.line([(w*0.65, h*0.45), (w*0.75, h*0.45)], fill=frame_color, width=2)
        
        # Wheels
        wheel_color = (30, 30, 30, 255)
        draw.ellipse([w*0.1, h*0.65, w*0.3, h*0.95], outline=wheel_color, width=3)
        draw.ellipse([w*0.7, h*0.65, w*0.9, h*0.95], outline=wheel_color, width=3)
        
        # Spokes
        for angle in range(0, 360, 45):
            x1 = w*0.2 + w*0.1 * np.cos(np.radians(angle)) * 0.4
            y1 = h*0.8 + h*0.15 * np.sin(np.radians(angle)) * 0.4
            draw.line([(w*0.2, h*0.8), (x1, y1)], fill=wheel_color, width=1)
            
            x2 = w*0.8 + w*0.1 * np.cos(np.radians(angle)) * 0.4
            y2 = h*0.8 + h*0.15 * np.sin(np.radians(angle)) * 0.4
            draw.line([(w*0.8, h*0.8), (x2, y2)], fill=wheel_color, width=1)
        
        # Rider
        # Head
        draw.ellipse([w*0.25, h*0.05, w*0.4, h*0.2], fill=color + (220,))
        
        # Body (leaning forward)
        body_points = [
            (w*0.3, h*0.2),   # Neck
            (w*0.35, h*0.4),  # Shoulders
            (w*0.3, h*0.5),   # Lower back
            (w*0.25, h*0.45), # Chest
            (w*0.28, h*0.2)   # Back to neck
        ]
        draw.polygon(body_points, fill=color + (220,))
        
        # Arms to handlebars
        draw.line([(w*0.32, h*0.25), (w*0.7, h*0.45)], fill=color + (200,), width=4)
        
        # Legs
        leg_color = tuple(int(c * 0.7) for c in color) + (220,)
        if state and state.get('pedaling', True):
            # Animated pedaling
            pedal_phase = state.get('pedal_phase', 0)
            
            # Left leg
            knee_y = h*0.55 + np.sin(pedal_phase) * h*0.05
            draw.polygon([(w*0.3, h*0.5), (w*0.35, h*0.5), 
                         (w*0.4, knee_y), (w*0.35, h*0.7)], 
                        fill=leg_color)
            
            # Right leg
            knee_y2 = h*0.55 - np.sin(pedal_phase) * h*0.05
            draw.polygon([(w*0.3, h*0.5), (w*0.25, h*0.5), 
                         (w*0.2, knee_y2), (w*0.25, h*0.7)], 
                        fill=leg_color)
    
    def generate_traffic_scenario(self, scenario_params=None):
        """Generate a complete traffic scenario with physics simulation"""
        if scenario_params is None:
            # Random scenario
            scenario = random.choice(self.scenario_templates)
            time_condition = random.choice(self.time_conditions)['name']
            weather = random.choice(self.weather_conditions)['name']
            traffic_pattern = random.choice(list(self.scene_db['traffic_patterns'].keys()))
        else:
            scenario = scenario_params.get('scenario', 'straight_road')
            time_condition = scenario_params.get('time', 'day')
            weather = scenario_params.get('weather', 'clear')
            traffic_pattern = scenario_params.get('traffic', 'normal')
        
        # Get traffic density parameters
        pattern = self.scene_db['traffic_patterns'][traffic_pattern]
        
        # Initialize object states for consistent behavior
        object_states = []
        
        # Generate cars
        num_cars = int(random.uniform(0.8, 1.2) * pattern['car_density'] * 10)
        for _ in range(num_cars):
            state = self.generate_object_state(0, scenario, pattern['speed_modifier'])
            object_states.append(state)
        
        # Generate pedestrians
        num_pedestrians = int(random.uniform(0.8, 1.2) * pattern['pedestrian_density'] * 10)
        for _ in range(num_pedestrians):
            state = self.generate_object_state(1, scenario, pattern['speed_modifier'])
            object_states.append(state)
        
        # Generate cyclists
        num_cyclists = int(random.uniform(0.8, 1.2) * pattern['cyclist_density'] * 10)
        for _ in range(num_cyclists):
            state = self.generate_object_state(2, scenario, pattern['speed_modifier'])
            object_states.append(state)
        
        return {
            'scenario': scenario,
            'time_condition': time_condition,
            'weather': weather,
            'traffic_pattern': traffic_pattern,
            'object_states': object_states
        }
    
    def generate_object_state(self, class_id, scenario, speed_modifier):
        """Generate initial state for an object with physics parameters"""
        class_info = self.classes[class_id]
        behaviors = self.scene_db['object_behaviors'][class_info['name']]
        
        # Select behavior based on scenario
        if scenario == 'intersection':
            behavior_weights = {'straight': 0.4, 'turning_left': 0.3, 'turning_right': 0.2, 'stopped': 0.1}
        elif scenario == 'pedestrian_crossing':
            if class_id == 1:  # Pedestrian
                behavior_weights = {'crossing': 0.6, 'walking': 0.3, 'standing': 0.1}
            else:
                behavior_weights = {'straight': 0.5, 'slow': 0.3, 'stopped': 0.2}
        else:
            behavior_weights = {'straight': 0.6, 'slow': 0.3, 'stopped': 0.1}
        
        # Filter valid behaviors
        valid_behaviors = [b for b in behavior_weights.keys() if b in behaviors]
        weights = [behavior_weights.get(b, 0.1) for b in valid_behaviors]
        behavior_name = np.random.choice(valid_behaviors, p=np.array(weights)/sum(weights))
        behavior = behaviors[behavior_name]
        
        # Initial position based on lane preference
        if class_info['lane_preference'] == 'road':
            x = random.uniform(50, 366)
            y = random.uniform(168, 248)
        elif class_info['lane_preference'] == 'sidewalk':
            x = random.uniform(50, 366)
            y = random.choice([random.uniform(100, 168), random.uniform(248, 316)])
        else:  # bike_lane
            x = random.uniform(50, 366)
            y = random.choice([random.uniform(163, 173), random.uniform(243, 253)])
        
        # Velocity with variance
        base_vx, base_vy = behavior['velocity']
        vx = base_vx * speed_modifier + random.uniform(-behavior['variance'], behavior['variance'])
        vy = base_vy * speed_modifier + random.uniform(-behavior['variance'], behavior['variance'])
        
        # Size
        base_size = random.randint(*class_info['size_range'])
        aspect_ratio = random.uniform(*class_info['aspect_ratio'])
        
        return {
            'class_id': class_id,
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'width': int(base_size * aspect_ratio),
            'height': base_size,
            'behavior': behavior_name,
            'rotation': random.uniform(-5, 5) if class_id == 0 else 0,
            'lights_on': random.random() > 0.5,
            'walking': behavior_name in ['walking', 'crossing'],
            'pedaling': behavior_name != 'stopped',
            'walk_phase': random.uniform(0, 2 * np.pi),
            'pedal_phase': random.uniform(0, 2 * np.pi)
        }
    
    def simulate_frame(self, scenario_data, frame_num, delta_time=0.1):
        """Simulate one frame with physics updates"""
        # Update object positions
        for state in scenario_data['object_states']:
            # Physics update
            state['x'] += state['vx'] * delta_time
            state['y'] += state['vy'] * delta_time
            
            # Update animation phases
            if state.get('walking', False):
                state['walk_phase'] += 0.1
            if state.get('pedaling', False):
                state['pedal_phase'] += 0.2
            
            # Boundary checks and wrapping
            if state['x'] < -state['width']:
                state['x'] = self.image_size[0] + state['width']
            elif state['x'] > self.image_size[0] + state['width']:
                state['x'] = -state['width']
            
            if state['y'] < -state['height']:
                state['y'] = self.image_size[1] + state['height']
            elif state['y'] > self.image_size[1] + state['height']:
                state['y'] = -state['height']
        
        # Generate frame
        return self.render_frame(scenario_data)
    
    def render_frame(self, scenario_data):
        """Render a single frame from scenario data"""
        # Create background
        img = self.generate_background(
            scenario_data['scenario'],
            scenario_data['time_condition'],
            scenario_data['weather']
        )
        
        annotations = []
        
        # Sort objects by y-coordinate for proper depth ordering
        sorted_objects = sorted(scenario_data['object_states'], key=lambda obj: obj['y'])
        
        # Render objects
        for obj_state in sorted_objects:
            # Skip objects outside visible area
            if (obj_state['x'] + obj_state['width'] < 0 or 
                obj_state['x'] > self.image_size[0] or
                obj_state['y'] + obj_state['height'] < 0 or 
                obj_state['y'] > self.image_size[1]):
                continue
            
            # Generate object
            obj_img = self.generate_object(obj_state['class_id'], obj_state)
            
            # Paste object
            x = int(obj_state['x'])
            y = int(obj_state['y'])
            img.paste(obj_img, (x, y), obj_img)
            
            # Add annotation
            cx = (x + obj_img.width / 2) / self.image_size[0]
            cy = (y + obj_img.height / 2) / self.image_size[1]
            w = obj_img.width / self.image_size[0]
            h = obj_img.height / self.image_size[1]
            
            # Ensure annotation is within bounds
            if 0 <= cx <= 1 and 0 <= cy <= 1 and w > 0 and h > 0:
                annotations.append({
                    'class_id': obj_state['class_id'],
                    'cx': cx,
                    'cy': cy,
                    'w': w,
                    'h': h,
                    'object_id': id(obj_state),  # For tracking
                    'state': obj_state  # Full state for UI interaction
                })
        
        return img, annotations
    
    def generate_image_with_annotations(self, image_id, split='train'):
        """Generate single image with annotations using scenario simulation"""
        # Generate scenario
        scenario_data = self.generate_traffic_scenario()
        
        # Render frame
        img, annotations = self.render_frame(scenario_data)
        
        return img, annotations, scenario_data
    
    def generate_sequence(self, num_frames=10, scenario_params=None):
        """Generate a sequence of frames for video training"""
        scenario_data = self.generate_traffic_scenario(scenario_params)
        sequence = []
        
        for frame_num in range(num_frames):
            img, annotations = self.simulate_frame(scenario_data, frame_num)
            sequence.append({
                'frame_num': frame_num,
                'image': img,
                'annotations': annotations
            })
        
        return sequence, scenario_data
    
    def export_for_ui_training(self, scenario_data, output_path):
        """Export scenario data for live UI training"""
        # Convert to JSON-serializable format
        export_data = {
            'scenario': scenario_data['scenario'],
            'time_condition': scenario_data['time_condition'],
            'weather': scenario_data['weather'],
            'traffic_pattern': scenario_data['traffic_pattern'],
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'objects': []
        }
        
        for obj in scenario_data['object_states']:
            export_data['objects'].append({
                'class_id': obj['class_id'],
                'class_name': self.classes[obj['class_id']]['name'],
                'initial_position': {'x': obj['x'], 'y': obj['y']},
                'velocity': {'vx': obj['vx'], 'vy': obj['vy']},
                'size': {'width': obj['width'], 'height': obj['height']},
                'behavior': obj['behavior'],
                'properties': {
                    'rotation': obj.get('rotation', 0),
                    'lights_on': obj.get('lights_on', False)
                }
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_data
    
    def generate_dataset(self):
        """Generate complete dataset with realistic scenarios"""
        print(f"Generating synthetic traffic dataset in {self.output_dir}")
        print(f"Seed: {self.seed} (for reproducibility)")
        
        # Generate training set
        print(f"\nGenerating {self.num_train} training images...")
        train_manifest = []
        train_scenarios = []
        
        for i in range(self.num_train):
            # Generate and render scenario
            img, annotations, scenario_data = self.generate_image_with_annotations(i, 'train')
            
            # Save image
            image_path = os.path.join(self.output_dir, 'train', 'images', f'{i:06d}.jpg')
            img.save(image_path, 'JPEG', quality=95)
            
            # Save annotations in YOLO format
            label_path = os.path.join(self.output_dir, 'train', 'labels', f'{i:06d}.txt')
            with open(label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} "
                           f"{ann['w']:.6f} {ann['h']:.6f}\n")
            
            # Save scenario for UI training
            scenario_path = os.path.join(self.output_dir, 'train', 'scenarios', f'{i:06d}.json')
            os.makedirs(os.path.dirname(scenario_path), exist_ok=True)
            self.export_for_ui_training(scenario_data, scenario_path)
            
            train_manifest.append({
                'image': os.path.basename(image_path),
                'annotations': annotations,
                'scenario': scenario_data['scenario'],
                'conditions': {
                    'time': scenario_data['time_condition'],
                    'weather': scenario_data['weather'],
                    'traffic': scenario_data['traffic_pattern']
                }
            })
            train_scenarios.append(scenario_data)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{self.num_train} training images")
        
        # Generate validation set with diverse scenarios
        print(f"\nGenerating {self.num_val} validation images...")
        val_manifest = []
        val_scenarios = []
        
        # Ensure validation has all scenario types
        scenario_cycle = self.scenario_templates * ((self.num_val // len(self.scenario_templates)) + 1)
        
        for i in range(self.num_val):
            # Use diverse scenarios for validation
            scenario_params = {
                'scenario': scenario_cycle[i],
                'time': self.time_conditions[i % len(self.time_conditions)]['name'],
                'weather': self.weather_conditions[i % len(self.weather_conditions)]['name'],
                'traffic': list(self.scene_db['traffic_patterns'].keys())[i % 3]
            }
            
            # Generate scenario with specified parameters
            scenario_data = self.generate_traffic_scenario(scenario_params)
            img, annotations = self.render_frame(scenario_data)
            
            # Save image
            image_path = os.path.join(self.output_dir, 'val', 'images', f'{i:06d}.jpg')
            img.save(image_path, 'JPEG', quality=95)
            
            # Save annotations
            label_path = os.path.join(self.output_dir, 'val', 'labels', f'{i:06d}.txt')
            with open(label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} "
                           f"{ann['w']:.6f} {ann['h']:.6f}\n")
            
            val_manifest.append({
                'image': os.path.basename(image_path),
                'annotations': annotations,
                'scenario': scenario_data['scenario'],
                'conditions': {
                    'time': scenario_data['time_condition'],
                    'weather': scenario_data['weather'],
                    'traffic': scenario_data['traffic_pattern']
                }
            })
            val_scenarios.append(scenario_data)
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{self.num_val} validation images")
        
        # Generate video sequences for temporal training
        print(f"\nGenerating video sequences for temporal training...")
        sequences_dir = os.path.join(self.output_dir, 'sequences')
        os.makedirs(sequences_dir, exist_ok=True)
        
        num_sequences = 10
        frames_per_sequence = 30
        
        for seq_idx in range(num_sequences):
            sequence, scenario_data = self.generate_sequence(frames_per_sequence)
            
            seq_dir = os.path.join(sequences_dir, f'sequence_{seq_idx:03d}')
            os.makedirs(seq_dir, exist_ok=True)
            
            # Save frames and annotations
            for frame_data in sequence:
                frame_path = os.path.join(seq_dir, f'frame_{frame_data["frame_num"]:04d}.jpg')
                frame_data['image'].save(frame_path, 'JPEG', quality=95)
                
                label_path = os.path.join(seq_dir, f'frame_{frame_data["frame_num"]:04d}.txt')
                with open(label_path, 'w') as f:
                    for ann in frame_data['annotations']:
                        f.write(f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} "
                               f"{ann['w']:.6f} {ann['h']:.6f}\n")
            
            # Save sequence metadata
            sequence_info = {
                'num_frames': frames_per_sequence,
                'scenario': scenario_data['scenario'],
                'conditions': {
                    'time': scenario_data['time_condition'],
                    'weather': scenario_data['weather'],
                    'traffic': scenario_data['traffic_pattern']
                },
                'fps': 10  # 10 FPS for training
            }
            
            with open(os.path.join(seq_dir, 'sequence_info.json'), 'w') as f:
                json.dump(sequence_info, f, indent=2)
        
        # Save comprehensive dataset info
        dataset_info = {
            'classes': [self.classes[i]['name'] for i in range(3)],
            'num_classes': 3,
            'image_size': self.image_size,
            'train_samples': self.num_train,
            'val_samples': self.num_val,
            'num_sequences': num_sequences,
            'frames_per_sequence': frames_per_sequence,
            'format': 'YOLO',
            'seed': self.seed,
            'generation_time': datetime.now().isoformat(),
            'scenarios': self.scenario_templates,
            'time_conditions': [t['name'] for t in self.time_conditions],
            'weather_conditions': [w['name'] for w in self.weather_conditions],
            'statistics': {
                'train': self.calculate_dataset_statistics(train_manifest),
                'val': self.calculate_dataset_statistics(val_manifest)
            }
        }
        
        with open(os.path.join(self.output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Total images: {self.num_train + self.num_val}")
        print(f"Video sequences: {num_sequences} ({frames_per_sequence} frames each)")
        print(f"Classes: cars, pedestrians, cyclists")
        print(f"Scenarios: {', '.join(self.scenario_templates)}")
        print(f"Format: YOLO (normalized xywh) with scenario metadata")
    
    def calculate_dataset_statistics(self, manifest):
        """Calculate dataset statistics for analysis"""
        class_counts = {0: 0, 1: 0, 2: 0}
        scenario_counts = {}
        total_objects = 0
        
        for item in manifest:
            for ann in item['annotations']:
                class_counts[ann['class_id']] += 1
                total_objects += 1
            
            scenario = item['scenario']
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        return {
            'total_objects': total_objects,
            'class_distribution': {
                'cars': class_counts[0],
                'pedestrians': class_counts[1],
                'cyclists': class_counts[2]
            },
            'scenario_distribution': scenario_counts,
            'avg_objects_per_image': total_objects / len(manifest) if manifest else 0
        }


def main():
    parser = argparse.ArgumentParser(description='Generate realistic synthetic traffic dataset')
    parser.add_argument('--output-dir', default='./datasets/traffic_detection/',
                       help='Output directory for dataset')
    parser.add_argument('--num-train', type=int, default=1000,
                       help='Number of training images')
    parser.add_argument('--num-val', type=int, default=200,
                       help='Number of validation images')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: auto from output dir)')
    parser.add_argument('--generate-ui-data', action='store_true',
                       help='Generate additional data for UI training')
    
    args = parser.parse_args()
    
    generator = SyntheticTrafficGenerator(
        args.output_dir,
        args.num_train,
        args.num_val,
        args.seed
    )
    
    # Generate main dataset
    generator.generate_dataset()
    
    # Optionally generate UI training examples
    if args.generate_ui_data:
        print("\nGenerating UI training examples...")
        ui_dir = os.path.join(args.output_dir, 'ui_training')
        os.makedirs(ui_dir, exist_ok=True)
        
        # Generate interactive scenarios
        for i in range(5):
            scenario_params = {
                'scenario': generator.scenario_templates[i % len(generator.scenario_templates)],
                'time': 'day',
                'weather': 'clear',
                'traffic': 'normal'
            }
            
            # Generate 10-second sequence at 10 FPS
            sequence, scenario_data = generator.generate_sequence(100, scenario_params)
            
            # Save sequence frames for UI preview
            seq_preview_dir = os.path.join(ui_dir, f'scenario_{i:02d}_frames')
            os.makedirs(seq_preview_dir, exist_ok=True)
            
            # Save first, middle, and last frames as preview
            preview_indices = [0, len(sequence)//2, len(sequence)-1]
            for idx in preview_indices:
                frame = sequence[idx]
                frame_path = os.path.join(seq_preview_dir, f'preview_{idx:03d}.jpg')
                frame['image'].save(frame_path, 'JPEG', quality=90)
            
            # Export scenario data for UI
            ui_scenario_path = os.path.join(ui_dir, f'scenario_{i:02d}.json')
            generator.export_for_ui_training(scenario_data, ui_scenario_path)
            
            print(f"  Generated UI scenario {i+1}/5")
        
        print("UI training data generated!")


if __name__ == '__main__':
    main()