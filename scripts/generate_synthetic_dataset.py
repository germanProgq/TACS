#!/usr/bin/env python3
"""
Generate synthetic 3D traffic dataset for TACSNet training.
Creates realistic annotated images with 3D models, multiple viewpoints,
and realistic street scenes from various camera angles.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import argparse
import hashlib
from datetime import datetime
import math

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
        
        # 3D camera parameters for multiple viewpoints - focused on capturing objects
        self.camera_configs = [
            {
                'name': 'intersection_high',
                'position': (0, -15, 8),  # x, y, z in meters - closer and lower
                'rotation': (-55, 0, 0),  # pitch, yaw, roll in degrees - look down sharply at street
                'fov': 90,  # wider fov to capture street level
                'description': 'High angle intersection view looking at street'
            },
            {
                'name': 'street_level',
                'position': (10, -10, 2.5),  # street corner position
                'rotation': (-35, 45, 0),  # angle down towards street center
                'fov': 85,
                'description': 'Street level corner view of traffic'
            },
            {
                'name': 'traffic_cam',
                'position': (0, -18, 6),  # standard traffic cam position
                'rotation': (-50, 0, 0),  # look down at street traffic
                'fov': 80,
                'description': 'Standard traffic camera angle focused on road'
            },
            {
                'name': 'side_view',
                'position': (15, 0, 4),  # side position looking across road
                'rotation': (-45, -90, 0),  # look down and across road
                'fov': 85,
                'description': 'Side street monitoring of road traffic'
            },
            {
                'name': 'diagonal_high',
                'position': (8, -8, 7),  # diagonal corner position
                'rotation': (-60, 45, 0),  # look down sharply at street intersection
                'fov': 80,
                'description': 'Diagonal overhead view of intersection'
            },
            {
                'name': 'low_angle',
                'position': (6, -12, 1.8),  # low mounted street camera
                'rotation': (-25, 30, 0),  # look slightly down at road
                'fov': 95,  # very wide fov for street level coverage
                'description': 'Low mounted camera capturing road activity'
            }
        ]
        
        # 3D world parameters
        self.world_scale = 50.0  # meters
        self.road_width = 7.0    # meters per lane
        self.sidewalk_width = 2.5  # meters
        self.building_height_range = (10, 30)  # meters
        
        # Class definitions with 3D properties
        self.classes = {
            0: {
                'name': 'car',
                'color_variants': [(200, 50, 50), (50, 50, 200), (150, 150, 150), (50, 50, 50)],
                'size_3d': (4.5, 1.5, 1.8),  # length, height, width in meters
                'speed_range': (8.0, 15.0),  # m/s (30-55 km/h)
                'lane_preference': 'road',
                'model_type': 'sedan'  # sedan, suv, truck, bus
            },
            1: {
                'name': 'pedestrian', 
                'color_variants': [(100, 180, 100), (180, 100, 100), (100, 100, 180)],
                'size_3d': (0.5, 1.7, 0.4),  # depth, height, width in meters
                'speed_range': (0.8, 1.5),  # m/s (walking speed)
                'lane_preference': 'sidewalk',
                'model_type': 'person'
            },
            2: {
                'name': 'cyclist',
                'color_variants': [(50, 100, 200), (200, 100, 50), (100, 200, 50)],
                'size_3d': (1.8, 1.8, 0.6),  # length, height, width in meters
                'speed_range': (3.0, 7.0),  # m/s (10-25 km/h)
                'lane_preference': 'bike_lane',
                'model_type': 'bicycle'
            }
        }
        
        # 3D model variants for diversity
        self.model_variants = {
            'car': {
                'sedan': {'size_scale': 1.0, 'height_scale': 1.0},
                'suv': {'size_scale': 1.1, 'height_scale': 1.3},
                'truck': {'size_scale': 1.5, 'height_scale': 1.4},
                'bus': {'size_scale': 2.5, 'height_scale': 1.8}
            },
            'pedestrian': {
                'adult': {'size_scale': 1.0, 'height_scale': 1.0},
                'child': {'size_scale': 0.7, 'height_scale': 0.7},
                'tall': {'size_scale': 1.0, 'height_scale': 1.1}
            },
            'cyclist': {
                'regular': {'size_scale': 1.0, 'height_scale': 1.0},
                'sport': {'size_scale': 0.9, 'height_scale': 0.95},
                'cargo': {'size_scale': 1.3, 'height_scale': 1.1}
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
    
    def project_3d_to_2d(self, point_3d, camera):
        """Project 3D world coordinates to 2D image coordinates"""
        x, y, z = point_3d
        cam_x, cam_y, cam_z = camera['position']
        pitch, yaw, roll = [np.radians(angle) for angle in camera['rotation']]
        
        # Translate to camera coordinates
        dx = x - cam_x
        dy = y - cam_y
        dz = z - cam_z
        
        # Apply camera rotations (view matrix)
        # Camera looks down Y axis in world space
        # First apply yaw (rotation around Z axis)
        cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
        x1 = dx * cos_yaw - dy * sin_yaw
        y1 = dx * sin_yaw + dy * cos_yaw
        z1 = dz
        
        # Then apply pitch (rotation around X axis)
        cos_pitch, sin_pitch = np.cos(-pitch), np.sin(-pitch)
        x2 = x1
        y2 = y1 * cos_pitch - z1 * sin_pitch
        z2 = y1 * sin_pitch + z1 * cos_pitch
        
        # Camera space: +Z forward, +X right, +Y up
        # Convert from world to camera coordinates
        cam_x = x2
        cam_y = -z2  # Up is negative Z in world
        cam_z = y2   # Forward is Y in world
        
        # Check if point is in front of camera
        if cam_z <= 0.1:  # Behind camera
            return None
            
        # Perspective projection
        fov_rad = np.radians(camera['fov'])
        f = self.image_size[0] / (2 * np.tan(fov_rad / 2))
        
        # Project to image plane
        x_2d = f * cam_x / cam_z + self.image_size[0] / 2
        y_2d = f * cam_y / cam_z + self.image_size[1] / 2
        
        # Check if within image bounds with margin
        margin = 50  # Allow some margin for partial objects
        if -margin <= x_2d < self.image_size[0] + margin and -margin <= y_2d < self.image_size[1] + margin:
            return (int(x_2d), int(y_2d))
        return None
    
    def get_3d_bounding_box(self, position, size, rotation):
        """Get 8 corners of 3D bounding box"""
        x, y, z = position
        length, height, width = size
        yaw = np.radians(rotation)
        
        # Define corners in object space
        corners = [
            (-length/2, -width/2, 0),      # Front bottom left
            (length/2, -width/2, 0),       # Front bottom right
            (length/2, width/2, 0),        # Rear bottom right
            (-length/2, width/2, 0),       # Rear bottom left
            (-length/2, -width/2, height), # Front top left
            (length/2, -width/2, height),  # Front top right
            (length/2, width/2, height),   # Rear top right
            (-length/2, width/2, height),  # Rear top left
        ]
        
        # Rotate and translate to world space
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        world_corners = []
        
        for cx, cy, cz in corners:
            # Rotate around Z axis
            rx = cx * cos_yaw - cy * sin_yaw
            ry = cx * sin_yaw + cy * cos_yaw
            # Translate
            world_corners.append((x + rx, y + ry, z + cz))
        
        return world_corners
    
    def initialize_scene_database(self):
        """Initialize persistent scene elements for consistency"""
        self.scene_db = {
            'road_layouts': self.generate_road_layouts(),
            'traffic_patterns': self.generate_traffic_patterns(),
            'object_behaviors': self.generate_object_behaviors()
        }
    
    def generate_road_layouts(self):
        """Generate 3D road layout templates"""
        layouts = {}
        
        # 3D Intersection layout (coordinates in meters)
        layouts['intersection'] = {
            'roads': [
                {'start': (-25, 0, 0), 'end': (25, 0, 0), 'width': self.road_width * 2, 'lanes': 4},
                {'start': (0, -25, 0), 'end': (0, 25, 0), 'width': self.road_width * 2, 'lanes': 4}
            ],
            'sidewalks': [
                {'start': (-25, -self.road_width - self.sidewalk_width, 0), 
                 'end': (25, -self.road_width - self.sidewalk_width, 0), 
                 'width': self.sidewalk_width, 'height': 0.15},
                {'start': (-25, self.road_width + self.sidewalk_width, 0), 
                 'end': (25, self.road_width + self.sidewalk_width, 0), 
                 'width': self.sidewalk_width, 'height': 0.15}
            ],
            'buildings': [
                {'position': (-15, -15, 0), 'size': (10, 10, 20), 'type': 'office'},
                {'position': (15, -15, 0), 'size': (12, 8, 15), 'type': 'residential'},
                {'position': (-15, 15, 0), 'size': (8, 12, 25), 'type': 'commercial'},
                {'position': (15, 15, 0), 'size': (10, 10, 18), 'type': 'mixed'}
            ],
            'traffic_lights': [
                {'position': (-5, -5, 5), 'direction': 'north'},
                {'position': (5, 5, 5), 'direction': 'south'},
                {'position': (-5, 5, 5), 'direction': 'east'},
                {'position': (5, -5, 5), 'direction': 'west'}
            ]
        }
        
        # Straight road layout
        layouts['straight_road'] = {
            'roads': [
                {'start': (-30, 0, 0), 'end': (30, 0, 0), 'width': self.road_width * 3, 'lanes': 6}
            ],
            'sidewalks': [
                {'start': (-30, -self.road_width * 1.5 - self.sidewalk_width, 0), 
                 'end': (30, -self.road_width * 1.5 - self.sidewalk_width, 0), 
                 'width': self.sidewalk_width, 'height': 0.15},
                {'start': (-30, self.road_width * 1.5 + self.sidewalk_width, 0), 
                 'end': (30, self.road_width * 1.5 + self.sidewalk_width, 0), 
                 'width': self.sidewalk_width, 'height': 0.15}
            ],
            'buildings': [
                {'position': (-20, -20, 0), 'size': (15, 10, 22), 'type': 'office'},
                {'position': (10, -18, 0), 'size': (20, 8, 18), 'type': 'commercial'},
                {'position': (-15, 18, 0), 'size': (25, 10, 16), 'type': 'residential'},
                {'position': (20, 20, 0), 'size': (10, 15, 24), 'type': 'mixed'}
            ],
            'street_furniture': [
                {'position': (-10, -12, 0), 'type': 'lamp_post', 'height': 6},
                {'position': (0, -12, 0), 'type': 'lamp_post', 'height': 6},
                {'position': (10, -12, 0), 'type': 'lamp_post', 'height': 6},
                {'position': (-5, 12, 0), 'type': 'bus_stop', 'size': (3, 2, 2.5)},
                {'position': (15, 12, 0), 'type': 'bench', 'size': (2, 0.5, 0.8)}
            ]
        }
        
        # Parking area layout
        layouts['parking_area'] = {
            'roads': [
                {'start': (-25, -5, 0), 'end': (25, -5, 0), 'width': self.road_width, 'lanes': 2}
            ],
            'sidewalks': [
                {'start': (-25, -5 - self.sidewalk_width, 0), 
                 'end': (25, -5 - self.sidewalk_width, 0), 
                 'width': self.sidewalk_width, 'height': 0.15}
            ],
            'buildings': [
                {'position': (0, 15, 0), 'size': (40, 20, 15), 'type': 'commercial'}
            ],
            'parking_spaces': [
                {'position': (i * 3, 5, 0), 'size': (2.5, 5, 0), 'occupied': random.random() > 0.3}
                for i in range(-8, 9)
            ]
        }
        
        # Bike lane layout
        layouts['bike_lane'] = layouts['straight_road'].copy()  # Similar to straight road
        
        # Pedestrian crossing layout
        layouts['pedestrian_crossing'] = {
            'roads': [
                {'start': (-20, 0, 0), 'end': (20, 0, 0), 'width': self.road_width * 2, 'lanes': 4}
            ],
            'sidewalks': [
                {'start': (-20, -self.road_width - self.sidewalk_width, 0), 
                 'end': (20, -self.road_width - self.sidewalk_width, 0), 
                 'width': self.sidewalk_width, 'height': 0.15},
                {'start': (-20, self.road_width + self.sidewalk_width, 0), 
                 'end': (20, self.road_width + self.sidewalk_width, 0), 
                 'width': self.sidewalk_width, 'height': 0.15}
            ],
            'crosswalks': [
                {'position': (0, 0, 0), 'width': 3, 'length': self.road_width * 2, 'stripes': 10}
            ],
            'buildings': [
                {'position': (-10, -15, 0), 'size': (15, 10, 20), 'type': 'office'},
                {'position': (10, 15, 0), 'size': (15, 10, 18), 'type': 'residential'}
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
        """Generate 3D object movement behaviors"""
        behaviors = {
            'car': {
                'straight': {'velocity': (10.0, 0, 0), 'variance': 2.0},
                'turning_left': {'velocity': (5.0, -5.0, 0), 'variance': 1.0},
                'turning_right': {'velocity': (5.0, 5.0, 0), 'variance': 1.0},
                'stopped': {'velocity': (0, 0, 0), 'variance': 0.0}
            },
            'pedestrian': {
                'walking': {'velocity': (1.2, 0, 0), 'variance': 0.3},
                'crossing': {'velocity': (0, 1.2, 0), 'variance': 0.2},
                'standing': {'velocity': (0, 0, 0), 'variance': 0.0}
            },
            'cyclist': {
                'riding': {'velocity': (5.0, 0, 0), 'variance': 1.0},
                'slow': {'velocity': (2.5, 0, 0), 'variance': 0.5},
                'stopped': {'velocity': (0, 0, 0), 'variance': 0.0}
            }
        }
        return behaviors
    
    def render_3d_scene(self, scenario, camera, time_condition, weather):
        """Render 3D scene from specified camera viewpoint"""
        # Create base image with sky gradient
        img = self.create_sky_background(time_condition, weather)
        draw = ImageDraw.Draw(img)
        
        # Get scene layout
        layout = self.scene_db['road_layouts'].get(scenario, self.scene_db['road_layouts']['straight_road'])
        
        # Render ground plane first
        self.render_ground_plane(img, draw, camera)
        
        # Render scene elements in order (back to front)
        self.render_buildings(img, draw, layout.get('buildings', []), camera)
        self.render_roads_3d(img, draw, layout.get('roads', []), camera)
        self.render_sidewalks_3d(img, draw, layout.get('sidewalks', []), camera)
        self.render_street_furniture(img, draw, layout.get('street_furniture', []), camera)
        self.render_traffic_lights(img, draw, layout.get('traffic_lights', []), camera)
        
        # Apply atmospheric effects
        img = self.apply_atmospheric_effects(img, weather, camera['position'][2])
        
        return img
    
    def render_ground_plane(self, img, draw, camera):
        """Render ground plane to establish scene depth"""
        # Create ground plane grid
        ground_color = (100, 100, 100)  # Gray ground
        grid_size = 5  # 5 meter grid
        
        # Define ground plane corners in world space
        ground_corners = []
        for x in range(-30, 31, grid_size):
            for y in range(-30, 31, grid_size):
                ground_corners.append([
                    (x, y, 0),
                    (x + grid_size, y, 0),
                    (x + grid_size, y + grid_size, 0),
                    (x, y + grid_size, 0)
                ])
        
        # Project and draw ground squares
        for square in ground_corners:
            projected = []
            for corner in square:
                p2d = self.project_3d_to_2d(corner, camera)
                if p2d:
                    projected.append(p2d)
            
            if len(projected) == 4:
                # Draw ground square with slight variation
                color_var = random.randint(-10, 10)
                square_color = tuple(max(0, min(255, c + color_var)) for c in ground_color)
                draw.polygon(projected, fill=square_color, outline=(90, 90, 90))
    
    def render_buildings(self, img, draw, buildings, camera):
        """Render 3D buildings with perspective"""
        for building in buildings:
            pos = building['position']
            size = building['size']
            
            # Get building corners
            corners = [
                (pos[0] - size[0]/2, pos[1] - size[1]/2, pos[2]),
                (pos[0] + size[0]/2, pos[1] - size[1]/2, pos[2]),
                (pos[0] + size[0]/2, pos[1] + size[1]/2, pos[2]),
                (pos[0] - size[0]/2, pos[1] + size[1]/2, pos[2]),
                (pos[0] - size[0]/2, pos[1] - size[1]/2, pos[2] + size[2]),
                (pos[0] + size[0]/2, pos[1] - size[1]/2, pos[2] + size[2]),
                (pos[0] + size[0]/2, pos[1] + size[1]/2, pos[2] + size[2]),
                (pos[0] - size[0]/2, pos[1] + size[1]/2, pos[2] + size[2])
            ]
            
            # Project to 2D
            projected = []
            for corner in corners:
                p2d = self.project_3d_to_2d(corner, camera)
                if p2d:
                    projected.append(p2d)
            
            if len(projected) >= 4:
                # Draw building faces
                building_color = (120, 120, 140) if building['type'] == 'office' else (160, 140, 120)
                
                # Draw visible faces based on camera angle
                faces = [
                    [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],  # Sides
                    [4, 5, 6, 7]  # Top
                ]
                
                for face in faces:
                    if all(i < len(projected) for i in face):
                        points = [projected[i] for i in face]
                        draw.polygon(points, fill=building_color, outline=(80, 80, 80))
    
    def render_roads_3d(self, img, draw, roads, camera):
        """Render 3D roads with lane markings"""
        for road in roads:
            start, end = road['start'], road['end']
            width = road['width']
            
            # Create road surface points
            road_points = []
            num_segments = 20
            
            for i in range(num_segments + 1):
                t = i / num_segments
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                
                # Road edges
                left = (x - width/2, y, 0)
                right = (x + width/2, y, 0)
                
                left_2d = self.project_3d_to_2d(left, camera)
                right_2d = self.project_3d_to_2d(right, camera)
                
                if left_2d and right_2d:
                    road_points.append((left_2d, right_2d))
            
            # Draw road segments
            for i in range(len(road_points) - 1):
                points = [
                    road_points[i][0], road_points[i][1],
                    road_points[i+1][1], road_points[i+1][0]
                ]
                draw.polygon(points, fill=(60, 60, 60))
            
            # Draw lane markings
            if 'lanes' in road:
                for lane in range(1, road['lanes']):
                    lane_offset = (lane / road['lanes'] - 0.5) * width
                    
                    for i in range(0, num_segments, 2):
                        t = i / num_segments
                        x = start[0] + t * (end[0] - start[0])
                        y = start[1] + t * (end[1] - start[1]) + lane_offset
                        
                        p1 = self.project_3d_to_2d((x, y, 0), camera)
                        p2 = self.project_3d_to_2d((x + (end[0]-start[0])/num_segments, y, 0), camera)
                        
                        if p1 and p2:
                            draw.line([p1, p2], fill=(200, 200, 200), width=2)
    
    def render_sidewalks_3d(self, img, draw, sidewalks, camera):
        """Render 3D sidewalks"""
        for sidewalk in sidewalks:
            # Similar to roads but with different color and texture
            start, end = sidewalk['start'], sidewalk['end']
            width = sidewalk['width']
            
            # Create sidewalk points
            corners = [
                self.project_3d_to_2d((start[0], start[1], 0), camera),
                self.project_3d_to_2d((end[0], end[1], 0), camera),
                self.project_3d_to_2d((end[0], end[1] + width, 0), camera),
                self.project_3d_to_2d((start[0], start[1] + width, 0), camera)
            ]
            
            if all(corners):
                draw.polygon(corners, fill=(140, 140, 140), outline=(120, 120, 120))
    
    def render_street_furniture(self, img, draw, furniture, camera):
        """Render street furniture like lamp posts and benches"""
        for item in furniture:
            pos = item['position']
            
            if item['type'] == 'lamp_post':
                # Draw lamp post
                base = self.project_3d_to_2d(pos, camera)
                top = self.project_3d_to_2d((pos[0], pos[1], pos[2] + item['height']), camera)
                
                if base and top:
                    draw.line([base, top], fill=(80, 80, 80), width=3)
                    # Lamp
                    draw.ellipse([top[0]-5, top[1]-5, top[0]+5, top[1]+5], 
                               fill=(255, 255, 200))
            
            elif item['type'] in ['bus_stop', 'bench']:
                # Simplified furniture rendering
                size = item.get('size', (2, 1, 1))
                corners = self.get_3d_bounding_box(pos, size, 0)
                
                projected = []
                for corner in corners[:4]:  # Just bottom face
                    p2d = self.project_3d_to_2d(corner, camera)
                    if p2d:
                        projected.append(p2d)
                
                if len(projected) >= 4:
                    draw.polygon(projected, fill=(100, 80, 60))
    
    def render_traffic_lights(self, img, draw, traffic_lights, camera):
        """Render traffic lights"""
        for light in traffic_lights:
            pos = light['position']
            p2d = self.project_3d_to_2d(pos, camera)
            
            if p2d:
                # Draw traffic light pole and lights
                draw.rectangle([p2d[0]-3, p2d[1]-10, p2d[0]+3, p2d[1]+10], 
                             fill=(40, 40, 40))
                
                # Draw light states (simplified)
                colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0)]
                for i, color in enumerate(colors):
                    draw.ellipse([p2d[0]-3, p2d[1]-8+i*6, p2d[0]+3, p2d[1]-2+i*6], 
                               fill=color if i == 2 else (50, 50, 50))
    
    def render_3d_object(self, img, obj_state, bbox_2d, camera):
        """Render a 3D object based on its projected bounding box"""
        draw = ImageDraw.Draw(img)
        class_info = self.classes[obj_state['class_id']]
        color = obj_state.get('color', random.choice(class_info['color_variants']))
        
        if class_info['name'] == 'car':
            self.render_3d_car(draw, bbox_2d, color, obj_state)
        elif class_info['name'] == 'pedestrian':
            self.render_3d_pedestrian(draw, bbox_2d, color, obj_state)
        elif class_info['name'] == 'cyclist':
            self.render_3d_cyclist(draw, bbox_2d, color, obj_state)
    
    def render_3d_car(self, draw, bbox_2d, color, state):
        """Render a 3D car from bounding box points"""
        if len(bbox_2d) >= 8:
            # Draw car body faces
            # Bottom face
            draw.polygon(bbox_2d[:4], fill=tuple(int(c*0.7) for c in color))
            # Top face
            draw.polygon(bbox_2d[4:8], fill=color)
            # Side faces
            for i in range(4):
                j = (i + 1) % 4
                face = [bbox_2d[i], bbox_2d[j], bbox_2d[j+4], bbox_2d[i+4]]
                draw.polygon(face, fill=tuple(int(c*0.85) for c in color))
            
            # Windows (simplified)
            window_color = (50, 50, 70, 180)
            if len(bbox_2d) >= 8:
                # Windshield
                draw.polygon([bbox_2d[4], bbox_2d[5], 
                            tuple(int(0.7*bbox_2d[5][i] + 0.3*bbox_2d[6][i]) for i in range(2)),
                            tuple(int(0.7*bbox_2d[4][i] + 0.3*bbox_2d[7][i]) for i in range(2))],
                           fill=window_color)
    
    def render_3d_pedestrian(self, draw, bbox_2d, color, state):
        """Render a 3D pedestrian"""
        if len(bbox_2d) >= 4:
            # Simplified pedestrian as vertical rectangle
            center_x = sum(p[0] for p in bbox_2d[:4]) // 4
            bottom_y = max(p[1] for p in bbox_2d[:4])
            top_y = min(p[1] for p in bbox_2d[4:]) if len(bbox_2d) >= 8 else bottom_y - 40
            
            # Ensure proper coordinate order
            if top_y > bottom_y:
                top_y, bottom_y = bottom_y, top_y
            
            width = max(5, abs(bbox_2d[1][0] - bbox_2d[0][0]) // 3)
            
            # Body
            body_top = min(top_y + 10, bottom_y - 5)
            if body_top < bottom_y:
                draw.rectangle([center_x - width//2, body_top, 
                              center_x + width//2, bottom_y],
                              fill=color)
            
            # Head
            head_size = width // 1.5
            if top_y + head_size < body_top:
                draw.ellipse([center_x - width//3, top_y,
                             center_x + width//3, top_y + head_size],
                            fill=tuple(min(255, int(c*1.2)) for c in color[:3]))
    
    def render_3d_cyclist(self, draw, bbox_2d, color, state):
        """Render a 3D cyclist"""
        if len(bbox_2d) >= 4:
            # Draw bicycle frame
            center_x = sum(p[0] for p in bbox_2d[:4]) // 4
            center_y = sum(p[1] for p in bbox_2d[:4]) // 4
            
            # Wheels
            wheel_size = abs(bbox_2d[1][0] - bbox_2d[0][0]) // 4
            draw.ellipse([bbox_2d[0][0], center_y - wheel_size,
                         bbox_2d[0][0] + wheel_size*2, center_y + wheel_size],
                        outline=(30, 30, 30), width=3)
            draw.ellipse([bbox_2d[1][0] - wheel_size*2, center_y - wheel_size,
                         bbox_2d[1][0], center_y + wheel_size],
                        outline=(30, 30, 30), width=3)
            
            # Frame
            draw.line([bbox_2d[0][0] + wheel_size, center_y,
                      bbox_2d[1][0] - wheel_size, center_y],
                     fill=(100, 100, 100), width=2)
            
            # Rider (simplified)
            if len(bbox_2d) >= 8:
                rider_top = min(p[1] for p in bbox_2d[4:])
                draw.ellipse([center_x - wheel_size//2, rider_top,
                            center_x + wheel_size//2, rider_top + wheel_size],
                            fill=color)
    
    def apply_atmospheric_effects(self, img, weather, camera_height):
        """Apply weather and atmospheric effects"""
        if weather == 'fog':
            # Add fog effect that's stronger at distance
            fog_layer = Image.new('RGBA', self.image_size, (200, 200, 200, 100))
            img = Image.alpha_composite(img.convert('RGBA'), fog_layer).convert('RGB')
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        elif weather == 'rain':
            # Add rain streaks
            draw = ImageDraw.Draw(img)
            for _ in range(200):
                x = random.randint(0, self.image_size[0])
                y = random.randint(0, self.image_size[1])
                draw.line([x, y, x+1, y+8], fill=(150, 150, 180), width=1)
        
        return img
    
    def create_sky_background(self, time_condition, weather):
        """Create realistic sky gradient based on time and weather"""
        img = Image.new('RGB', self.image_size)
        draw = ImageDraw.Draw(img)
        
        # Sky colors based on time of day
        sky_colors = {
            'day': {'top': (135, 206, 235), 'bottom': (255, 255, 255)},
            'dusk': {'top': (255, 94, 77), 'bottom': (255, 154, 0)},
            'night': {'top': (25, 25, 112), 'bottom': (72, 61, 139)},
            'dawn': {'top': (255, 191, 0), 'bottom': (255, 229, 180)}
        }
        
        colors = sky_colors.get(time_condition, sky_colors['day'])
        
        # Apply weather modifications
        if weather == 'rain':
            colors = {'top': tuple(int(c * 0.6) for c in colors['top']),
                     'bottom': tuple(int(c * 0.7) for c in colors['bottom'])}
        elif weather == 'fog':
            colors = {'top': (200, 200, 200), 'bottom': (220, 220, 220)}
        
        # Create gradient
        for y in range(self.image_size[1]):
            ratio = y / self.image_size[1]
            r = int(colors['top'][0] * (1 - ratio) + colors['bottom'][0] * ratio)
            g = int(colors['top'][1] * (1 - ratio) + colors['bottom'][1] * ratio)
            b = int(colors['top'][2] * (1 - ratio) + colors['bottom'][2] * ratio)
            draw.rectangle([(0, y), (self.image_size[0], y + 1)], fill=(r, g, b))
        
        return img
    
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
        """Generate initial 3D state for an object with physics parameters"""
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
        
        # Handle case where no valid behaviors found
        if not valid_behaviors:
            # Use default behavior based on class
            if class_id == 0:  # Car
                valid_behaviors = ['straight']
            elif class_id == 1:  # Pedestrian
                valid_behaviors = ['walking']
            else:  # Cyclist
                valid_behaviors = ['riding']
            weights = [1.0]
        
        behavior_name = np.random.choice(valid_behaviors, p=np.array(weights)/sum(weights))
        behavior = behaviors[behavior_name]
        
        # Initial 3D position in visible area based on lane preference
        # Place objects in front of cameras (positive Y direction)
        if class_info['lane_preference'] == 'road':
            # Place on road lanes - cars drive along roads
            x = random.uniform(-8, 8)    # Across the road width
            y = random.uniform(0, 15)    # In front of camera (positive Y)
            z = 0  # Ground level
        elif class_info['lane_preference'] == 'sidewalk':
            # Place on sidewalks - pedestrians walk on sides
            x = random.uniform(-10, 10)
            y = random.uniform(0, 12)    # In front of camera
            # Choose sidewalk side
            if x > 0:
                x = random.uniform(6, 10)   # Right sidewalk
            else:
                x = random.uniform(-10, -6) # Left sidewalk  
            z = 0.15  # Sidewalk height
        else:  # bike_lane
            # Place in bike lanes - cyclists on road edges
            x = random.uniform(-6, 6)
            y = random.uniform(2, 10)    # In front of camera
            # Bias towards road edges
            if random.random() > 0.5:
                x = random.uniform(3, 6)    # Right bike lane
            else:
                x = random.uniform(-6, -3)  # Left bike lane
            z = 0
        
        # 3D velocity with variance
        base_vx, base_vy, base_vz = behavior['velocity']
        vx = base_vx * speed_modifier + random.uniform(-behavior['variance'], behavior['variance'])
        vy = base_vy * speed_modifier + random.uniform(-behavior['variance'], behavior['variance'])
        vz = base_vz  # Usually 0
        
        # Select model variant
        model_variants = list(self.model_variants[class_info['name']].keys())
        model_variant = random.choice(model_variants)
        
        # Get size from 3D specifications
        size_3d = class_info['size_3d']
        variant_scale = self.model_variants[class_info['name']][model_variant]
        
        return {
            'class_id': class_id,
            'x': x,
            'y': y,
            'z': z,
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'size_3d': size_3d,
            'model_variant': model_variant,
            'behavior': behavior_name,
            'rotation': random.uniform(-180, 180) if class_id == 0 else random.uniform(-45, 45),
            'color': random.choice(class_info['color_variants']),
            'lights_on': random.random() > 0.5,
            'animation_state': 'moving' if abs(vx) + abs(vy) > 0.1 else 'idle',
            'walking': behavior_name in ['walking', 'crossing'],
            'pedaling': behavior_name != 'stopped',
            'walk_phase': random.uniform(0, 2 * np.pi),
            'pedal_phase': random.uniform(0, 2 * np.pi)
        }
    
    def simulate_frame(self, scenario_data, frame_num, delta_time=0.1):
        """Simulate one frame with 3D physics updates"""
        # Update object positions
        for state in scenario_data['object_states']:
            # 3D Physics update
            state['x'] += state['vx'] * delta_time
            state['y'] += state['vy'] * delta_time
            state['z'] = state.get('z', 0) + state.get('vz', 0) * delta_time
            
            # Update animation phases
            if state.get('walking', False):
                state['walk_phase'] += 0.1
            if state.get('pedaling', False):
                state['pedal_phase'] += 0.2
            
            # Boundary checks and wrapping in 3D world
            world_bounds = self.world_scale / 2
            if state['x'] < -world_bounds:
                state['x'] = world_bounds
            elif state['x'] > world_bounds:
                state['x'] = -world_bounds
            
            if state['y'] < -world_bounds:
                state['y'] = world_bounds
            elif state['y'] > world_bounds:
                state['y'] = -world_bounds
        
        # Select camera for this frame (can vary for dynamic views)
        if 'camera' not in scenario_data:
            scenario_data['camera'] = random.choice(self.camera_configs)
        
        # Generate 3D frame
        return self.render_3d_frame(scenario_data, scenario_data['camera'])
    
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
        """Generate single image with annotations using 3D scenario simulation"""
        # Generate scenario
        scenario_data = self.generate_traffic_scenario()
        
        # Select random camera viewpoint
        camera = random.choice(self.camera_configs)
        scenario_data['camera'] = camera
        
        # Render 3D scene from camera viewpoint
        img, annotations = self.render_3d_frame(scenario_data, camera)
        
        return img, annotations, scenario_data
    
    def render_3d_frame(self, scenario_data, camera):
        """Render a frame from 3D scene with proper perspective"""
        # Create 3D scene background
        img = self.render_3d_scene(
            scenario_data['scenario'],
            camera,
            scenario_data['time_condition'],
            scenario_data['weather']
        )
        
        annotations = []
        
        # Sort objects by distance from camera for proper occlusion
        camera_pos = np.array(camera['position'])
        sorted_objects = sorted(scenario_data['object_states'], 
                              key=lambda obj: -np.linalg.norm(
                                  np.array([obj['x'], obj['y'], obj.get('z', 0)]) - camera_pos
                              ))
        
        # Render objects in 3D
        for obj_state in sorted_objects:
            # Get 3D bounding box
            position_3d = (obj_state['x'], obj_state['y'], obj_state.get('z', 0))
            class_info = self.classes[obj_state['class_id']]
            size_3d = class_info['size_3d']
            
            # Apply model variant scaling
            if 'model_variant' in obj_state:
                variant = self.model_variants[class_info['name']].get(obj_state['model_variant'], {})
                size_3d = tuple(s * variant.get('size_scale', 1.0) for s in size_3d)
            
            # Get 3D bounding box corners
            bbox_3d = self.get_3d_bounding_box(position_3d, size_3d, obj_state.get('rotation', 0))
            
            # Project to 2D
            bbox_2d = []
            for corner in bbox_3d:
                point_2d = self.project_3d_to_2d(corner, camera)
                if point_2d:
                    bbox_2d.append(point_2d)
            
            if len(bbox_2d) >= 4:  # Need at least 4 points visible
                # Render 3D object
                self.render_3d_object(img, obj_state, bbox_2d, camera)
                
                # Calculate 2D bounding box for annotation
                xs = [p[0] for p in bbox_2d]
                ys = [p[1] for p in bbox_2d]
                x_min, x_max = max(0, min(xs)), min(self.image_size[0], max(xs))
                y_min, y_max = max(0, min(ys)), min(self.image_size[1], max(ys))
                
                if x_max > x_min and y_max > y_min:
                    cx = (x_min + x_max) / 2 / self.image_size[0]
                    cy = (y_min + y_max) / 2 / self.image_size[1]
                    w = (x_max - x_min) / self.image_size[0]
                    h = (y_max - y_min) / self.image_size[1]
                    
                    annotations.append({
                        'class_id': obj_state['class_id'],
                        'cx': cx,
                        'cy': cy,
                        'w': w,
                        'h': h,
                        'object_id': id(obj_state),
                        'camera_view': camera['name'],
                        '3d_info': {
                            'position': position_3d,
                            'rotation': obj_state.get('rotation', 0),
                            'size': size_3d
                        }
                    })
        
        return img, annotations
    
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
    
    def export_for_web_interface(self, scenario_data, output_path):
        """Export 3D models and scene data for web interface visualization"""
        # Create comprehensive export data with 3D information
        export_data = {
            'scenario': scenario_data['scenario'],
            'time_condition': scenario_data['time_condition'],
            'weather': scenario_data['weather'],
            'traffic_pattern': scenario_data['traffic_pattern'],
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'world_config': {
                'scale': self.world_scale,
                'road_width': self.road_width,
                'sidewalk_width': self.sidewalk_width
            },
            'camera_configs': self.camera_configs,
            'scene_layout': self.scene_db['road_layouts'][scenario_data['scenario']],
            'objects': [],
            'models_3d': {}
        }
        
        # Export object states with 3D information
        for obj in scenario_data['object_states']:
            class_info = self.classes[obj['class_id']]
            model_variant = obj.get('model_variant', list(self.model_variants[class_info['name']].keys())[0])
            
            export_data['objects'].append({
                'id': obj.get('id', id(obj)),
                'class_id': obj['class_id'],
                'class_name': class_info['name'],
                'model_variant': model_variant,
                'position_3d': {
                    'x': obj['x'], 
                    'y': obj['y'], 
                    'z': obj.get('z', 0)
                },
                'rotation': obj.get('rotation', 0),
                'velocity_3d': {
                    'vx': obj['vx'], 
                    'vy': obj['vy'], 
                    'vz': obj.get('vz', 0)
                },
                'size_3d': class_info['size_3d'],
                'behavior': obj['behavior'],
                'properties': {
                    'color': obj.get('color', class_info['color_variants'][0]),
                    'lights_on': obj.get('lights_on', False),
                    'animation_state': obj.get('animation_state', 'idle')
                }
            })
        
        # Export 3D model definitions
        export_data['models_3d'] = self.export_3d_models()
        
        # Save main JSON file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Also save simplified GLTF-style model descriptions
        models_dir = os.path.join(os.path.dirname(output_path), 'models_3d')
        os.makedirs(models_dir, exist_ok=True)
        
        for class_id, class_info in self.classes.items():
            model_file = os.path.join(models_dir, f"{class_info['name']}_models.json")
            with open(model_file, 'w') as f:
                json.dump(self.generate_3d_model_data(class_id), f, indent=2)
        
        return export_data
    
    def export_3d_models(self):
        """Export 3D model definitions for web rendering"""
        models = {}
        
        for class_id, class_info in self.classes.items():
            class_name = class_info['name']
            models[class_name] = {
                'base_size': class_info['size_3d'],
                'variants': self.model_variants.get(class_name, {}),
                'geometry': self.generate_model_geometry(class_name),
                'materials': self.generate_model_materials(class_name),
                'animations': self.generate_model_animations(class_name)
            }
        
        return models
    
    def generate_3d_model_data(self, class_id):
        """Generate detailed 3D model data for a specific class"""
        class_info = self.classes[class_id]
        class_name = class_info['name']
        
        if class_name == 'car':
            return self.generate_car_3d_model()
        elif class_name == 'pedestrian':
            return self.generate_pedestrian_3d_model()
        elif class_name == 'cyclist':
            return self.generate_cyclist_3d_model()
    
    def generate_car_3d_model(self):
        """Generate 3D car model data"""
        return {
            'type': 'car',
            'vertices': [
                # Simplified car body vertices (normalized to unit size)
                [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],  # Bottom
                [-0.4, -0.4, 0.3], [0.4, -0.4, 0.3], [0.4, 0.4, 0.3], [-0.4, 0.4, 0.3],  # Mid
                [-0.3, -0.3, 0.6], [0.3, -0.3, 0.6], [0.3, 0.3, 0.6], [-0.3, 0.3, 0.6],  # Top
            ],
            'faces': [
                # Bottom, sides, top, front, back
                [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0],
                [4, 8, 9, 5], [5, 9, 10, 6], [6, 10, 11, 7], [7, 11, 8, 4]
            ],
            'wheels': [
                {'position': [-0.3, -0.5, 0.1], 'radius': 0.15},
                {'position': [0.3, -0.5, 0.1], 'radius': 0.15},
                {'position': [-0.3, 0.5, 0.1], 'radius': 0.15},
                {'position': [0.3, 0.5, 0.1], 'radius': 0.15}
            ],
            'materials': {
                'body': {'type': 'metal', 'roughness': 0.4, 'metalness': 0.8},
                'windows': {'type': 'glass', 'opacity': 0.3, 'reflectivity': 0.7},
                'wheels': {'type': 'rubber', 'roughness': 0.9, 'metalness': 0.1}
            }
        }
    
    def generate_pedestrian_3d_model(self):
        """Generate 3D pedestrian model data"""
        return {
            'type': 'pedestrian',
            'body_parts': {
                'head': {'position': [0, 0, 0.85], 'size': [0.15, 0.15, 0.2]},
                'torso': {'position': [0, 0, 0.5], 'size': [0.3, 0.2, 0.5]},
                'left_arm': {'position': [-0.2, 0, 0.5], 'size': [0.08, 0.08, 0.4]},
                'right_arm': {'position': [0.2, 0, 0.5], 'size': [0.08, 0.08, 0.4]},
                'left_leg': {'position': [-0.1, 0, 0.25], 'size': [0.1, 0.1, 0.5]},
                'right_leg': {'position': [0.1, 0, 0.25], 'size': [0.1, 0.1, 0.5]}
            },
            'animations': {
                'idle': {'duration': 2.0, 'loop': True},
                'walk': {'duration': 1.0, 'loop': True},
                'run': {'duration': 0.6, 'loop': True}
            },
            'materials': {
                'skin': {'type': 'organic', 'roughness': 0.7},
                'clothing': {'type': 'fabric', 'roughness': 0.85}
            }
        }
    
    def generate_cyclist_3d_model(self):
        """Generate 3D cyclist model data"""
        return {
            'type': 'cyclist',
            'components': {
                'bicycle_frame': {
                    'vertices': [
                        [-0.4, 0, 0.3], [0.4, 0, 0.3],  # Main frame
                        [-0.35, 0, 0.1], [0.35, 0, 0.1],  # Bottom bracket
                        [0.3, 0, 0.5]  # Handlebars
                    ],
                    'connections': [[0, 1], [0, 2], [1, 3], [2, 3], [1, 4]]
                },
                'wheels': [
                    {'position': [-0.4, 0, 0.2], 'radius': 0.3, 'spokes': 12},
                    {'position': [0.4, 0, 0.2], 'radius': 0.3, 'spokes': 12}
                ],
                'rider': {
                    'position': [0, 0, 0.6],
                    'pose': 'riding',
                    'lean_angle': 15  # degrees forward
                }
            },
            'materials': {
                'frame': {'type': 'metal', 'color': 'variable'},
                'wheels': {'type': 'rubber', 'roughness': 0.9},
                'rider': {'type': 'mixed', 'uses_pedestrian_materials': True}
            }
        }
    
    def generate_model_geometry(self, class_name):
        """Generate basic geometry for 3D models"""
        # Simplified geometry definitions
        return {
            'type': 'box',  # Can be extended to 'mesh' with actual vertices
            'subdivision_level': 2
        }
    
    def generate_model_materials(self, class_name):
        """Generate material definitions for 3D models"""
        return {
            'diffuse': 'variable',  # Will use color_variants
            'specular': 0.5,
            'roughness': 0.6,
            'metalness': 0.3 if class_name == 'car' else 0.0
        }
    
    def generate_model_animations(self, class_name):
        """Generate animation definitions for 3D models"""
        if class_name == 'pedestrian':
            return ['idle', 'walk', 'run', 'stand']
        elif class_name == 'cyclist':
            return ['pedaling', 'coasting', 'stopped']
        else:
            return ['moving', 'stopped']
    
    def export_3d_model_library(self):
        """Export complete 3D model library for web use"""
        models_dir = os.path.join(self.output_dir, 'web_models')
        
        # Export main model library
        library = {
            'format_version': '1.0',
            'export_date': datetime.now().isoformat(),
            'coordinate_system': 'right_handed_y_up',
            'units': 'meters',
            'models': self.export_3d_models(),
            'materials': {
                'asphalt': {'diffuse': (60, 60, 60), 'roughness': 0.9},
                'concrete': {'diffuse': (140, 140, 140), 'roughness': 0.8},
                'glass': {'diffuse': (50, 50, 70), 'opacity': 0.3, 'reflectivity': 0.7},
                'metal': {'diffuse': 'variable', 'roughness': 0.4, 'metalness': 0.8},
                'rubber': {'diffuse': (30, 30, 30), 'roughness': 0.9}
            },
            'environments': {
                'day': {'sun_angle': 45, 'sun_intensity': 1.0, 'ambient': 0.3},
                'dusk': {'sun_angle': 15, 'sun_intensity': 0.7, 'ambient': 0.2},
                'night': {'sun_angle': -20, 'sun_intensity': 0.1, 'ambient': 0.1},
                'dawn': {'sun_angle': 10, 'sun_intensity': 0.6, 'ambient': 0.25}
            }
        }
        
        with open(os.path.join(models_dir, 'model_library.json'), 'w') as f:
            json.dump(library, f, indent=2)
        
        print(f"Exported 3D model library to {models_dir}")
    
    def generate_dataset(self):
        """Generate complete dataset with realistic scenarios"""
        print(f"Generating synthetic traffic dataset in {self.output_dir}")
        print(f"Seed: {self.seed} (for reproducibility)")
        
        # Generate training set
        print(f"\nGenerating {self.num_train} training images...")
        train_manifest = []
        train_scenarios = []
        
        # Create directories for 3D models export
        models_export_dir = os.path.join(self.output_dir, 'web_models')
        os.makedirs(models_export_dir, exist_ok=True)
        
        for i in range(self.num_train):
            # Generate and render scenario with 3D
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
            
            # Save scenario for web interface
            scenario_path = os.path.join(self.output_dir, 'train', 'scenarios', f'{i:06d}.json')
            os.makedirs(os.path.dirname(scenario_path), exist_ok=True)
            self.export_for_web_interface(scenario_data, scenario_path)
            
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
            
            # Select camera for validation
            camera = self.camera_configs[i % len(self.camera_configs)]
            scenario_data['camera'] = camera
            
            # Render 3D frame
            img, annotations = self.render_3d_frame(scenario_data, camera)
            
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
        
        # Export 3D models for web interface
        print(f"\nExporting 3D models for web interface...")
        self.export_3d_model_library()
        
        print(f"\nDataset generation complete!")
        print(f"Total images: {self.num_train + self.num_val}")
        print(f"Video sequences: {num_sequences} ({frames_per_sequence} frames each)")
        print(f"Classes: cars, pedestrians, cyclists")
        print(f"Scenarios: {', '.join(self.scenario_templates)}")
        print(f"Camera viewpoints: {len(self.camera_configs)} different angles")
        print(f"3D models exported to: {models_export_dir}")
        print(f"Format: YOLO (normalized xywh) with 3D metadata")
        print(f"\nThe dataset includes:")
        print(f"- 3D rendered scenes from multiple viewpoints")
        print(f"- Realistic lighting and weather conditions")
        print(f"- 3D model exports for web visualization")
        print(f"- Full scene metadata for each frame")
    
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
            
            # Export scenario data for UI/web interface
            ui_scenario_path = os.path.join(ui_dir, f'scenario_{i:02d}.json')
            generator.export_for_web_interface(scenario_data, ui_scenario_path)
            
            print(f"  Generated UI scenario {i+1}/5")
        
        print("UI training data generated!")


if __name__ == '__main__':
    main()