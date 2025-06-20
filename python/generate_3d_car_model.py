#!/usr/bin/env python3
"""
Generate realistic 3D car models for synthetic dataset
"""

import numpy as np
from PIL import Image, ImageDraw
import math

class Car3DModel:
    def __init__(self):
        # Car dimensions in meters
        self.length = 4.5
        self.width = 1.8
        self.height = 1.5
        self.wheel_radius = 0.35
        self.ground_clearance = 0.15
        
    def get_car_vertices(self, position=(0, 0, 0), rotation=0, car_type='sedan'):
        """Generate 3D vertices for a car model"""
        x, y, z = position
        
        # Base car body vertices (before rotation)
        if car_type == 'sedan':
            vertices = self._get_sedan_vertices()
        elif car_type == 'suv':
            vertices = self._get_suv_vertices()
        elif car_type == 'truck':
            vertices = self._get_truck_vertices()
        else:
            vertices = self._get_sedan_vertices()
        
        # Apply rotation around Z axis
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        
        rotated_vertices = []
        for vx, vy, vz in vertices:
            rx = vx * cos_r - vy * sin_r
            ry = vx * sin_r + vy * cos_r
            rotated_vertices.append((rx + x, ry + y, vz + z))
        
        return rotated_vertices
    
    def _get_sedan_vertices(self):
        """Get vertices for a sedan car model"""
        l, w, h = self.length/2, self.width/2, self.height
        gc = self.ground_clearance
        
        # Car body is made of multiple sections for realism
        vertices = []
        
        # Bottom chassis (raised from ground)
        chassis_points = [
            (-l, -w, gc), (l, -w, gc), (l, w, gc), (-l, w, gc),
            (-l, -w, gc+0.3), (l, -w, gc+0.3), (l, w, gc+0.3), (-l, w, gc+0.3)
        ]
        
        # Hood (front section) - sloped
        hood_points = [
            (l*0.3, -w*0.9, gc+0.3), (l, -w*0.9, gc+0.3),
            (l, w*0.9, gc+0.3), (l*0.3, w*0.9, gc+0.3),
            (l*0.3, -w*0.8, gc+0.6), (l*0.9, -w*0.8, gc+0.5),
            (l*0.9, w*0.8, gc+0.5), (l*0.3, w*0.8, gc+0.6)
        ]
        
        # Windshield and cabin
        cabin_points = [
            (l*0.3, -w*0.8, gc+0.6), (l*0.3, w*0.8, gc+0.6),
            (-l*0.1, w*0.8, gc+0.6), (-l*0.1, -w*0.8, gc+0.6),
            (l*0.1, -w*0.7, gc+h), (l*0.1, w*0.7, gc+h),
            (-l*0.3, w*0.7, gc+h), (-l*0.3, -w*0.7, gc+h)
        ]
        
        # Rear section with trunk
        rear_points = [
            (-l*0.3, -w*0.8, gc+0.6), (-l, -w*0.9, gc+0.4),
            (-l, w*0.9, gc+0.4), (-l*0.3, w*0.8, gc+0.6),
            (-l*0.3, -w*0.7, gc+h*0.9), (-l*0.8, -w*0.8, gc+0.5),
            (-l*0.8, w*0.8, gc+0.5), (-l*0.3, w*0.7, gc+h*0.9)
        ]
        
        # Wheels positions (not vertices, but mounting points)
        wheel_mounts = [
            (l*0.7, -w, gc),   # Front right
            (l*0.7, w, gc),    # Front left
            (-l*0.7, -w, gc),  # Rear right
            (-l*0.7, w, gc)    # Rear left
        ]
        
        vertices.extend(chassis_points)
        vertices.extend(hood_points)
        vertices.extend(cabin_points)
        vertices.extend(rear_points)
        
        return vertices
    
    def _get_suv_vertices(self):
        """Get vertices for an SUV model (taller, boxier)"""
        l, w, h = self.length/2, self.width/2, self.height * 1.3
        gc = self.ground_clearance * 1.5
        
        vertices = []
        
        # Higher chassis
        chassis = [
            (-l, -w, gc), (l, -w, gc), (l, w, gc), (-l, w, gc),
            (-l, -w, gc+0.4), (l, -w, gc+0.4), (l, w, gc+0.4), (-l, w, gc+0.4)
        ]
        
        # Boxier body
        body = [
            (-l*0.9, -w*0.95, gc+0.4), (l*0.9, -w*0.95, gc+0.4),
            (l*0.9, w*0.95, gc+0.4), (-l*0.9, w*0.95, gc+0.4),
            (-l*0.9, -w*0.95, gc+h), (l*0.8, -w*0.95, gc+h),
            (l*0.8, w*0.95, gc+h), (-l*0.9, w*0.95, gc+h)
        ]
        
        vertices.extend(chassis)
        vertices.extend(body)
        
        return vertices
    
    def _get_truck_vertices(self):
        """Get vertices for a pickup truck model"""
        l, w, h = self.length/2 * 1.2, self.width/2, self.height
        gc = self.ground_clearance * 1.3
        
        vertices = []
        
        # Truck bed
        bed = [
            (-l, -w, gc), (-l*0.2, -w, gc), (-l*0.2, w, gc), (-l, w, gc),
            (-l, -w, gc+0.6), (-l*0.2, -w, gc+0.6), (-l*0.2, w, gc+0.6), (-l, w, gc+0.6)
        ]
        
        # Cabin
        cabin = [
            (-l*0.2, -w*0.95, gc), (l*0.8, -w*0.95, gc),
            (l*0.8, w*0.95, gc), (-l*0.2, w*0.95, gc),
            (-l*0.2, -w*0.9, gc+h), (l*0.6, -w*0.9, gc+h),
            (l*0.6, w*0.9, gc+h), (-l*0.2, w*0.9, gc+h)
        ]
        
        vertices.extend(bed)
        vertices.extend(cabin)
        
        return vertices
    
    def render_car(self, image, draw, vertices, camera_params, color=(200, 50, 50)):
        """Render car model onto image given vertices and camera parameters"""
        # Group vertices into faces for rendering
        faces = self._get_car_faces(len(vertices))
        
        # Project vertices to 2D
        projected = []
        for vertex in vertices:
            p2d = self._project_to_2d(vertex, camera_params)
            if p2d:
                projected.append(p2d)
            else:
                projected.append(None)
        
        # Sort faces by depth (painter's algorithm)
        face_depths = []
        for face in faces:
            if all(projected[i] is not None for i in face):
                # Calculate face center depth
                center = np.mean([vertices[i] for i in face], axis=0)
                depth = self._calculate_depth(center, camera_params)
                face_depths.append((depth, face))
        
        # Draw faces back to front
        face_depths.sort(reverse=True)
        
        for depth, face in face_depths:
            points = [projected[i] for i in face if projected[i] is not None]
            if len(points) >= 3:
                # Vary color based on face orientation for shading
                face_normal = self._calculate_face_normal(
                    vertices[face[0]], vertices[face[1]], vertices[face[2]]
                )
                shade_factor = 0.7 + 0.3 * max(0, face_normal[2])
                face_color = tuple(int(c * shade_factor) for c in color)
                
                draw.polygon(points, fill=face_color, outline=tuple(int(c*0.8) for c in face_color))
        
        # Draw wheels
        self._render_wheels(draw, vertices, projected, camera_params)
        
        # Add details like windows, lights
        self._render_car_details(draw, vertices, projected, camera_params)
    
    def _get_car_faces(self, num_vertices):
        """Define faces for the car model based on vertex count"""
        # This is a simplified face definition
        # In practice, you'd have specific face indices for each car part
        faces = []
        
        # Chassis faces (first 8 vertices)
        if num_vertices >= 8:
            faces.extend([
                [0, 1, 2, 3],  # Bottom
                [4, 5, 6, 7],  # Top
                [0, 1, 5, 4],  # Front
                [2, 3, 7, 6],  # Back
                [0, 3, 7, 4],  # Left
                [1, 2, 6, 5]   # Right
            ])
        
        # Add more faces for other parts...
        return faces
    
    def _project_to_2d(self, point_3d, camera_params):
        """Project 3D point to 2D screen coordinates"""
        # Similar to the projection in generate_synthetic_dataset.py
        # but with improved perspective calculation
        x, y, z = point_3d
        cam_x, cam_y, cam_z = camera_params['position']
        pitch, yaw, roll = [np.radians(angle) for angle in camera_params['rotation']]
        
        # Translate to camera coordinates
        dx = x - cam_x
        dy = y - cam_y
        dz = z - cam_z
        
        # Apply camera rotations
        # Yaw (rotation around Z)
        cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
        x1 = dx * cos_yaw - dy * sin_yaw
        y1 = dx * sin_yaw + dy * cos_yaw
        z1 = dz
        
        # Pitch (rotation around X)
        cos_pitch, sin_pitch = np.cos(-pitch), np.sin(-pitch)
        x2 = x1
        y2 = y1 * cos_pitch - z1 * sin_pitch
        z2 = y1 * sin_pitch + z1 * cos_pitch
        
        # Camera space
        cam_x = x2
        cam_y = -z2
        cam_z = y2
        
        if cam_z <= 0.1:
            return None
        
        # Perspective projection
        fov_rad = np.radians(camera_params['fov'])
        f = camera_params['image_size'][0] / (2 * np.tan(fov_rad / 2))
        
        x_2d = f * cam_x / cam_z + camera_params['image_size'][0] / 2
        y_2d = f * cam_y / cam_z + camera_params['image_size'][1] / 2
        
        # Check bounds
        if 0 <= x_2d < camera_params['image_size'][0] and 0 <= y_2d < camera_params['image_size'][1]:
            return (int(x_2d), int(y_2d))
        return None
    
    def _calculate_depth(self, point, camera_params):
        """Calculate depth of point from camera"""
        cam_pos = np.array(camera_params['position'])
        return np.linalg.norm(np.array(point) - cam_pos)
    
    def _calculate_face_normal(self, v1, v2, v3):
        """Calculate face normal for shading"""
        edge1 = np.array(v2) - np.array(v1)
        edge2 = np.array(v3) - np.array(v1)
        normal = np.cross(edge1, edge2)
        return normal / np.linalg.norm(normal)
    
    def _render_wheels(self, draw, vertices, projected, camera_params):
        """Render car wheels"""
        # Simplified wheel rendering
        wheel_color = (30, 30, 30)
        # Would add actual wheel rendering logic here
        pass
    
    def _render_car_details(self, draw, vertices, projected, camera_params):
        """Render car details like windows, lights"""
        # Add windows with transparency effect
        # Add headlights and taillights
        # Would add actual detail rendering logic here
        pass


def test_car_model():
    """Test the 3D car model generation"""
    # Create test image
    img = Image.new('RGB', (800, 600), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Create car model
    car = Car3DModel()
    
    # Test camera parameters
    camera = {
        'position': (10, -10, 5),
        'rotation': (-30, 45, 0),
        'fov': 60,
        'image_size': (800, 600)
    }
    
    # Generate and render cars
    positions = [
        ((0, 0, 0), 0, 'sedan'),
        ((5, 5, 0), 45, 'suv'),
        ((-5, 5, 0), -30, 'truck')
    ]
    
    colors = [(200, 50, 50), (50, 50, 200), (50, 150, 50)]
    
    for i, (pos, rot, car_type) in enumerate(positions):
        vertices = car.get_car_vertices(pos, np.radians(rot), car_type)
        car.render_car(img, draw, vertices, camera, colors[i])
    
    img.save('test_3d_car_model.jpg')
    print("Saved test car model to test_3d_car_model.jpg")


if __name__ == "__main__":
    test_car_model()