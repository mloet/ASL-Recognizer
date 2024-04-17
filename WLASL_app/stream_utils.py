import numpy as np
import torch

def calculate_normal_vector(point1, point2, point3):
  vector1 = np.array(point2) - np.array(point1)
  vector2 = np.array(point3) - np.array(point1)
  normal_vector = np.cross(vector1, vector2)
  magnitude = np.linalg.norm(normal_vector)
  
  return normal_vector/magnitude

def transform_points(points, normal_vector):
  normal_vector = normal_vector / np.linalg.norm(normal_vector)
  
  z_axis = np.array([0, 0, 1])
  if np.allclose(normal_vector, z_axis):
    rotation_matrix = np.eye(3)
  else:
    rotation_axis = np.cross(normal_vector, z_axis)
    rotation_angle = np.arccos(np.dot(normal_vector, z_axis))
    
    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                                                    [rotation_axis[2], 0, -rotation_axis[0]],
                                                                    [-rotation_axis[1], rotation_axis[0], 0]]) + \
                                      (1 - np.cos(rotation_angle)) * np.outer(rotation_axis, rotation_axis)
  
  transformed_points = np.dot(rotation_matrix, points.T).T
  
  return transformed_points

def project(point):
  px, py, pz = point

  x1, y1 = (0, 0)
  x2, y2 = (0, 0)

  if not any(c > 1 or c < 0 for c in point[:2]):
    return point
  if(px < 0 and py < 0):
    return np.array([0, 0, pz])
  if(px < 0 and py > 1):
    return np.array([0, 1, pz])
  if(px > 1 and py > 1):
    return np.array([1, 1, pz])
  if(px > 1 and py < 0):
    return np.array([1, 0, pz])
  if(px < 0):
    x1, y1 = (0, 1)
    x2, y2 = (0, 0)
  if(px > 1):
    x1, y1 = (1, 1)
    x2, y2 = (1, 0)
  if(py < 0):
    x1, y1 = (1, 0)
    x2, y2 = (0, 0)
  if(py > 1):
    x1, y1 = (0, 1)
    x2, y2 = (1, 1)

  dx = x2 - x1
  dy = y2 - y1
  
  t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)
  
  projected_x = x1 + t * dx
  projected_y = y1 + t * dy
  
  return np.array([projected_x, projected_y, pz])

def midpoint3d(p1, p2):
  return np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2])

def extract_keypoints_3d(results, prev):
    pose = np.array([project([landmark.x, landmark.y, landmark.z]) for landmark in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,3))
    neck = midpoint3d(pose[11], pose[12])
    hips = midpoint3d(pose[23], pose[24])
    pose = np.append(pose, np.array([neck, hips]), axis = 0)[[0, 33, 12, 14, 16, 11, 13, 15, 34, 5, 2, 8, 7], :]
    drw = pose[4] - prev[4]
    dlw = pose[7] - prev[7]
    lh = np.array([project([landmark.x, landmark.y, landmark.z]) for landmark in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else prev[13:34] + dlw
    rh = np.array([project([landmark.x, landmark.y, landmark.z]) for landmark in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else prev[34:] + drw
    kp = np.concatenate([pose, lh, rh])

    return transform_points(kp, calculate_normal_vector(kp[2], kp[5], kp[8]))