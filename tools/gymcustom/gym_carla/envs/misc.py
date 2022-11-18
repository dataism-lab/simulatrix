# 2022: Ildar Nurgaliev (i.nurgaliev@docet.ai)

import math
from typing import Tuple

import carla
import numpy as np
from matplotlib.path import Path


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def compute_angle(u, v):
    return -math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])


def get_location(vehicle) -> Tuple[float, float]:
    """
    Get the position of a vehicle
    :param vehicle: the vehicle whose position is to get
    :return: speed as a float in Kmh
    """
    trans = vehicle.get_transform()
    return trans.location.x, trans.location.y


def get_pos(vehicle) -> Tuple[float, float, float]:
    """
    Get the position of a vehicle
    :param vehicle: the vehicle whose position is to get
    :return: speed as a float in Kmh and yaw in degrees
    """
    trans = vehicle.get_transform()
    # yaw = trans.rotation.yaw / 180 * np.pi
    yaw = trans.rotation.yaw
    return trans.location.x, trans.location.y, yaw


def get_box_info(vehicle):
    """
    Get the ego bounding box info
    :param vehicle: the vehicle whose info is to get
    :return: a tuple of half length, width of the vehicle
    """
    bb = vehicle.bounding_box
    l = bb.extent.x
    w = bb.extent.y
    info = (l, w)
    return info


def get_local_pose(global_pose, ego_pose):
    """
    Transform vehicle to ego coordinate
    :param global_pose: surrounding vehicle's global pose
    :param ego_pose: ego vehicle pose
    :return: tuple of the pose of the surrounding vehicle in ego coordinate
    """
    x, y, yaw = global_pose
    ego_x, ego_y, ego_yaw = ego_pose
    R = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                  [-np.sin(ego_yaw), np.cos(ego_yaw)]])
    vec_local = R.dot(np.array([x - ego_x, y - ego_y]))
    yaw_local = yaw - ego_yaw
    local_pose = (vec_local[0], vec_local[1], yaw_local)
    return local_pose


def get_pixel_info(local_info, d_behind, obs_range, image_size):
    """
    Transform local vehicle info to pixel info, with ego placed at lower center of image.
    Here the ego local coordinate is left-handed, the pixel coordinate is also left-handed,
    with its origin at the left bottom.
    :param local_info: local vehicle info in ego coordinate
    :param d_behind: distance from ego to bottom of FOV
    :param obs_range: length of edge of FOV
    :param image_size: size of edge of image
    :return: tuple of pixel level info, including (x, y, yaw, l, w) all in pixels
    """
    x, y, yaw, l, w = local_info
    x_pixel = (x + d_behind) / obs_range * image_size
    y_pixel = y / obs_range * image_size + image_size / 2
    yaw_pixel = yaw
    l_pixel = l / obs_range * image_size
    w_pixel = w / obs_range * image_size
    pixel_tuple = (x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel)
    return pixel_tuple


def get_way_direction(point_origin, point_end) -> float:
    # clockwise 0 and +180; counter-clockwise  0 and -180
    x1, y1 = point_origin
    x2, y2 = point_end

    dx = x2 - x1
    dy = y2 - y1
    rad = math.atan(abs(dy) / (abs(dx) + 1e-7))
    angle = rad * (180. / math.pi)
    if dx < 0 and dy < 0:
        angle = -180 + angle
    elif dx < 0:
        angle = 180 - angle
    elif dy < 0:
        angle = -angle
    return angle


def angle_vec(angle: float) -> np.array:
    rad = math.radians(angle)
    vec = np.array([np.cos(rad), np.sin(rad)])
    return vec / np.linalg.norm(vec)


def angle_difference(angle1: float, angle2: float) -> Tuple[float, bool]:
    direction_unit1 = [np.cos(math.radians(angle1)), np.sin(math.radians(angle1))]
    direction_unit2 = [np.cos(math.radians(angle2)), np.sin(math.radians(angle2))]
    dot = np.dot(direction_unit1, direction_unit2)
    rad = np.arccos(dot)
    angle = math.degrees(rad)
    is_reversed = dot.item() < 0
    return angle, is_reversed


def get_bounding_polygon(actor) -> np.ndarray:
    # Get x, y and yaw of the actor
    trans = actor.get_transform()
    x = trans.location.x
    y = trans.location.y
    yaw = trans.rotation.yaw / 180 * np.pi
    # Get length and width
    bb = actor.bounding_box
    l = bb.extent.x
    w = bb.extent.y
    # Get bounding box polygon in the actor's local coordinate
    poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
    # Get rotation matrix to transform to global coordinate
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    # Get global bounding box polygon
    poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
    return poly


def is_left_of_center(way_point_0: np.array, way_point_1: np.array, ego_point: np.array) -> bool:
    way_vec = way_point_1 - way_point_0
    ego_vec = ego_point - way_point_0
    det = np.linalg.det([way_vec, ego_vec])
    return det < 0


def get_poly_from_info(info):
    """
    Get polygon for info, which is a tuple of (x, y, yaw, l, w) in a certain coordinate
    :param info: tuple of x,y position, yaw angle, and half length and width of vehicle
    :return: a numpy array of size 4x2 of the vehicle rectangle corner points position
    """
    x, y, yaw, l, w = info
    poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
    return poly


def get_pixels_inside_vehicle(pixel_info, pixel_grid):
    """
    Get pixels inside a vehicle, given its pixel level info (x, y, yaw, l, w)
    :param pixel_info: pixel level info of the vehicle
    :param pixel_grid: pixel_grid of the image, a tall numpy array pf x, y pixels
    :return: the pixels that are inside the vehicle
    """
    poly = get_poly_from_info(pixel_info)
    p = Path(poly)  # make a polygon
    grid = p.contains_points(pixel_grid)
    isinPoly = np.where(grid == True)
    pixels = np.take(pixel_grid, isinPoly, axis=0)[0]
    return pixels


def get_vehicle_lon_speed(carla_vehicle: carla.Vehicle):
    """
     Get the longitudinal speed of a carla vehicle
     :param carla_vehicle: the carla vehicle
     :type carla_vehicle: carla.Vehicle
     :return: speed of a carla vehicle [m/s]
     :rtype: float64
    """
    carla_velocity_vec3 = carla_vehicle.get_velocity()
    vec4 = np.array([carla_velocity_vec3.x,
                     carla_velocity_vec3.y,
                     carla_velocity_vec3.z, 1]).reshape(4, 1)
    carla_trans = np.array(carla_vehicle.get_transform().get_matrix())
    carla_trans.reshape(4, 4)
    carla_trans[0:3, 3] = 0.0
    vel_in_vehicle = np.linalg.inv(carla_trans) @ vec4
    return vel_in_vehicle[0]


def get_lane_dis(p1: np.array, p2: np.array, p3: np.array) -> float:
    """
    Get the distance from P3 perpendicular to a line drawn between P1 and P2.
    """
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3, axis=-1)) / np.linalg.norm(p2 - p1)


def track_length(waypoints: np.array) -> float:
    track_dists = np.linalg.norm(waypoints[:-1] - waypoints[1:], axis=1)
    last_dist = np.linalg.norm(waypoints[0] - waypoints[-1])
    return sum(track_dists) + last_dist


def get_preview_lane_dis(waypoints, x, y, idx=2):
    """
    Calculate distance from (x, y) to a certain waypoint
    :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
    :param x: x position of vehicle
    :param y: y position of vehicle
    :param idx: index of the waypoint to which the distance is calculated
    :return: a tuple of the distance and the waypoint orientation
    """
    waypt = waypoints[idx]
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    w = np.array([np.cos(waypt[2] / 180 * np.pi), np.sin(waypt[2] / 180 * np.pi)])
    cross = np.cross(w, vec / lv)
    dis = - lv * cross
    return dis, w


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)
    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return d_angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def set_carla_transform(pose):
    """
    Get a carla transform object given pose.
    :param pose: list if size 3, indicating the wanted [x, y, yaw] of the transform
    :return: a carla transform object
    """
    transform = carla.Transform()
    transform.location.x = pose[0]
    transform.location.y = pose[1]
    transform.rotation.yaw = pose[2]
    return transform


def is_on_straight(point1, point2, point3) -> bool:
    """
    Determines whether the 3 two-dimensional points lie close to the same line.
    (and thus the current track segment may be considered a straight)
    Points MUST be ordered as p1 -> p2 -> p3.
    """
    largest = _distance(point1, point3)
    d12 = _distance(point1, point2)
    d23 = _distance(point2, point3)
    return largest > (d12 + d23) * 0.98


def _distance(p1, p2) -> float:
    return np.linalg.norm(np.array(p1 - p2))
