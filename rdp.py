#!/usr/bin/env python
# Software License Agreement (proprietary)
# @author    Waleed Uddin <mwaleed2018@gmail.com>

import numpy as np

# Controls the coarsness of path. Bigger values simplify curve more and output less points
CURVE_SIMPLIFICATION_EPSILON = 0.2


def path_to_numpy_array(path_msg):
    """
    Convert a ROS nav_msgs/Path message to a NumPy array of coordinates.
    """
    coordinates = []

    for pose_stamped in path_msg.poses:
        x = pose_stamped.pose.position.x
        y = pose_stamped.pose.position.y
        coordinates.append([x, y])

    return np.array(coordinates)


def perpendicular_distance(point, line_start, line_end):
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    point = np.array(point)

    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    else:
        return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)


def rdp(points):
    """
    Implementation of Ramer-Douglas-Peucker algorithm
    https://cartography-playground.gitlab.io/playgrounds/douglas-peucker-algorithm/
    """
    if len(points) < 3:
        return points

    dmax = 0.0
    index = 0
    end = len(points)

    for i in range(1, end - 1):
        d = perpendicular_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax > CURVE_SIMPLIFICATION_EPSILON:
        # If a point found outside of epsilon tolerance, call rdp again in two segments that this point
        # partitions and we repeat rdp in those two segments in the same way
        rec_results1 = rdp(points[:index + 1])
        rec_results2 = rdp(points[index:])
        result = np.vstack((rec_results1[:-1], rec_results2))  # Stack arrays vertically
    else:
        result = np.array([points[0], points[-1]])

    return result


def simplify_path(path_msg):
    path = path_to_numpy_array(path_msg)
    simplified_path = rdp(path).reshape(-1).tolist()
    return simplified_path
