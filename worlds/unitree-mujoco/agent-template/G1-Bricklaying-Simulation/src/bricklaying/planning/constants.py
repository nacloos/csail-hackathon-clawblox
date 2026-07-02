"""
Some key pose and joint angle definitions for planning.
"""    
import numpy as np
import pinocchio as pin


# Example yaw-aligned end effector poses for brick pick/place
_left_q1 = pin.Quaternion(np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0)
_left_q2 = pin.Quaternion(np.cos(np.pi / 16), 0, np.sin(np.pi / 16), 0)
_left_q = _left_q2 * _left_q1
R_LEFT_NOMINAL_PICK = _left_q.matrix()

_right_q1 = pin.Quaternion(np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0)
_right_q2 = pin.Quaternion(np.cos(np.pi / 16), 0, np.sin(np.pi / 16), 0)
_right_q = _right_q2 * _right_q1
R_RIGHT_NOMINAL_PICK = _right_q.matrix()


# Palm-in end effector pose
R_LEFT_PALM_IN = np.eye(3)
R_RIGHT_PALM_IN = np.eye(3)


# Downward resting arm joint configuration
Q_ARMS_REST = np.array([
    0.275,  0.225,  0.,  1.0,  0.,  0.,  0.,
    0.275, -0.225,  0.,  1.0,  0.,  0.,  0.,
])

# 15-DOF upper-body rest pose [waist_yaw=0, left_arm×7, right_arm×7]
Q_UPPER_BODY_REST = np.concatenate([[0.0], Q_ARMS_REST])