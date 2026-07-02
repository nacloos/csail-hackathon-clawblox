"""
Joint configuration for Unitree G1 robot.
Defines joint indices, groupings, and hardware-specific mappings.

To-do's:
- JointConfig's are hardcoded, but can and should just be loaded from URDF directly.
- Only defined for upper body.
"""
from __future__ import annotations

from enum import IntEnum
from typing import List, Dict
from dataclasses import dataclass
import numpy as np


# ===== Body Joints =====

class G1JointIndex(IntEnum):
    """
    Joint indices for Unitree G1 robot low-level control.
    These map directly to motor_cmd array indices in DDS messages.
    """

    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11

    # Waist
    WaistYaw = 12
    WaistRoll = 13      # NOTE: INVALID with waist locked
    WaistPitch = 14     # NOTE: INVALID with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   
    LeftWristYaw = 21     

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    # Other
    NotUsedJoint0 = 29
    NotUsedJoint1 = 30
    NotUsedJoint2 = 31
    NotUsedJoint3 = 32
    NotUsedJoint4 = 33
    NotUsedJoint5 = 34


class G1JointGroup:
    """Predefined joint groupings for easy access"""

    # Arms
    LEFT_ARM = [
        G1JointIndex.LeftShoulderPitch,
        G1JointIndex.LeftShoulderRoll,
        G1JointIndex.LeftShoulderYaw,
        G1JointIndex.LeftElbow,
        G1JointIndex.LeftWristRoll,
        G1JointIndex.LeftWristPitch,
        G1JointIndex.LeftWristYaw,
    ]

    RIGHT_ARM = [
        G1JointIndex.RightShoulderPitch,
        G1JointIndex.RightShoulderRoll,
        G1JointIndex.RightShoulderYaw,
        G1JointIndex.RightElbow,
        G1JointIndex.RightWristRoll,
        G1JointIndex.RightWristPitch,
        G1JointIndex.RightWristYaw,
    ]

    BOTH_ARMS = LEFT_ARM + RIGHT_ARM

    # Upper body 15 DoF
    UPPER_BODY = [G1JointIndex.WaistYaw] + LEFT_ARM + RIGHT_ARM

    # Waist
    WAIST = [
        G1JointIndex.WaistYaw,
        G1JointIndex.WaistRoll,
        G1JointIndex.WaistPitch,
    ]

    # Waist joints locked
    WAIST_LOCKED = [
        G1JointIndex.WaistRoll,
        G1JointIndex.WaistPitch,
    ]

    # Legs TODO


# ===== URDF joint name mapping =====

JOINT_URDF_NAME: Dict[G1JointIndex, str] = {
    G1JointIndex.WaistYaw:           'waist_yaw_joint',
    G1JointIndex.LeftShoulderPitch:  'left_shoulder_pitch_joint',
    G1JointIndex.LeftShoulderRoll:   'left_shoulder_roll_joint',
    G1JointIndex.LeftShoulderYaw:    'left_shoulder_yaw_joint',
    G1JointIndex.LeftElbow:          'left_elbow_joint',
    G1JointIndex.LeftWristRoll:      'left_wrist_roll_joint',
    G1JointIndex.LeftWristPitch:     'left_wrist_pitch_joint',
    G1JointIndex.LeftWristYaw:       'left_wrist_yaw_joint',
    G1JointIndex.RightShoulderPitch: 'right_shoulder_pitch_joint',
    G1JointIndex.RightShoulderRoll:  'right_shoulder_roll_joint',
    G1JointIndex.RightShoulderYaw:   'right_shoulder_yaw_joint',
    G1JointIndex.RightElbow:         'right_elbow_joint',
    G1JointIndex.RightWristRoll:     'right_wrist_roll_joint',
    G1JointIndex.RightWristPitch:    'right_wrist_pitch_joint',
    G1JointIndex.RightWristYaw:      'right_wrist_yaw_joint',
}


def get_q_indices(group: List[G1JointIndex], model) -> List[int]:
    """Map a DDS joint group to pinocchio q-vector indices for a given model."""
    name_to_q = {model.names[i]: model.joints[i].idx_q for i in range(1, len(model.names))}
    return [name_to_q[JOINT_URDF_NAME[j]] for j in group]


# ===== Hand Joints =====

class Dex3JointIndex(IntEnum):
    """
    Joint indices for robot hand control.
    
    Each hand has 7 DOF: thumb (3 joints) + middle finger (2 joints) + index finger (2 joints).
    
    Joint conventions:
        Thumb0: Rotational joint, 0.5 = middle position (same for both hands)
        
        Left hand:
            - Thumb1, Thumb2: positive values close
            - Middle0, Middle1, Index0, Index1: negative values close
        
        Right hand (opposite convention):
            - Thumb1, Thumb2: negative values close
            - Middle0, Middle1, Index0, Index1: positive values close
    """

    Thumb0 = 0   # Rotational joint. 0.5 = middle (both hands)
    Thumb1 = 1   # Base joint. Left: positive=closed, Right: negative=closed
    Thumb2 = 2   # Tip joint. Left: positive=closed, Right: negative=closed
    Middle0 = 3  # Base joint. Left: negative=closed, Right: positive=closed
    Middle1 = 4  # Tip joint. Left: negative=closed, Right: positive=closed
    Index0 = 5   # Base joint. Left: negative=closed, Right: positive=closed
    Index1 = 6   # Tip joint. Left: negative=closed, Right: positive=closed


# ===== Data Structures =====

@dataclass
class JointGains:
    """PD gains for a single joint."""
    kp: float  # Position gain
    kd: float  # Damping gain


@dataclass
class JointLimits:
    """Joint limits for safety."""
    q_min: float      # Minimum position (rad)
    q_max: float      # Maximum position (rad)
    dq_max: float     # Maximum velocity (rad/s)
    tau_max: float    # Maximum torque (Nm)


@dataclass
class JointConfig:
    """Complete configuration for a single joint."""
    index: G1JointIndex | Dex3JointIndex
    name: str
    gains: JointGains
    limits: JointLimits
    
    def __post_init__(self):
        # Validate
        assert self.limits.q_min < self.limits.q_max
        assert self.limits.dq_max > 0
        assert self.limits.tau_max > 0


class GroupJointConfiguration:
    """
    Configuration for a group of joints.
    Maps each joint to its gains and limits.
    Effort, joint position and velocity limits taken from urdf.
    """
    
    # Main configuration storage
    _config: Dict[IntEnum, JointConfig] = {}

    @classmethod
    def get_config(cls, joint: int) -> JointConfig:
        """Get configuration for a specific body joint"""
        return cls._config[joint]
    
    @classmethod
    def get_gains(cls, joint: int) -> JointGains:
        """Get PD gains for a body joint"""
        return cls._config[joint].gains
    
    @classmethod
    def get_gains_arrays(cls, joint_list: List[int]) -> dict[str, np.ndarray]:
        """Get kp, kd arrays for multiple joints"""
        return {
            'kp': np.array([cls._config[j].gains.kp for j in joint_list]),
            'kd': np.array([cls._config[j].gains.kd for j in joint_list]),
        }
    
    @classmethod
    def get_limits_arrays(cls, joint_list: List[int]) -> dict[str, np.ndarray]:
        """Get limit arrays for multiple joints"""
        return {
            'q_min': np.array([cls._config[j].limits.q_min for j in joint_list]),
            'q_max': np.array([cls._config[j].limits.q_max for j in joint_list]),
            'dq_max': np.array([cls._config[j].limits.dq_max for j in joint_list]),
            'tau_max': np.array([cls._config[j].limits.tau_max for j in joint_list]),
        }
    
    @classmethod
    def clamp_positions(cls, q: np.ndarray, joint_list: List[int]) -> np.ndarray:
        """Clamp positions to limits"""
        limits = cls.get_limits_arrays(joint_list)
        return np.clip(q, limits['q_min'], limits['q_max'])
    
    @classmethod
    def clamp_velocities(cls, dq: np.ndarray, joint_list: List[int]) -> np.ndarray:
        """Clamp velocities to limits"""
        limits = cls.get_limits_arrays(joint_list)
        return np.clip(dq, -limits['dq_max'], limits['dq_max'])
    
    @classmethod
    def clamp_torques(cls, tau: np.ndarray, joint_list: List[int]) -> np.ndarray:
        """Clamp torques to limits"""
        limits = cls.get_limits_arrays(joint_list)
        return np.clip(tau, -limits['tau_max'], limits['tau_max'])


# ===== Configurations =====

class G1JointConfiguration(GroupJointConfiguration):
    
    # Build configuration dictionary
    _config: Dict[G1JointIndex, JointConfig] = {
        # Left arm
        G1JointIndex.LeftShoulderPitch: JointConfig(
            index=G1JointIndex.LeftShoulderPitch,
            name="left_shoulder_pitch",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-3.0892, q_max=2.6704, dq_max=37.0, tau_max=25.0)
        ),
        G1JointIndex.LeftShoulderRoll: JointConfig(
            index=G1JointIndex.LeftShoulderRoll,
            name="left_shoulder_roll",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-1.5882, q_max=2.2515, dq_max=37.0, tau_max=25.0)
        ),
        G1JointIndex.LeftShoulderYaw: JointConfig(
            index=G1JointIndex.LeftShoulderYaw,
            name="left_shoulder_yaw",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-2.618, q_max=2.618, dq_max=37.0, tau_max=25.0)
        ),
        G1JointIndex.LeftElbow: JointConfig(
            index=G1JointIndex.LeftElbow,
            name="left_elbow_pitch",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-1.0472, q_max=2.0944, dq_max=37.0, tau_max=25.0)
        ),
        G1JointIndex.LeftWristRoll: JointConfig(
            index=G1JointIndex.LeftWristRoll,
            name="left_wrist_roll",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-1.97222205, q_max=1.97222205, dq_max=37.0, tau_max=25.0)
        ),
        G1JointIndex.LeftWristPitch: JointConfig(
            index=G1JointIndex.LeftWristPitch,
            name="left_wrist_pitch",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-1.61442956, q_max=1.61442956, dq_max=22.0, tau_max=5.0)
        ),
        G1JointIndex.LeftWristYaw: JointConfig(
            index=G1JointIndex.LeftWristYaw,
            name="left_wrist_yaw",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-1.61442956, q_max=1.61442956, dq_max=22.0, tau_max=5.0)
        ),
        
        # Right arm
        G1JointIndex.RightShoulderPitch: JointConfig(
            index=G1JointIndex.RightShoulderPitch,
            name="right_shoulder_pitch",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-3.0892, q_max=2.6704, dq_max=37.0, tau_max=25.0)
        ),
        G1JointIndex.RightShoulderRoll: JointConfig(
            index=G1JointIndex.RightShoulderRoll,
            name="right_shoulder_roll",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-1.5882, q_max=2.2515, dq_max=37.0, tau_max=25.0)
        ),
        G1JointIndex.RightShoulderYaw: JointConfig(
            index=G1JointIndex.RightShoulderYaw,
            name="right_shoulder_yaw",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-2.618, q_max=2.618, dq_max=37.0, tau_max=25.0)
        ),
        G1JointIndex.RightElbow: JointConfig(
            index=G1JointIndex.RightElbow,
            name="right_elbow_pitch",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-1.0472, q_max=2.0944, dq_max=37.0, tau_max=25.0)
        ),
        G1JointIndex.RightWristRoll: JointConfig(
            index=G1JointIndex.RightWristRoll,
            name="right_wrist_roll",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-1.97222205, q_max=1.97222205, dq_max=37.0, tau_max=25.0)
        ),
        G1JointIndex.RightWristPitch: JointConfig(
            index=G1JointIndex.RightWristPitch,
            name="right_wrist_pitch",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-1.61442956, q_max=1.61442956, dq_max=22.0, tau_max=5.0)
        ),
        G1JointIndex.RightWristYaw: JointConfig(
            index=G1JointIndex.RightWristYaw,
            name="right_wrist_yaw",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-1.61442956, q_max=1.61442956, dq_max=22.0, tau_max=5.0)
        ),
        
        # Waist
        G1JointIndex.WaistYaw: JointConfig(
            index=G1JointIndex.WaistYaw,
            name="waist_yaw",
            gains=JointGains(kp=60.0, kd=1.5),
            ## Manually limited to a subset of DoF range!
            # limits=JointLimits(q_min=-2.618, q_max=2.618, dq_max=32.0, tau_max=88.0)
            limits=JointLimits(q_min=-np.pi / 4, q_max=np.pi / 4, dq_max=32.0, tau_max=88.0)
        ),
        G1JointIndex.WaistRoll: JointConfig(
            index=G1JointIndex.WaistRoll,
            name="waist_roll",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-0.52, q_max=0.52, dq_max=30.0, tau_max=35.0)
        ),
        G1JointIndex.WaistPitch: JointConfig(
            index=G1JointIndex.WaistPitch,
            name="waist_pitch",
            gains=JointGains(kp=60.0, kd=1.5),
            limits=JointLimits(q_min=-0.52, q_max=0.52, dq_max=30.0, tau_max=35.0)
        ),

        # Legs TODO
    }


class LeftDex3JointConfiguration(GroupJointConfiguration):
    
    # Build configuration dictionary
    _config: Dict[Dex3JointIndex, JointConfig] = {
        Dex3JointIndex.Thumb0: JointConfig(
            index=Dex3JointIndex.Thumb0,
            name="thumb_rotation",
            gains=JointGains(kp=5.0, kd=0.4),
            limits=JointLimits(q_min=-1.04719755, q_max=1.04719755, dq_max=3.14, tau_max=2.45)
        ),
        Dex3JointIndex.Thumb1: JointConfig(
            index=Dex3JointIndex.Thumb1,
            name="thumb_base",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=-0.61086523, q_max=1.04719755, dq_max=12.0, tau_max=1.4)
        ),
        Dex3JointIndex.Thumb2: JointConfig(
            index=Dex3JointIndex.Thumb2,
            name="thumb_tip",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=0.0, q_max=1.74532925, dq_max=12.0, tau_max=1.4)
        ),
        Dex3JointIndex.Middle0: JointConfig(
            index=Dex3JointIndex.Middle0,
            name="middle_base",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=-1.57079632, q_max=0.0, dq_max=12.0, tau_max=1.4)
        ),
        Dex3JointIndex.Middle1: JointConfig(
            index=Dex3JointIndex.Middle1,
            name="middle_tip",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=-1.74532925, q_max=0.0, dq_max=12.0, tau_max=1.4)
        ),
        Dex3JointIndex.Index0: JointConfig(
            index=Dex3JointIndex.Index0,
            name="index_base",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=-1.57079632, q_max=0.0, dq_max=12.0, tau_max=1.4)
        ),
        Dex3JointIndex.Index1: JointConfig(
            index=Dex3JointIndex.Index1,
            name="index_tip",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=-1.74532925, q_max=0.0, dq_max=12.0, tau_max=1.4)
        ),
    }


class RightDex3JointConfiguration(GroupJointConfiguration):
    
    # Build configuration dictionary
    _config: Dict[Dex3JointIndex, JointConfig] = {
        Dex3JointIndex.Thumb0: JointConfig(
            index=Dex3JointIndex.Thumb0,
            name="thumb_rotation",
            gains=JointGains(kp=5.0, kd=0.4),
            limits=JointLimits(q_min=-1.04719755, q_max=1.04719755, dq_max=3.14, tau_max=2.45)
        ),
        Dex3JointIndex.Thumb1: JointConfig(
            index=Dex3JointIndex.Thumb1,
            name="thumb_base",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=-1.04719755, q_max=0.61086523, dq_max=12.0, tau_max=1.4)
        ),
        Dex3JointIndex.Thumb2: JointConfig(
            index=Dex3JointIndex.Thumb2,
            name="thumb_tip",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=-1.74532925, q_max=0.0, dq_max=12.0, tau_max=1.4)
        ),
        Dex3JointIndex.Middle0: JointConfig(
            index=Dex3JointIndex.Middle0,
            name="middle_base",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=0.0, q_max=1.57079632, dq_max=12.0, tau_max=1.4)
        ),
        Dex3JointIndex.Middle1: JointConfig(
            index=Dex3JointIndex.Middle1,
            name="middle_tip",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=0.0, q_max=1.74532925, dq_max=12.0, tau_max=1.4)
        ),
        Dex3JointIndex.Index0: JointConfig(
            index=Dex3JointIndex.Index0,
            name="index_base",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=0.0, q_max=1.57079632, dq_max=12.0, tau_max=1.4)
        ),
        Dex3JointIndex.Index1: JointConfig(
            index=Dex3JointIndex.Index1,
            name="index_tip",
            gains=JointGains(kp=1.0, kd=0.2),
            limits=JointLimits(q_min=0.0, q_max=1.74532925, dq_max=12.0, tau_max=1.4)
        ),
    }


# ===== Presets =====

p2 = np.pi / 2
p4 = np.pi / 4

HAND_OPEN_LEFT = np.array([0.0, -0.5, 0, 0, 0, 0, 0])
HAND_OPEN_RIGHT = np.array([0.0, 0.5, 0, 0, 0, 0, 0])

HAND_GRASP_LEFT = np.array([0.0, 0, p4, -p4, -p4, -p4, -p4])
HAND_GRASP_RIGHT = np.array([0.0, 0, -p4, p4, p4, p4, p4])

HAND_CLOSED_LEFT = np.array([0.0, 1, p2, -p2, -p2, -p2, -p2])
HAND_CLOSED_RIGHT = np.array([0.0, -1, -p2, p2, p2, p2, p2])
