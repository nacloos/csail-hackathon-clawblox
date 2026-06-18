import mujoco
import numpy as np
import pygame
import sys
import struct
import time
import threading
import quaternion

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from dataclasses import dataclass

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.utils.thread import RecurrentThread

import config
if config.ROBOT=="g1":
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_ as HandState_default
else:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"

TOPIC_RIGHT_HANDCMD = "rt/dex3/right/cmd"
TOPIC_RIGHT_HANDSTATE = "rt/dex3/right/state"

TOPIC_LEFT_HANDCMD = "rt/dex3/left/cmd"
TOPIC_LEFT_HANDSTATE = "rt/dex3/left/state"

TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"

TOPIC_REALSENSE_COLOR = "rt/realsense/color"
TOPIC_REALSENSE_DEPTH = "rt/realsense/depth"

MOTOR_SENSOR_NUM = 3
NUM_MOTOR_IDL_GO = 20
NUM_MOTOR_IDL_HG = 35

hand = True
motion = True
camera = True

x_vel = 0.0
y_vel = 0.0
theta_vel = 0.0
theta = 0.0
  
# -- DDS image message type -- #  
@dataclass
@annotate.final
@annotate.autoid("sequential")
class SimImage_(idl.IdlStruct, typename="sim.msg.dds_.SimImage_"):
    height: types.uint32
    width: types.uint32
    encoding: str
    data: types.sequence[types.uint8]


# -- Optional ROS2 support for cmd_vel -- #

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False


if RCLPY_AVAILABLE:
    class G1TwistCmdNode(Node):
        def __init__(self):
            super().__init__('g1_twist_cmd_node')
            self.subscription = self.create_subscription(Twist, 'cmd_vel', self.callback_twist, 10)

        def callback_twist(self, msg):
            global x_vel, y_vel, theta_vel
            x_vel = msg.linear.x
            y_vel = msg.linear.y
            theta_vel = msg.angular.z


class UnitreeSdk2Bridge:

    def __init__(self, mj_model, mj_data, lock=None):
        global hand, motion
        self.mj_model = mj_model
        self.mj_data = mj_data
        self._lock = lock

        self.num_motor = 29
        if hand:
            self.num_hand_joint = 14
        else:
            self.num_hand_joint = 0
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.have_imu = False
        self.have_frame_sensor = False
        self.dt = self.mj_model.opt.timestep
        self.idl_type = (self.num_motor > NUM_MOTOR_IDL_GO) # 0: unitree_go, 1: unitree_hg

        self.joystick = None

        # Check sensor
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name == "imu_quat":
                self.have_imu_ = True
            if name == "frame_pos":
                self.have_frame_sensor_ = True

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread = RecurrentThread(interval=self.dt, target=self.PublishLowState, name="sim_lowstate")
        self.lowStateThread.Start()

        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(interval=self.dt, target=self.PublishHighState, name="sim_highstate")
        self.HighStateThread.Start()

        if hand:
            self.left_hand_state = HandState_default()
            self.right_hand_state = HandState_default()
            self.left_hand_state_puber = ChannelPublisher(TOPIC_LEFT_HANDSTATE, HandState_)
            self.right_hand_state_puber = ChannelPublisher(TOPIC_RIGHT_HANDSTATE, HandState_)

            self.right_hand_state_puber.Init()
            self.left_hand_state_puber.Init()
            self.RightHandStateThread = RecurrentThread(interval=self.dt, target=self.PublishRightHandState, name="sim_righthandstate")
            self.LeftHandStateThread = RecurrentThread(interval=self.dt, target=self.PublishLeftHandState, name="sim_lefthandstate")
            self.RightHandStateThread.Start()
            self.LeftHandStateThread.Start()

        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)

        if hand:
            self.left_hand_cmd_suber = ChannelSubscriber(TOPIC_LEFT_HANDCMD, HandCmd_)
            self.right_hand_cmd_suber = ChannelSubscriber(TOPIC_RIGHT_HANDCMD, HandCmd_)
            self.left_hand_cmd_suber.Init(self.LeftHandCmdHandler, 10)
            self.right_hand_cmd_suber.Init(self.RightHandCmdHandler, 10)

        # Optional ROS2 motion control
        if motion:
            if not RCLPY_AVAILABLE:
                print("WARNING: rclpy not available, disabling motion/cmd_vel support")
                motion = False
            else:
                rclpy.init()
                time.sleep(3)
                self.g1_twist_cmd_node = G1TwistCmdNode()

        self.PrintSceneInformation()

        if motion:
            ros_feed_thread = threading.Thread(target=self.spin_node, daemon=True)
            ros_feed_thread.start()
            update_twist_thread = threading.Thread(target=self.update_twist, daemon=True)
            update_twist_thread.start()

        # Camera publishing over DDS
        if camera:
            self.camera_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "realsense"
            )
            if self.camera_id < 0:
                print("WARNING: 'realsense' camera not found in MuJoCo model, disabling camera publishing")
            else:
                self._cam_w = self.mj_model.cam_resolution[self.camera_id][0]
                self._cam_h = self.mj_model.cam_resolution[self.camera_id][1]

                self._color_pub = ChannelPublisher(TOPIC_REALSENSE_COLOR, SimImage_)
                self._color_pub.Init()
                self._depth_pub = ChannelPublisher(TOPIC_REALSENSE_DEPTH, SimImage_)
                self._depth_pub.Init()

                camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
                camera_thread.start()
                print(f"Camera publishing enabled: {self._cam_w}x{self._cam_h} on {TOPIC_REALSENSE_COLOR}, {TOPIC_REALSENSE_DEPTH}")

    def _camera_loop(self):
        """Render color + depth at ~30Hz and publish over DDS."""
        # Create renderer in this thread — OpenGL context is thread-local
        renderer = mujoco.Renderer(self.mj_model, self._cam_h, self._cam_w)

        interval = 1.0 / 30.0
        while True:
            start = time.perf_counter()

            if self._lock:
                self._lock.acquire()

            renderer.update_scene(self.mj_data, self.camera_id)
            color = renderer.render().copy()

            renderer.enable_depth_rendering()
            renderer.update_scene(self.mj_data, self.camera_id)
            raw_depth = renderer.render().copy()
            renderer.disable_depth_rendering()

            if self._lock:
                self._lock.release()

            # MuJoCo Renderer already returns linear depth in meters
            h, w = raw_depth.shape
            metric_depth = raw_depth.astype(np.float32)

            color_msg = SimImage_(
                height=h, width=w,
                encoding="rgb8",
                data=color.flatten().tolist(),
            )
            depth_mm = (metric_depth * 1000).clip(0, 65535).astype(np.uint16)
            depth_msg = SimImage_(
                height=h, width=w,
                encoding="16UC1",
                data=list(depth_mm.tobytes()),
            )

            self._color_pub.Write(color_msg)
            self._depth_pub.Write(depth_msg)

            elapsed = time.perf_counter() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    
    def update_twist(self):
        while True:
            time.sleep(0.1)
            global x_vel, theta_vel, theta
            
            #print("x position now: " + str(self.mj_data.mocap_pos[0,0]))
            self.mj_data.mocap_pos[0,0] = self.mj_data.mocap_pos[0,0] + (x_vel * 0.005)*np.cos(theta)
            self.mj_data.mocap_pos[0,1] = self.mj_data.mocap_pos[0,1] + (x_vel * 0.005)*np.sin(theta)
            theta = theta + theta_vel * 0.01
            
            q = quaternion.from_euler_angles(0,0,theta)
            #print("q: " + str(quaternion.as_float_array(q)))
            #print("theta:" + str(theta))
            self.mj_data.mocap_quat[0] = quaternion.as_float_array(q)


    def spin_node(self):
        while True:
            time.sleep(0.1)
            rclpy.spin_once(self.g1_twist_cmd_node)

    def LowCmdHandler(self, msg: LowCmd_):
        if self.mj_data != None:
            for i in range(self.num_motor):
                self.mj_data.ctrl[i] = (msg.motor_cmd[i].tau + msg.motor_cmd[i].kp * (msg.motor_cmd[i].q - self.mj_data.sensordata[i]) + msg.motor_cmd[i].kd * (msg.motor_cmd[i].dq- self.mj_data.sensordata[i + self.num_motor + self.num_hand_joint]))

    def LeftHandCmdHandler(self, msg: HandCmd_):
        for i in range(7):
            self.mj_data.ctrl[i+29] = msg.motor_cmd[i].q
        return

    def RightHandCmdHandler(self, msg: HandCmd_):
        for i in range(7):
            self.mj_data.ctrl[i+36] = msg.motor_cmd[i].q
        return

    def PublishLowState(self):
        if self.mj_data != None:
            for i in range(self.num_motor):
                self.low_state.motor_state[i].q = self.mj_data.sensordata[i]
                self.low_state.motor_state[i].dq = self.mj_data.sensordata[i + self.num_motor + self.num_hand_joint]
                self.low_state.motor_state[i].tau_est = self.mj_data.sensordata[i + (2 * (self.num_motor + self.num_hand_joint))]

            if self.have_frame_sensor_:

                self.low_state.imu_state.quaternion[0] = self.mj_data.sensordata[self.dim_motor_sensor + 0]
                self.low_state.imu_state.quaternion[1] = self.mj_data.sensordata[self.dim_motor_sensor + 1]
                self.low_state.imu_state.quaternion[2] = self.mj_data.sensordata[self.dim_motor_sensor + 2]
                self.low_state.imu_state.quaternion[3] = self.mj_data.sensordata[self.dim_motor_sensor + 3]

                self.low_state.imu_state.gyroscope[0] = self.mj_data.sensordata[self.dim_motor_sensor + 4]
                self.low_state.imu_state.gyroscope[1] = self.mj_data.sensordata[self.dim_motor_sensor + 5]
                self.low_state.imu_state.gyroscope[2] = self.mj_data.sensordata[self.dim_motor_sensor + 6]

                self.low_state.imu_state.accelerometer[0] = self.mj_data.sensordata[self.dim_motor_sensor + 7]
                self.low_state.imu_state.accelerometer[1] = self.mj_data.sensordata[self.dim_motor_sensor + 8]
                self.low_state.imu_state.accelerometer[2] = self.mj_data.sensordata[self.dim_motor_sensor + 9]
        self.low_state_puber.Write(self.low_state)


    def PublishRightHandState(self):
        for i in range(7):
                self.right_hand_state.motor_state[i].q = self.mj_data.sensordata[36 + i]
                self.right_hand_state.motor_state[i].dq = self.mj_data.sensordata[79 + i]
                self.right_hand_state.motor_state[i].tau_est = self.mj_data.sensordata[122 + i]
        self.right_hand_state_puber.Write(self.right_hand_state)

    def PublishLeftHandState(self):
        for i in range(7):
                self.left_hand_state.motor_state[i].q = self.mj_data.sensordata[29 + i]
                self.left_hand_state.motor_state[i].dq = self.mj_data.sensordata[72 + i]
                self.left_hand_state.motor_state[i].tau_est = self.mj_data.sensordata[115 + i]
        self.left_hand_state_puber.Write(self.left_hand_state)

    def PublishHighState(self):

        if self.mj_data != None:
            self.high_state.position[0] = self.mj_data.sensordata[
                self.dim_motor_sensor + 10
            ]
            self.high_state.position[1] = self.mj_data.sensordata[
                self.dim_motor_sensor + 11
            ]
            self.high_state.position[2] = self.mj_data.sensordata[
                self.dim_motor_sensor + 12
            ]

            self.high_state.velocity[0] = self.mj_data.sensordata[
                self.dim_motor_sensor + 13
            ]
            self.high_state.velocity[1] = self.mj_data.sensordata[
                self.dim_motor_sensor + 14
            ]
            self.high_state.velocity[2] = self.mj_data.sensordata[
                self.dim_motor_sensor + 15
            ]

        self.high_state_puber.Write(self.high_state)

    def PrintSceneInformation(self):
        print(" ")

        print("<<------------- Link ------------->> ")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                print("link_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Joint ------------->> ")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                print("joint_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Actuator ------------->>")
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name:
                print("actuator_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Sensor ------------->>")
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name:
                print(
                    "sensor_index:",
                    index,
                    ", name:",
                    name,
                    ", dim:",
                    self.mj_model.sensor_dim[i],
                )
            index = index + self.mj_model.sensor_dim[i]
        print(" ")

        print("<<------------- Camera ------------->> ")
        for i in range(self.mj_model.ncam):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_CAMERA, i)
            if name:
                print("camera_index:", i, ", name:", name)
        print(" ")



class ElasticBand:

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
