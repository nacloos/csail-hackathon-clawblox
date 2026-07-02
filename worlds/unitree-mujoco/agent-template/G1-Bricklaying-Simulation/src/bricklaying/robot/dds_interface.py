"""
To-do's:
- currently implemented for only upper-body control. extend for other uses.
- perhaps pull assumptions such as locked waist out of this file
- understand how torque feedforward works. useful for grasping?
"""

from __future__ import annotations

import time
from threading import Thread, Lock
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, HandCmd_, HandState_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.utils.crc import CRC

from .joint_config import (
    G1JointIndex, G1JointGroup, Dex3JointIndex,
    G1JointConfiguration, LeftDex3JointConfiguration, RightDex3JointConfiguration,
    HAND_OPEN_LEFT, HAND_OPEN_RIGHT,
    HAND_GRASP_LEFT, HAND_GRASP_RIGHT,
    HAND_CLOSED_LEFT, HAND_CLOSED_RIGHT,
)


class DDSInterface:
    """
    Manages DDS communication with the robot.
    Runs subscribers in background threads, provides thread-safe access.
    """
    def __init__(self, network_interface: str, use_nav: bool = False):
        try:
            ChannelFactoryInitialize(0, network_interface)
        except Exception as e:
            print(f"Initialization error: {e}")

        self._use_nav = use_nav

        # Locks for thread-safe read/write
        self.state_lock = Lock()
        self.cmd_lock = Lock()

        # Pub/sub frequency
        self.read_delay = 0.002
        self.write_delay = 0.002

        # State buffers (wait for first message)
        self._low_state = None
        self._left_hand_state = None
        self._right_hand_state = None
        self._nav_state = None

        # Track message timestamps for staleness detection
        self._last_low_state_time: float | None = None
        self._last_hand_state_time: float | None = None
        self._last_nav_state_time: float | None = None

        # Command buffers
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        #self.left_hand_cmd = unitree_hg_msg_dds__HandCmd_()
        #self.right_hand_cmd = unitree_hg_msg_dds__HandCmd_()

        # Checksum for body command
        self.low_cmd_crc = CRC()

        # Constants for cmd messages
        self.upper_body_gains   = G1JointConfiguration.get_gains_arrays(G1JointGroup.UPPER_BODY)
        self.waist_locked_gains = G1JointConfiguration.get_gains_arrays(G1JointGroup.WAIST_LOCKED)
        # Assume left and right gains are the same
        self.hand_gains = LeftDex3JointConfiguration.get_gains_arrays(list(Dex3JointIndex))
        
        # Start
        self.running = True
        
        # Set up thread for subscribing
        self._setup_subscribers()
        
        # Wait for first DDS message
        print("Waiting for robot state messages...")
        start_time = time.time()
        timeout = 3.0  # seconds
        while self._last_low_state_time is None \
            or (self._use_nav and self._last_nav_state_time is None):
            #or self._last_hand_state_time is None \
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    "Failed to receive initial state messages. "
                    "Check that robot is powered on and network connection is correct."
                )
            time.sleep(0.01)
        print("Robot state messages received!")

        # Initialize waist lock target (roll + pitch held at current angle)
        waist_q, _ = self.get_waist_state()
        self.waist_locked_target = waist_q[1:]  # [roll, pitch]

        # Initialize commands from current upper-body state
        ub_q, _ = self.get_upper_body_state()
        self.set_upper_body_target(ub_q, np.zeros_like(ub_q), np.zeros_like(ub_q))
        l, r = self.get_hand_state()
        self.set_hand_target(l, r)

        # Set up thread for publishing
        self._setup_publishers()

    # ===== DDS Interface =====
    
    def _setup_subscribers(self):
        """Create DDS subscribers for state feedback."""
        # Create subscribers
        self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.left_hand_state_subscriber = ChannelSubscriber("rt/dex3/left/state", HandState_)
        self.right_hand_state_subscriber = ChannelSubscriber("rt/dex3/right/state", HandState_)
        self.low_state_subscriber.Init()
        self.left_hand_state_subscriber.Init()
        self.right_hand_state_subscriber.Init()
        if self._use_nav:
            self.nav_state_subscriber = ChannelSubscriber("rt/nav/cur_pos", String_)
            self.nav_state_subscriber.Init()

        # Start subscribing thread
        self.subscribe_thread = Thread(target=self._subscribe_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

    def _setup_publishers(self):
        """Create DDS publishers for commands."""
        # Continuous publishers
        #[FLAG]: Set to arm SDK when running not sim
        #self.low_cmd_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)  # Only accessible when G1 in motion mode
        
        self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.left_hand_cmd_publisher = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self.right_hand_cmd_publisher = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self.low_cmd_publisher.Init()
        self.left_hand_cmd_publisher.Init()
        self.right_hand_cmd_publisher.Init()

        # Discrete publishers
        if self._use_nav:
            self.nav_cmd_publisher = ChannelPublisher("rt/nav/nav_cmd", String_)
            self.nav_cmd_publisher.Init()

        # Start publishing thread
        self.publish_thread = Thread(target=self._publish_cmd)
        self.publish_thread.daemon = True
        self.publish_thread.start()

    def _subscribe_state(self):
        """Thread-safe read from subscribers."""
        while self.running:
            # Read from subscribers
            low_state = self.low_state_subscriber.Read()
            left_hand_state = self.left_hand_state_subscriber.Read()
            right_hand_state = self.right_hand_state_subscriber.Read()
            if self._use_nav:
                nav_state_msg = self.nav_state_subscriber.Read()

            # Write to buffers
            with self.state_lock:
                if low_state is not None:
                    self._low_state = low_state
                    self._last_low_state_time = time.time()
                if left_hand_state is not None:
                    self._left_hand_state = left_hand_state
                    self._last_hand_state_time = time.time()
                if right_hand_state is not None:
                    self._right_hand_state = right_hand_state
                    self._last_hand_state_time = time.time()
                if self._use_nav and nav_state_msg is not None:
                    try:
                        x, y, qz, qw = (float(v) for v in nav_state_msg.data.split(','))
                        self._nav_state = (x, y, qz, qw)
                        self._last_nav_state_time = time.time()
                    except Exception as e:
                        print(f"DDSInterface: failed to parse nav_state message: {e}")

            # Thread delay
            time.sleep(self.read_delay)

    def _publish_cmd(self):
        """Thread-safe write to publishers."""
        while self.running:
            # Publish from buffers
            with self.cmd_lock:
                self.low_cmd_publisher.Write(self.low_cmd)
                self.left_hand_cmd_publisher.Write(self.left_hand_cmd)
                self.right_hand_cmd_publisher.Write(self.right_hand_cmd)
            
            # Thread delay
            time.sleep(self.write_delay)

    # ===== Command Generation =====

    def _build_low_cmd(self, ub_q, ub_dq, ub_tau):
        """Create low cmd message for publishing."""
        cmd = unitree_hg_msg_dds__LowCmd_()

        # Enable arm SDK
        cmd.motor_cmd[G1JointIndex.NotUsedJoint0].q = 1

        # Write upper-body joints (waist_yaw + both arms = 15 DOF)
        for i, joint in enumerate(G1JointGroup.UPPER_BODY):
            cmd.motor_cmd[joint].mode = 1
            cmd.motor_cmd[joint].q    = ub_q[i]
            cmd.motor_cmd[joint].dq   = ub_dq[i]
            cmd.motor_cmd[joint].tau  = ub_tau[i]
            cmd.motor_cmd[joint].kp   = self.upper_body_gains['kp'][i]
            cmd.motor_cmd[joint].kd   = self.upper_body_gains['kd'][i]

        # Hold waist roll + pitch at locked target
        for i, joint in enumerate(G1JointGroup.WAIST_LOCKED):
            cmd.motor_cmd[joint].mode = 1
            cmd.motor_cmd[joint].q    = self.waist_locked_target[i]
            cmd.motor_cmd[joint].dq   = 0
            cmd.motor_cmd[joint].tau  = 0
            cmd.motor_cmd[joint].kp   = self.waist_locked_gains['kp'][i]
            cmd.motor_cmd[joint].kd   = self.waist_locked_gains['kd'][i]

        # Compute checksum
        cmd.crc = self.low_cmd_crc.Crc(cmd)

        return cmd

    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode
    
    def _build_hand_cmd(self, hand_q_target):
        """Create hand cmd message for publishing."""
        cmd = unitree_hg_msg_dds__HandCmd_()

        # Write to hand joints
        for i, joint in enumerate(list(Dex3JointIndex)):
            ris_mode = self._RIS_Mode(id=joint, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            cmd.motor_cmd[joint].mode = motor_mode
            cmd.motor_cmd[joint].q    = hand_q_target[i]
            cmd.motor_cmd[joint].dq   = 0.0
            cmd.motor_cmd[joint].tau  = 0.0
            cmd.motor_cmd[joint].kp   = self.hand_gains['kp'][i]
            cmd.motor_cmd[joint].kd   = self.hand_gains['kd'][i]

        return cmd

    # ===== External Interface =====

    def is_state_healthy(self, max_age: float = 0.1) -> bool:
        """Check if we're receiving fresh state data."""
        now = time.time()
        return (
            (now - self._last_low_state_time) < max_age and
            (now - self._last_hand_state_time) < max_age
        )
    
    def get_waist_state(self):
        with self.state_lock:
            if self._low_state is None:
                return None, None
            waist_q = np.array([self._low_state.motor_state[j].q for j in G1JointGroup.WAIST])
            waist_dq = np.array([self._low_state.motor_state[j].dq for j in G1JointGroup.WAIST])
        return waist_q, waist_dq

    def get_arm_state(self):
        with self.state_lock:
            if self._low_state is None:
                return None, None
            arm_q = np.array([self._low_state.motor_state[j].q for j in G1JointGroup.BOTH_ARMS])
            arm_dq = np.array([self._low_state.motor_state[j].dq for j in G1JointGroup.BOTH_ARMS])
        return arm_q, arm_dq

    def get_upper_body_state(self):
        with self.state_lock:
            if self._low_state is None:
                return None, None
            q  = np.array([self._low_state.motor_state[j].q  for j in G1JointGroup.UPPER_BODY])
            dq = np.array([self._low_state.motor_state[j].dq for j in G1JointGroup.UPPER_BODY])
        return q, dq

    def get_hand_state(self):
        with self.state_lock:
            if self._left_hand_state is None or self._right_hand_state is None:
                return None, None
            left_q = np.array([m.q for m in self._left_hand_state.motor_state])
            right_q = np.array([m.q for m in self._right_hand_state.motor_state])
        return left_q, right_q

    def set_upper_body_target(self, ub_q, ub_dq, ub_tau):
        """Write 15-element upper-body command [waist_yaw, left×7, right×7] to DDS buffer."""
        ub_q_safe   = G1JointConfiguration.clamp_positions( ub_q,  G1JointGroup.UPPER_BODY)
        ub_dq_safe  = G1JointConfiguration.clamp_velocities(ub_dq, G1JointGroup.UPPER_BODY)
        ub_tau_safe = G1JointConfiguration.clamp_torques(   ub_tau, G1JointGroup.UPPER_BODY)

        with self.cmd_lock:
            self.low_cmd = self._build_low_cmd(ub_q_safe, ub_dq_safe, ub_tau_safe)

    def set_hand_target(self, left_hand_q_target, right_hand_q_target):
        """Write target values to hand cmd buffers."""
        # Pre-process target
        left_hand_q_safe = LeftDex3JointConfiguration.clamp_positions(
            left_hand_q_target, list(Dex3JointIndex)
        )
        right_hand_q_safe = RightDex3JointConfiguration.clamp_positions(
            right_hand_q_target, list(Dex3JointIndex)
        )
        # TODO incorporate dq? tau?

        # Build commands
        left_hand_cmd = self._build_hand_cmd(left_hand_q_safe)
        right_hand_cmd = self._build_hand_cmd(right_hand_q_safe)

        # Write to buffers
        with self.cmd_lock:
            self.left_hand_cmd = left_hand_cmd
            self.right_hand_cmd = right_hand_cmd

    def _require_nav(self):
        if not self._use_nav:
            raise RuntimeError("Nav is not enabled. Pass use_nav=True to DDSInterface.")

    def get_nav_state(self) -> tuple[float, float, float, float]:
        """Return the latest (x, y, qz, qw) pose from the nav stack."""
        self._require_nav()
        with self.state_lock:
            return self._nav_state

    def is_nav_healthy(self, max_age: float = 0.5) -> bool:
        """True if a nav pose message was received within max_age seconds."""
        self._require_nav()
        with self.state_lock:
            if self._last_nav_state_time is None:
                return False
            return (time.time() - self._last_nav_state_time) < max_age

    def send_nav_goto(self, x: float, y: float, qz: float, qw: float):
        """Send a GOTO command to the nav stack."""
        self._require_nav()
        self.nav_cmd_publisher.Write(String_(f"GOTO:{x},{y},{qz},{qw}"))

    def send_nav_step(self, direction: str):
        """Send a STEP command (RIGHT, LEFT, FORWARD, BACK)."""
        self._require_nav()
        self.nav_cmd_publisher.Write(String_(f"STEP:{direction}"))

    def send_nav_turn(self, direction: str):
        """Send a TURN command (RIGHT, LEFT)."""
        self._require_nav()
        self.nav_cmd_publisher.Write(String_(f"TURN:{direction}"))

    def set_hand_mode(self, left: str, right: str):
        """
        Command hands by named mode.

        Args:
            left:  Mode for left hand  — "open", "grasp", or "close"
            right: Mode for right hand — "open", "grasp", or "close"
        """
        _left = {
            "open":  HAND_OPEN_LEFT,
            "grasp": HAND_GRASP_LEFT,
            "close": HAND_CLOSED_LEFT,
        }[left]
        _right = {
            "open":  HAND_OPEN_RIGHT,
            "grasp": HAND_GRASP_RIGHT,
            "close": HAND_CLOSED_RIGHT,
        }[right]
        self.set_hand_target(_left, _right)

    def shutdown(self):
        """Clean shutdown of DDS interface"""
        print("Shutting down DDSInterface...")
        input("Press Enter to shut down... \n[CAREFUL! ARE JOINTS HOME? IF NOT, GO TO DAMPING FIRST!]")
        input("Are you sure? Press Enter to confirm...")

        # Stop background threads
        self.running = False
        self.publish_thread.join(timeout=1.0)
        self.subscribe_thread.join(timeout=1.0)

        # Publish cleanup message
        try:
            # 0: Disable arm_sdk
            self.low_cmd.motor_cmd[G1JointIndex.NotUsedJoint0].q = 0
            self.low_cmd.crc = self.low_cmd_crc.Crc(self.low_cmd)
            self.low_cmd_publisher.Write(self.low_cmd)
            time.sleep(0.1)
        except Exception as e:
            print(f"Error during safe shutdown: {e}")

        # DDS cleanup happens automatically on __del__


if __name__ == "__main__":
    import sys

    NETWORK_INTERFACE = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    print(f"Testing DDSInterface on network interface: {NETWORK_INTERFACE}")

    try:
        dds = DDSInterface(NETWORK_INTERFACE)
    except TimeoutError as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Test 1: Check connection and read initial states
    # ----------------------------------------------------------------
    print("\n--- Connection Test ---")

    # Check state health
    print(f"State healthy: {dds.is_state_healthy()}")

    # Read upper-body state (waist_yaw + both arms = 15 DOF)
    ub_q, ub_dq = dds.get_upper_body_state()
    print(f"Upper-body positions (rad): {np.round(ub_q, 3)}")   # [waist_yaw, L×7, R×7]
    print(f"Upper-body velocities (rad/s): {np.round(ub_dq, 3)}")

    # Read hand state
    left_q, right_q = dds.get_hand_state()
    print(f"Left hand positions (rad): {np.round(left_q, 3)}")
    print(f"Right hand positions (rad): {np.round(right_q, 3)}")

    # Read waist state (all 3 axes, for diagnostics)
    waist_q, waist_dq = dds.get_waist_state()
    print(f"Waist positions (rad): {np.round(waist_q, 3)}")

    # ----------------------------------------------------------------
    # Test 2: Publish current states to arms and hands
    # ----------------------------------------------------------------
    print("\n--- Empty Commands Test ---")

    # Publish current upper-body state as target (should hold position)
    dds.set_upper_body_target(ub_q, np.zeros_like(ub_q), np.zeros_like(ub_q))

    # Publish current hand state as target (should hold position)
    dds.set_hand_target(left_q, right_q)

    print("Published current states as targets. The robot should hold its current pose for 3 seconds.")
    time.sleep(3)

    # ----------------------------------------------------------------
    # Test 3: Actuate hands to open and close
    # ----------------------------------------------------------------
    print("\n--- Hand Actuation Test ---")

    print("Opening hands (3s)...")
    dds.set_hand_mode("open", "open")
    time.sleep(2)

    print("Grasping (3s)...")
    dds.set_hand_mode("grasp", "grasp")
    time.sleep(2)

    print("Closing hands fully (3s)...")
    dds.set_hand_mode("close", "close")
    time.sleep(2) 

    print("\nTest complete.")
    dds.shutdown()

