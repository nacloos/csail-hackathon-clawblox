"""
Unitree G1 Robot Terminal Interface

This script provides terminal-based interface to the high-level API for the Unitree G1 robot, 
using text commands to control robot movements and mode changes.

Usage: python3 g1_terminal_interface.py [network_interface]

For example, if running code from ground station, link name should be the
    port connecting to the G1 router, eg 'eno1'
"""
import time
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


class G1TerminalInterface:
    def __init__(self, network_interface):
        """Initialize the G1 terminal robot interface."""
        try:
            ChannelFactoryInitialize(0, network_interface)
            self.loco_client = LocoClient()
            self.loco_client.SetTimeout(10.0)
            self.loco_client.Init()
            print("G1 robot connection initialized successfully")
        except Exception as e:
            print(f"Failed to initialize G1 robot connection: {e}")
            sys.exit(1)

        self.network_interface = network_interface
        self.running = True
        self.damping_mode()
        
        # Movement parameters
        self.move_speed = 0.3
        self.turn_speed = 0.3
    
    def print_commands(self):
        header = "=" * 70
        print(f"\n{header}")
        print("UNITREE G1 ROBOT TERMINAL CONTROLLER".center(70))
        print(header)

        commands = {
            "damping": "Damping Mode (L2+A)",
            "zero": "Zero Torque Mode (L2+B)",
            "ready": "Ready Mode (L2+Up)",
            "motion": "Motion Mode (R1+X)",
            "forward": "Move Forward",
            "backward": "Move Backward",
            "left": "Turn Left",
            "right": "Turn Right",
            "stop": "Stop Movement",
            "shake": "Shake Hand",
            "help": "Show this help menu",
            "quit": "Exit the controller",
        }

        for cmd, desc in commands.items():
            print(f"  {cmd:<12} - {desc}")

        print(f"{header}")
        print(f"Current Mode: {self.current_mode.upper()}")
        print(header)
    
    def damping_mode(self):
        try:
            print("Setting Damping Mode...")
            self.loco_client.Damp()
            self.current_mode = "damping"
            print("Damping Mode activated")
        except Exception as e:
            print(f"Error entering Damping Mode: {e}")

    def zero_torque_mode(self):
        if self.current_mode == "damping":
            try:
                print("Setting Zero Torque Mode...")
                self.loco_client.ZeroTorque()
                self.current_mode = "zero"
                print("Zero Torque Mode activated")
            except Exception as e:
                print(f"Error entering Zero Torque Mode: {e}")
        else:
            print(f"Current mode must be `damping`, not {self.current_mode}.")

    def ready_mode(self):
        if self.current_mode == "damping":
            try:
                print("Setting Ready Mode...")
                self.loco_client.SetFsmId(4)
                self.current_mode = "ready"
                print("Ready Mode activated")
            except Exception as e:
                print(f"Error entering Ready Mode: {e}")
        else:
            print(f"Current mode must be `damping`, not {self.current_mode}.")
    
    def motion_mode(self):
        if self.current_mode == "ready":
            try:
                print("Entering Motion Mode...")
                self.loco_client.SetFsmId(500)
                self.current_mode = "motion"
                print("Motion Mode activated")
            except Exception as e:
                print(f"Error entering Motion Mode: {e}")
        else:
            print(f"Current mode must be `ready`, not {self.current_mode}.")
    
    def move_forward(self):
        try:
            print("Moving Forward...")
            self.loco_client.Move(self.move_speed, 0, 0)
        except Exception as e:
            print(f"Error moving forward: {e}")
    
    def move_backward(self):
        try:
            print("Moving Backward...")
            self.loco_client.Move(-self.move_speed, 0, 0)
        except Exception as e:
            print(f"Error moving backward: {e}")
    
    def turn_left(self):
        try:
            print("Turning Left...")
            self.loco_client.Move(0, 0, self.turn_speed)
        except Exception as e:
            print(f"Error turning left: {e}")
    
    def turn_right(self):
        try:
            print("Turning Right...")
            self.loco_client.Move(0, 0, -self.turn_speed)
        except Exception as e:
            print(f"Error turning right: {e}")
    
    def stop_movement(self):
        try:
            print("Stopping Movement...")
            self.loco_client.StopMove()
            print("Movement stopped")
        except Exception as e:
            print(f"Error stopping movement: {e}")
    
    def shake_hand(self):
        try:
            print("Shaking Hand...")
            self.loco_client.ShakeHand()
            time.sleep(7.)
            self.loco_client.ShakeHand()
            print("Hand shake completed")
        except Exception as e:
            print(f"Error shaking hand: {e}")
    
    def process_command(self, command):
        command = command.strip().lower()
        
        if command == "damping":
            self.damping_mode()
        elif command == "zero":
            self.zero_torque_mode()
        elif command == "ready":
            self.ready_mode()
        elif command == "motion":
            self.motion_mode()
        elif command == "forward":
            self.move_forward()
        elif command == "backward":
            self.move_backward()
        elif command == "left":
            self.turn_left()
        elif command == "right":
            self.turn_right()
        elif command == "stop":
            self.stop_movement()
        elif command == "shake":
            self.shake_hand()
        elif command == "help":
            self.print_commands()
        elif command in ["quit", "exit"]:
            print("Exiting...")
            self.running = False
        else:
            print(f"Unknown command: {command}")
            print("Type 'help' for available commands")
    
    def run(self):
        print("WARNING: Please ensure there are no obstacles around the G1 robot!")
        input("Press Enter to continue...")
        self.print_commands()
        try:
            while self.running:
                command = input(f"\nG1[{self.current_mode}]> ").strip()
                if command:
                    self.process_command(command)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.running = False
        try:
            if self.loco_client:
                self.loco_client.Damp()  # Return to safe mode
                print("G1 robot returned to damping mode")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(0.5)
        print("G1 Controller shutdown complete")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        print("Example: python3 g1_terminal_interface.py eno1")
        sys.exit(1)
    
    network_interface = sys.argv[1]
    
    try:
        interface = G1TerminalInterface(network_interface)
        interface.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
