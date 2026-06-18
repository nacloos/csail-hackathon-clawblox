import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_bricklaying.modules.unitree import Unitree


# Hardcoding
x_place = 0.525
x_place_side = 0.50
y_place_left = 0.235
y_place_middle = 0.0
y_place_right = -0.235
y_place_midleft = 0.1175
y_place_midright = -0.1175
z_place_bottom = 0.10
z_place_middle = 0.175
z_place_top = 0.25
# ordered_positions = [
#     [x_place_side, y_place_left, z_place_bottom],
#     [x_place, y_place_middle, z_place_bottom],
#     [x_place_side, y_place_right, z_place_bottom],
#     [x_place_side, y_place_left, z_place_middle],
#     [x_place, y_place_middle, z_place_middle],
#     [x_place_side, y_place_right, z_place_middle],
#     [x_place_side, y_place_left, z_place_top],
#     [x_place, y_place_middle, z_place_top],
#     [x_place_side, y_place_right, z_place_top],
# ]
ordered_positions = [
    [x_place_side, y_place_left, z_place_bottom],
    [x_place, y_place_middle, z_place_bottom],
    [x_place_side, y_place_right, z_place_bottom],
    [x_place_side, y_place_midleft, z_place_middle],
    [x_place_side, y_place_midright, z_place_middle],
    [x_place_side, y_place_middle, z_place_top],
]
# ordered_colors = [
#     "red",
#     "red",
#     "red",
#     "red",
#     "red",
#     "red",
#     "red",
#     "red",
#     "red",
# ]
ordered_colors = [
    "red",
    "red",
    "red",
    "red",
    "red",
    "red",
]
x_world_to_exclusion_W = 0.4


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} networkInterface")
        print("Example: eno1")
        sys.exit(1)
    
    # Init network interface
    network_interface = sys.argv[1]
    try:
        ChannelFactoryInitialize(0, network_interface)
    except Exception as e:
        print(f"Initialization error: {e}")
        sys.exit(1)

    # Initialize robot
    robot_id = 164
    unitree = Unitree(robot_id)
    unitree.move_home()
    input("Press Enter to continue...")
    # Iterate through stacking steps
    for p_place, color in zip(ordered_positions, ordered_colors):
        
        T_inclusions = []
        while not T_inclusions:
            print("Preparing to grasp brick.")
            input("Press Enter to continue...")

            # Estimate brick candidates
            Ts, colors = unitree.estimate_bricks()

            T_inclusions = [
                Ts[i] for i in range(len(Ts)) if Ts[i][0, 3] < x_world_to_exclusion_W and colors[i] == color
            ]

            if not T_inclusions:
                print(f"No matching bricks found for color {color}. Retrying...")

        # Compute pick/place poses
        T_pick = T_inclusions[0]  # Just pick first one
        T_place = np.eye(4)
        T_place[:3, 3] = p_place

        # Execute
        unitree.pick_and_place(T_pick, T_place)

    # Safe shutdown
    unitree.shutdown()

