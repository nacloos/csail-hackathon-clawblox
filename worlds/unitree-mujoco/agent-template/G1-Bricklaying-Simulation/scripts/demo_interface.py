import numpy as np
import time
from flask import Flask, render_template, request

from modules.unitree import Unitree

app = Flask(__name__)

# Hardcoded placement grid (wrt robot viewpoint)
ordered_positions = [
    [0.47,  0.235, 0.08],  # bottom left
    [0.50,  0.000, 0.08],  # bottom middle
    [0.47, -0.235, 0.08],  # bottom right
    [0.47,  0.235, 0.15],  # middle left
    [0.50,  0.000, 0.15],  # middle middle
    [0.47, -0.235, 0.15],  # middle right
    [0.47,  0.235, 0.22],  # top left
    [0.50,  0.000, 0.22],  # top middle
    [0.47, -0.235, 0.22],  # top right
]

x_world_to_exclusion_W = 0.375

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/server_process_layout/', methods=['POST'])
def server_process_layout():
    data = request.get_json()
    brick_values = data["brick_values"]
    print("Received layout:", brick_values)

    # Iterate through GUI instructions
    for i, brick_code in enumerate(brick_values):
        if brick_code == 0:
            continue  # empty slot, skip
        target_color = "red" if brick_code == 2 else "gray"
        p_place = ordered_positions[i]

        # --- Pick step ---
        T_inclusions = []
        while not T_inclusions:
            print(f"Looking for {target_color} brick...")
            Ts, colors = unitree.estimate_bricks()
            T_inclusions = [
                Ts[j] for j in range(len(Ts)) if Ts[j][0, 3] < x_world_to_exclusion_W and colors[j] == target_color
            ]
            if not T_inclusions:
                print(f"No matching {target_color} bricks, retrying...")

        T_pick = T_inclusions[0]  # pick first candidate

        # --- Place step ---
        T_place = np.eye(4)
        T_place[:3, 3] = p_place

        # Execute
        unitree.pick_and_place(T_pick, T_place)
        
        # Wait for brick re-load
        wait_duration = 5.
        print(f"Re-load brick now! Waiting {wait_duration} seconds.")
        time.sleep(wait_duration)

    return 'Trajectory completed successfully'

#Robot setup
robot_id = 165
unitree = Unitree(robot_id)
unitree.move_home()
try:
    app.run(use_reloader=False, host="0.0.0.0")
finally:
    unitree.shutdown()
