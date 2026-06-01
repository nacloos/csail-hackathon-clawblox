# Unitree MuJoCo Notes

Tight working knowledge for the current non-Docker Unitree MuJoCo setup.

## Runtime

- Launch with `ROS_LOCALHOST_ONLY=1`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `ROS_LOG_DIR=/tmp/ros_logs`, `YOLO_CONFIG_DIR=/tmp/ultralytics`, and `MPLCONFIGDIR=/tmp/matplotlib`.
- Use `--enable-cmd-vel` for base motion. This moves the MuJoCo mocap base through ROS2 `cmd_vel`; it is not a walking controller.
- Current spectator URL is `http://127.0.0.1:19140/`.
- Camera publishing must stay out of the simulator hot loop. Current bridge renders from a copied `MjData` snapshot and writes camera DDS from a subprocess.
- Avoid heavy `/observe` polling during control. Sparse qvel/contact snapshots are useful; aggressive polling perturbs timing.

## Control Rules

- Keep one DDS control owner through init, navigation, perception, pick, and place. Overlapping publishers or stale hold targets corrupt conclusions.
- Hold the upper body continuously while planning, perception, and `cmd_vel` navigation run. Holds need gravity compensation; position plus zero torque causes lag and spikes.
- Use right arm plus waist for pick/place IK (`PICK_FREE_MODE=right_waist`), matching `demo/interface.py`.
- Keep the left hand closed and skip ad hoc left-arm parking for the current successful route (`PREINIT_LEFT_PARK=0`, `LEFT_HAND_MODE=close`). Left parking was nondeterministic and sometimes destabilized before navigation.
- Do not route a grasped brick through demo init. Carry directly to place; earlier pick-return with payload caused large qvel spikes.
- Staged pre-pick is the working route: current hand up to safe z, lateral over target, rotate at safe z, settle to pre-pick, then descend.
- Moving arm execution must use gravity feedforward. The key bug before run 79 was that the custom servo path supported `urdf_model` but `servo_execute()` did not pass it, so pre-pick ran with zero torque feedforward.
- Clean rerun after run 79 failed at `prepick_stage_up`: right wrist pitch velocity spiked (`max_qvel=8.91`, `upper_max_dq=8.08`) around waypoint 10 although the IK path matched the successful run. Current temp harness adds servo velocity recovery: hold measured posture with gravity compensation when upper-body velocity exceeds the recovery threshold, reset command to measured state, and continue/retry. Runner now uses `PREPICK_STAGE_ATTEMPTS=3`.
- Success must require real carry evidence, not only final target distance: check picked brick displacement, z lift, hand/brick contacts, release pose, and target distance in a consistent world frame.

## Verified Success

- Run 79 is the first strict success for the current approach.
- Log: `/tmp/codex_servo_pick_attempt_run79_success.jsonl`.
- Settings:
  - `PREPICK_ROUTE_MODE=staged`
  - `PICK_TRACK_TOL=0.18`
  - `NAV_TARGET_SD=0.0095`
  - `NAV_TARGET_X=0.43`
  - `IK_ABORT_STEP=0.35`
  - `PREPICK_STAGE_ATTEMPTS=1`
  - `PREPICK_ROUTE_CLEARANCE=0.18`
  - `PREPICK_HIGH_Z=0.25`
  - `PICK_FREE_MODE=right_waist`
  - `PREINIT_LEFT_PARK=0`
  - `LEFT_HAND_MODE=close`
  - `RIGHT_KP=25`, `RIGHT_KD=1`
  - `EXTERNAL_HOLD_KP=25`, `EXTERNAL_HOLD_KD=1`
  - `PLACE_TARGET_ROW='0.09,-0.04,0.09,0,0,-25'`
- Evidence:
  - Navigation reached `pose_x=0.4264`, `sd=0.00893`, with quiet nav hold.
  - Staged pre-pick completed: up max qvel/error `0.81/0.043`, over `2.31/0.054`, rotate `1.23/0.036`, settle `0.34/0.029`.
  - `pick_descend` completed: max qvel `0.46`, max command error `0.031`.
  - `grasped` contacts included brick 1 with right thumb, middle, and index links.
  - `post_grasp_lift` moved brick 1 by `0.183 m`, z lift `0.176 m`, `carried_ok=true`.
  - `place` completed and release left brick 1 on the table at `[-0.568, 0.057, 0.959]`.
  - Final target distance was `0.115 m`, under `SUCCESS_TARGET_DIST=0.18`.

## Debug History

- Base motion is not the blocker. Adaptive `cmd_vel` pulses repeatedly reached the pick gate with negligible brick displacement.
- Demo-style direct pre-pick from rest often swept too low or destabilized after navigation.
- Segmented/staged routing reduced table collisions, but without gravity feedforward it stalled on right shoulder roll near `0.18 rad`.
- Raising tracking tolerance to `0.30` was unsafe: it hid lag until upper-body qvel spiked.
- Higher gains (`40/2`, and stronger all-upper tests) made instability worse.
- Blind joint-final interpolation for pre-pick was unsafe; IK endpoints could require large joint deltas and hit the table.
- The old false-positive run 54 placed near the target without a real carry. Do not accept target distance alone.

## Temp Harness

- Current experimental harness is `./verified_pick_place_harness.py`; it is a checkpoint of the temp harness, not production code.
- Convenience runner: `./run_verified_pick_place.sh` from `worlds/unitree-mujoco` opens a two-pane tmux session: fresh server on the left, verified harness on the right. By default it stops any existing process holding the configured API/spectator ports before launch because stale state can leave bricks on the floor. Use `--no-tmux` for current-shell flow, and `--no-server` only when managing the server separately.
- Important temp fix now verified by run 79: `servo_execute()` passes `demo.urdf_model` into `servo_joint_path()` so moving servo commands use gravity compensation.
- Porting this into repo/agent-facing control should preserve the run-79 gates and logging: nav readiness, contacts, max qvel, upper DDS qvel, max command error, carried brick displacement/z lift, release pose, and target distance.
