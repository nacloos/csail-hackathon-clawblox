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
- Runner tmux mode must forward control env overrides into the child shell. A failed `POST_NAV_DEMO_INIT=1` test showed the child silently re-defaulting to `0`; the runner now forwards the harness control vars it exports.
- `PREINIT_LEFT_PARK` must be independent of `PREINIT_RECOVERY`. It was previously nested under the unrelated right-shoulder recovery flag, so `PREINIT_LEFT_PARK=1` logged as enabled but did not run. The harness now executes left parking directly after initial hand setup.
- The old joint-vector left park is unsafe: it aborted at waypoint 901 with max qvel `50 rad/s` and left elbow velocity `23.7 rad/s`.
- Demo-style Cartesian left init pose `[0.15, 0.4, 0.15]` plans cleanly offline with waist/right side fixed (`1.60 rad` left delta, `0.066 rad` max step), but the current custom step-and-wait servo still failed at waypoint 1 (`upper_max_dq=8.87`, left elbow qvel `24.9`, final qvel `65`). This points at executor/control instability for large posture changes, not an IK path problem.
- Isolated executor tests show the init problem is more specific than "servo is slow":
  - Left-only Cartesian init with the demo controller failed after ~12s at joint tracking `0.182 rad` and left residual qvel spiked; scaled q/dq/tau failed earlier (`scale=4`: left elbow qvel `8.75`, `scale=8`: worse, self/body contacts and qvel spikes).
  - Symmetric demo-style both-arm init with identity hand rotations did not spike, but the demo controller aborted immediately on left EE rotation error.
  - `controller.py` now seeds trajectory sample 0 from measured `q_start` and rejects large adjacent joint steps before execution. This removed the first-waypoint IK-branch jump.
  - Post-fix `both_init` with current EE rotations ran ~4.1s before hitting the 5cm Cartesian tracking limit (`50.7mm`), with no hand/table contact. The next safe init experiment should be staged/slower or use init-specific tolerances plus an explicit qvel gate.
  - A low camera/viewer test (`UNITREE_SIM_CAMERA_FPS=0.1`, `UNITREE_VIEWER_DT=1.0`) improved idle sim speed but did not make `both_init` safe; execution aborted immediately on a DDS measured-joint discontinuity and left high qvel. Do not treat camera throttling as a control fix.
  - Conclusion: large Cartesian init moves still need trajectory shaping and stronger state-consistency gates; simply switching executor, scaling q/dq/tau, or throttling rendering is not enough.
- Important temp fix now verified by run 79: `servo_execute()` passes `demo.urdf_model` into `servo_joint_path()` so moving servo commands use gravity compensation.
- Current profiling logs elapsed time for settle waits, external camera capture, perception inference, nav pulses, IK precompute, servo execution, servo attempts, staged pre-pick stages, and total run.
- Clean profiling after the closure fix did not complete pick. It reached the nav gate, then failed in `prepick_stage_up_a0` due table contact from `left_hand_index_1_link`.
- Latest diagnostic run: nav gate after 9 estimates; `prepick_stage_up_a0` planned waist yaw from near `0` to `-0.785 rad` while left-arm joints stayed fixed. The left hand first touched the table at waypoint 40. This confirms the immediate geometry bug: right-arm+waist IK from rest sweeps the fixed left side into/along the table.
- Latest timing evidence: perception averages `21.4 s` per estimate (`6.5 s` capture, `14.9 s` inference); nav pulses average `4.5 s`; `prepick_stage_up_a0` spent `0.8 s` in IK precompute and `283.6 s` in step-and-wait servo execution. The pick-time bottleneck is the custom servo executor, not IK.
- The custom servo is intentionally very slow: `MAX_CMD_STEP=0.0005 rad` at `50 Hz` is only `0.025 rad/s` before tracking waits. Run 79's `prepick_stage_up_a0` advanced 60 seen waypoints over `295 s`; slowest waypoints 45-51 each logged `8-13 s` spans. Increasing this blindly is unsafe; use qvel/contact gates and staged tests.
- `MAX_CMD_STEP=0.001` is not safe for the current no-init staged route. A `STOP_AFTER_PICK_DESCEND=1` benchmark reached `prepick_stage_up_a0` waypoint 50 in `169 s`, but used all 8 velocity recoveries and logged 50 left-hand/table contacts starting at waypoint 40. The run was stopped manually; do not use this as a success candidate.
- Servo progress-loop table contact must abort immediately. The `0.001` benchmark exposed a harness bug where low-error/low-velocity table contact was logged but not fatal because the abort path was nested behind unrelated qerr/velocity thresholds. `verified_pick_place_harness.py` now aborts any non-allowed hand/table contact as soon as it is seen.
- Idle server profiling: default camera/viewer advanced at ~`0.82x` realtime after warmup; `UNITREE_SIM_CAMERA_FPS=0.1` and `UNITREE_VIEWER_DT=1.0` reached ~`0.90x`. Runtime overhead matters, but it is smaller than the step-and-wait executor bottleneck.
- Runner readiness bug fixed: `/observe` can return JSON before the MuJoCo runtime is ready. `run_verified_pick_place.sh` now requires `ready=true`, no runtime error, and nonzero tick/time before launching the harness.
- Controller diagnostics now log machine-readable failure reason/info, IK/execute timing, rotation tracking stats, and DDS-vs-`/observe` consistency. Use these fields before inferring a controller failure from console text.
- Latest left-only demo-controller init tests used `PREINIT_LEFT_EXECUTOR=demo`, `INIT_KEEP_CURRENT_ROT=1`, and clean DDS/`/observe` agreement before planning. `24s` failed at `6.67s` on left rotation tracking (`0.386 rad > 0.3`) with qvel already high; `48s` failed at `33.0s` on joint tracking (`0.266 rad > 0.25`) and had left rotation tracking up to `0.601 rad`. Slowing this same Cartesian left-park target is not sufficient, and relaxing thresholds would hide unsafe dynamics.
- The reference demos do not prove the left init path: `demo/pick.py` defines both init poses, but its executable flow sets `arm="right"` and `init_free_joints=right_q_idx`; `demo/pick_place*.py` are also right-arm init/pick paths with the left arm held fixed/closed.
- Isolated right-arm demo init now has an executor test: `EXECUTOR_TEST=right_init PREINIT_LEFT_EXECUTOR=demo`. `3s` and `10s` failed on the 5cm position-tracking limit with high qvel; `30s` completed once with clean post consistency (`max_q_delta 7e-5`, `max_dq_delta 0.0017`) and max tracking errors `q=0.213 rad`, position `31mm`, rotation `0.170 rad`.
- After adding the controller-internal qvel gate, the same `30s` right-init test fails safely at `0.402 s` with `failure_reason=velocity_tracking`, `dq_idx=11`, `dq_max=8.54 rad/s`. The old IK continuity limit in `controller.py` is `0.1 rad/sample` at `dt=0.01 s`, i.e. up to `10 rad/s`, so it can admit dynamically unsafe plans.
- `RUN_DEMO_INIT=1` wires the same right-arm demo init into the full run before nav, but the first full attempt failed during init on right rotation tracking (`0.434 rad > 0.3`) with high qvel. Treat right demo init as promising but not robust yet; it needs controller-internal qvel gating/recovery before being trusted in the full pick pipeline.
- `DualArmIK.solve()` has a smoothness cost but no hard per-step velocity bound. The next control change should make IK/planning velocity-aware or execute the path through a qvel-gated streaming controller; do not relax tracking thresholds to pass these tests.
- `DualArmIK.solve(max_step=...)` now enforces a hard per-sample bound for free joints. A `1e-6 rad` tolerance is needed in the continuity guard because IPOPT returns clamped steps like `0.010000002` for a `0.01` limit.
- External hold during IK precompute is required. With `DEMO_MAX_DQ_PER_STEP=0.02`, external hold kept precompute stable and right init executed `11.6s` before a qvel abort; without the external hold, the robot drifted/spiked before execution. The in-process hold thread is not reliable during CPU-heavy IPOPT precompute.
- With `DEMO_MAX_DQ_PER_STEP=0.01`, the original `demo/interface.py` right init target `[0.1,-0.3,0.15]` executed `20.7s` before qvel abort (`dq_idx=12`, `dq_max=8.33`) and ended near right-hand/right-hip self-contact. The wider `pick_place_brick_nav.py` target `[0.15,-0.4,0.15]` failed earlier (`0.74s`, `dq_idx=11`, `dq_max=9.40`), so target geometry alone is not the root cause.
- Direct original `TrajectoryController.execute()` on the `pick_place_brick_nav.py` 3s right-init path is not safe in this sim. Clean reset run spent `49.3s` precomputing 150 IK samples, then aborted at execution sample 0 with measured upper-body qvel `72.9 rad/s`; the robot had drifted into severe self-contact during the controller's in-process precompute hold. A prior dirty-state run precomputed in `52.6s` and aborted at `1.23s` on right EE position tracking (`51.4mm > 50mm`). This validates trying the demo path, but the failure is before pick/place.
- Holding the reset upper-body posture for `55s` with the same gravity-compensated DDS target is stable (`max observed qvel <0.001 rad/s`, max joint drift `0.0073 rad`), so the 49s demo failure is not caused by the hold command itself. Capping numeric threads (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`) cuts the same demo precompute from `49.3s` to `1.07s` and avoids precompute drift, but the original 3s motion still aborts at `0.177s` on qvel (`8.36 rad/s` at q index 11). Treat thread caps as required for demo-style IK, then solve trajectory aggressiveness separately.
- External precompute hold must be a separate DDS process and must not call `DDSInterface.shutdown()` at handoff: shutdown publishes `NotUsedJoint0.q=0` and can briefly disable arm SDK, causing a sample-0 velocity spike. Waiting for the hold process to publish before starting IK fixed the worst handoff race. With caps + external hold + `max_dq_per_step=0.02`, the 30s right-init path ran to `13.45s/30s` before qvel `13.36 rad/s`; the 60s path ran to `36.75s/60s` before qvel `14.46 rad/s`. Tracking stayed good (`<=0.165 rad` joint RMSE, `<=15.3mm` pos, `<=0.116 rad` rot), so the remaining issue is right-arm dynamic velocity spikes, not IK or Cartesian tracking.
- `max_dq_per_step=0.01` is not a safe fix for the demo right-init path: it produced early IK position/rotation warnings, then large state disagreement and qvel spikes during precompute. Prefer shaping the Cartesian path or segmenting around the high-velocity region over simply tightening the per-sample IK clamp.
- For the synchronized external-hold 60s right-init path, `DEMO_DIAG_COMMAND_DQ_SCALE=0` plus `DEMO_DIAG_COMMAND_TAU_MODE=gravity` is worse than the default RNEA/dq commands: it aborts at `0.39s` on joint tracking (`0.202 rad > 0.2`) while the default reaches `36.75s` before a velocity spike. RNEA/dq feedforward is not the main source of the late failure.
- Demo executor command variants tested: `DEMO_COMMAND_DQ_SCALE=0` plus `DEMO_COMMAND_TAU_MODE=gravity` still failed at `1.28s` on `dq_idx=11`; lower right gains (`RIGHT_KP=10`, `RIGHT_KD=2`) delayed that to `2.67s` but still failed and left worse DDS/`/observe` disagreement. The next executor needs acceleration/jerk-aware streaming or qvel recovery/reseed, not just zero velocity feedforward or lower gains.
- Hybrid executor status: demo IK precompute plus qvel/contact-gated servo playback is promising but not yet a full pick/place solution. `HYBRID_MAX_CMD_STEP=0.003`, `HYBRID_TRACK_TOL=0.12` failed at waypoint 43, and the old servo recovery made the velocity spike worse. Conservative settings `HYBRID_MAX_CMD_STEP=0.001`, `HYBRID_TRACK_TOL=0.04`, `HYBRID_CMD_HZ=100`, `HYBRID_MAX_RECOVERIES=0` are the current safe right-init baseline.
- Servo waypoint timing bug fixed: `servo_joint_path()` previously slept after clipped micro-steps but not after sending an in-range planned waypoint, so dense demo paths could burst many setpoints faster than `cmd_hz`. A pre-fix right-init rerun jumped through hundreds of waypoints, then spiked to max qvel `23.6 rad/s`. After adding the missing sleep, isolated hybrid right init completed in `105.2s` with max servo qvel `1.30 rad/s`, final command error `0.0119 rad`, no hand-table hits, and clean DDS/`/observe` agreement.
- Runner/harness hybrid defaults now match the conservative isolated success: `HYBRID_MAX_CMD_STEP=0.001`, `HYBRID_TRACK_TOL=0.04`, `HYBRID_UPPER_DQ_RECOVER=100.0`, `HYBRID_MAX_RECOVERIES=0`. These defaults are intentionally safe for profiling; they do not prove full pick/place speed yet.
- Full-run init can now select the executor with `INIT_EXECUTOR`; default remains `demo`, and `INIT_EXECUTOR=hybrid` routes `RUN_DEMO_INIT=1` through the qvel/contact-gated hybrid executor.
- Full-run hybrid init needs measured-posture velocity recovery. Without recovery, `RUN_DEMO_INIT=1 INIT_EXECUTOR=hybrid STOP_AFTER_INIT=1` failed near waypoint 1247 with qvel `11.4 rad/s` and no hand-table contact; with the post-precompute settle gate it could fail even earlier around waypoint 45. Lower right-arm gains delayed but did not solve the failure. With `HYBRID_MAX_CMD_STEP=0.001`, `HYBRID_TRACK_TOL=0.04`, `HYBRID_UPPER_DQ_RECOVER=1.0`, and `HYBRID_MAX_RECOVERIES=4`, full-run hybrid init completed in `140.8s`, used all four recoveries, had max servo qvel `4.22 rad/s`, final command error `0.0122 rad`, no hand-table hits, and clean DDS/`/observe` agreement. This is safer but still slow.
- Slower/tighter `HYBRID_MAX_CMD_STEP=0.0005`, `HYBRID_TRACK_TOL=0.025` stayed stable but was too slow for the efficiency target, reaching only about waypoint 472 after `108s` before the experiment was stopped.
- Step-executor safety checks must run every control tick, not only in throttled progress logs. A spike can cross the recovery and abort thresholds between 1 Hz logs. The harness now checks upper-body DDS velocity on every wait/progress tick and can start recovery immediately.
- Fixed-duration velocity recovery is unsafe; it can return while the robot is still accelerating. Recovery now holds until upper-body velocity settles or aborts if it crosses the unsafe threshold. This prevents silently continuing from a bad recovery, but it does not fully solve init: `HYBRID_MAX_CMD_STEP=0.001` with adaptive recovery still failed nondeterministically when a recovery entered an unstable posture, and `0.00075` also failed around waypoint 593. Higher damping (`RIGHT_KD=4`) made recovery much worse. The remaining issue is not just a threshold; the right-init executor needs a smoother dynamically stable approach or a different recovery/stabilization policy.
- Experimental stream mode exists behind `HYBRID_SERVO_MODE=stream`: it advances along dense demo paths once per command tick instead of sleeping per waypoint. It is not safe yet. Without post-precompute settling, it started with qvel about `1.0 rad/s` and failed at waypoint 0. With the settle handoff, it reached waypoint ~1864 in `82.7s` but still hit an unsafe velocity spike. Reseeding the command to measured posture during velocity hold made the spike worse (`max_qvel` about `68 rad/s`). Keep stream mode experimental; do not use it as the default pick/place executor.
- Offline planning comparison supports the right-arm demo ordering. From rest, staged-up pre-pick needs `0.786 rad` waist yaw and `2.63 rad` right-arm delta. After the demo right-arm init pose, the same stage needs only `0.162 rad` waist yaw and `1.13 rad` right-arm delta; right-only staged-up is also feasible offline.
- `POST_NAV_DEMO_INIT=1` now propagates through tmux, but it aborts before motion because straight nav already leaves the resting left hand in table contact (`left_hand_middle_1_link`). Small yaw and backward pulses move the contact point but do not clear it. The next control experiment should use the demo/time-indexed trajectory controller or scaled q/dq/tau executor for left init before nav, then demo right-arm init before pick.
- Porting this into repo/agent-facing control should preserve the run-79 gates and logging: nav readiness, contacts, max qvel, upper DDS qvel, max command error, carried brick displacement/z lift, release pose, and target distance.
