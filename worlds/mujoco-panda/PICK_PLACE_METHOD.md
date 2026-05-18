# Pick and Place Method Notes

This world can be manipulated reliably using only the normal Panda control API if
the controller plans against live geometry, not just object body centers.

The working method was:

1. Observe the live world before every move with `/observe`.
2. Use the actual finger pad geoms as the gripper contact model:
   - left pad: geom `69`
   - right pad: geom `77`
   - pad midpoint: `(geom69.world_position + geom77.world_position) / 2`
   - closing axis: normalized `geom69.world_position - geom77.world_position`
3. For each target object, read its geom pose and orientation from
   `model.geoms[*].world_position` and `world_xmat`.
4. Choose a horizontal closing axis aligned with a real object face:
   - cube blocks: one of the block's horizontal box axes
   - plank/brick: usually the narrow-width axis, so the fingers pinch across the
     short side rather than along the length
   - cylinders: any stable horizontal side-grasp axis
5. Solve IK for the pad midpoint, not the hand body:
   - position term: pad midpoint to target contact point
   - orientation term: gripper closing axis to target face axis
   - verticality term: pad z-axis approximately downward
6. Move through high-clearance waypoints:
   - open hover over object
   - open descent to side-contact height
   - close gripper
   - lift vertically
   - translate at clearance
   - lower to placement height
   - release
   - retreat
7. Send smooth interpolated `SetControl` commands instead of joint jumps. A
   cubic smoothstep interpolation over hundreds of small control updates worked
   well.
8. Re-observe after every close, lift, place, release, and retreat. If the object
   did not move with the gripper, do not continue blindly. Re-plan the grasp from
   the latest live geometry.

Useful practical details:

- The cube blocks were lifted successfully by targeting the pad midpoint near
  the block center at about `z=0.055`, with the closing axis horizontal and
  aligned to a block face.
- A soft-but-firm close worked better than using only the fully open/closed
  extremes. For cubes, a low gripper target around `5` held well. For wider
  bricks/planks, targets around `145` to `170` were useful.
- Held-object placement was more accurate by measuring the live offset
  `pad_midpoint - held_object_center` after lift, then preserving that offset at
  the placement target.
- For long pieces, verify reachability with several IK seeds. A grasp may be
  reachable from a neutral or task-specific seed even if solving from the current
  high retreat pose fails.
- The most important lesson was to observe all relevant geometry: object geoms,
  finger pad geoms, gripper orientation, and block/brick/plank axes. Body centers
  alone are not enough.
