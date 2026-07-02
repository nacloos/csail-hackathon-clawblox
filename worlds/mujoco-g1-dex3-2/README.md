# MuJoCo G1 Dex3 World Notes

This world uses `models/g1/scene_hands_modified.xml`, which includes
`models/g1/g1_with_hands_modified.xml`.

## Free-Base Walking Variant

This `-2` copy removes the mocap `base_link` wrapper and the equality weld that
fixed the pelvis in space. The pelvis is now the root body and has a
`freejoint`, so the robot can fall, balance, walk, and move through contact with
the floor.

The root freejoint is passive. The agent still controls all 43 robot actuators:
12 leg torques, 3 waist torques, 14 arm/wrist torques, and 14 Dex3 hand position
targets. Walking must emerge from those actuator commands and contact physics.

The initial pelvis height is lowered so the feet start on the ground instead of
hovering above it.

## Local Hand Actuator Fix

The left Dex3 hand index and middle finger joints are mirrored relative to the
right hand: their flexion ranges are negative. The original position actuator
`ctrlrange` values for those left fingers were positive, so their valid overlap
with the joint ranges was only `0`. As a result, commanding the left
index/middle fingers did not close them in simulation.

`models/g1/g1_with_hands_modified.xml` has been updated so the affected left
hand position actuator ranges match their joint ranges:

- `left_hand_thumb_1_joint`: `-0.724312 1.0472`
- `left_hand_thumb_2_joint`: `0 1.74533`
- `left_hand_middle_0_joint`: `-1.5708 0`
- `left_hand_middle_1_joint`: `-1.74533 0`
- `left_hand_index_0_joint`: `-1.5708 0`
- `left_hand_index_1_joint`: `-1.74533 0`

After this change, the left index/middle fingers close with negative position
commands, mirroring the right hand closing with positive commands.

## Brick Mass Adjustment

`models/g1/bricklaying_props.xml` now sets each brick mass to `0.08 kg`.
The previous value was `0.005 kg`, which is unrealistically light for a
`0.09 m x 0.20 m x 0.06 m` toy brick and made contact behavior easier to
destabilize. The new value represents a lightweight hollow plastic toy brick,
not a solid plastic block.

## Hard Contact Defaults

Collision geoms now use harder contact settings, `solref="0.004 1"` and
`solimp="0.99 0.995 0.0001 0.5 2"`. This reduces the large hand-brick and
table-brick penetrations observed with the default softer contacts. The values
are applied to the robot collision default and directly to the brick geoms.
