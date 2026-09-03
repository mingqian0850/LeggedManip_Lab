# B2-W + Z1 Isaac Lab asset

This directory contains a separate wheeled B2-W + Z1 model. It does not
replace or modify the existing footed `assets/b2_z1` model.

## Geometry and joints

- The B2-W base, leg, and wheel descriptions come from Unitree's official
  `unitree_ros` repository at commit
  `7d6075f7f58588b189b940130e3edab3c839b2df`.
- The Z1 links and gripper also come from that repository.
- The four official wheel links and continuous joints are retained, but are
  renamed from Unitree's `*_foot` convention to `*_wheel` for clarity.
- The B2-W body collision box has its upper surface at `z = 0.075 m` in the
  `base_link` frame. The adapter occupies `z = 0.075...0.110 m`, so the Z1
  mounting plane is exactly `0.035 m` above that surface.
- The provisional mount position is `(x, y, z) = (0.211, 0.0, 0.110) m` in
  `base_link`. Only the 35 mm vertical offset is hardware-measured. The XY
  placement and the simplified `130 x 120 mm` adapter footprint should be
  updated after measuring the real adapter or obtaining its CAD.

The adapter is currently a base-attached visual and collision box. Its mass is
not added to the base inertia because the real adapter mass and center of mass
are not yet known.

There is not yet a calibrated TCP frame. Before training pose tracking, measure
the transform from `link6` (or the gripper stator) to the real tool center and
add it as a fixed frame; using `link6` silently as the TCP would introduce a
systematic tracking offset.

## Runtime configuration

`b2w_z1_articulation_cfg.py` exposes two configurations. `B2W_Z1_CFG` uses the
hardware-like stacked Z-fold `[0.0, 0.15, -0.45, 0.30, 0.0, 0.0] rad` for
joints 1--6. This keeps the returning arm and gripper above the lidar. The pose
was reconstructed from a side photograph and is provisional; replace it with
encoder readback from the real robot when available. The arm's all-zero pose is
not a safe stow for this assembled model because it intersects the lidar region
and places joints 2 and 3 at their limits.

`B2W_Z1_TRAINING_CFG` instead resets the arm to the raised, forward-facing
ready pose `[0.0, 1.35, -1.65, 0.30, 0.0, 0.0] rad`. TCP-training tasks should
use this variant so the arm starts deployed and away from joint limits. Both
configurations have independent actuator groups for the 12 leg joints, four
continuous wheel joints, six Z1 joints, and the gripper. The PD gains,
armature, joint friction, and command delay are safe starting values inherited
from the existing project and WheelRL model; they are not identified B2-W + Z1
hardware parameters. Calibrate and/or randomize them before sim-to-real
deployment.

For direct use in Isaac Sim, open `b2w_z1_stow.usda`. It is a lightweight USD
wrapper around the generated `b2w_z1.usd` and authors the same stow joint state
in degrees. Keeping the preset in a wrapper means regenerating the base USD
does not erase it.

Do not reuse the current footed B2 task unchanged. A B2-W task needs explicit
wheel velocity actions and should replace foot-air-time/foot-slide rewards with
wheel contact, rolling, slip, and wheel-speed terms.

## Regeneration

Clone the official source and check out the pinned revision:

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git /tmp/unitree_ros_b2w
git -C /tmp/unitree_ros_b2w checkout 7d6075f7f58588b189b940130e3edab3c839b2df
```

Generate the reviewable URDF and a temporary URDF with absolute mesh paths:

```bash
python source/LeggedManip_Lab/LeggedManip_Lab/assets/b2w_z1/build_combined_urdf.py \
  --unitree-ros-root /tmp/unitree_ros_b2w \
  --output source/LeggedManip_Lab/LeggedManip_Lab/assets/b2w_z1/source/b2w_z1.urdf

python source/LeggedManip_Lab/LeggedManip_Lab/assets/b2w_z1/build_combined_urdf.py \
  --unitree-ros-root /tmp/unitree_ros_b2w \
  --output /tmp/b2w_z1_resolved.urdf \
  --resolve-mesh-paths
```

Convert with the Isaac Lab environment:

```bash
ACCEPT_EULA=Y /home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  /home/mingqian/IsaacLab/scripts/tools/convert_urdf.py \
  /tmp/b2w_z1_resolved.urdf \
  source/LeggedManip_Lab/LeggedManip_Lab/assets/b2w_z1/b2w_z1.usd \
  --joint-target-type none --headless --device cpu
```

Run the structural and physics smoke test:

```bash
ACCEPT_EULA=Y /home/mingqian/miniforge3/envs/env_isaaclab51/bin/python \
  source/LeggedManip_Lab/LeggedManip_Lab/assets/b2w_z1/validate_b2w_z1.py \
  --headless --device cpu
```

The upstream Unitree license is preserved at `source/LICENSE.unitree_ros`.
