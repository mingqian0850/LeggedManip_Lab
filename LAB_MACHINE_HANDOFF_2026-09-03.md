# B2 + Z1 Isaac Lab machine handoff

Last updated: 2026-09-03 (Europe/Berlin)

> This is the self-contained machine, repository and asset handoff for the
> Isaac Lab checkout. The companion research discussion remains in the
> separate WheelRL repository; the executable Isaac training plan is
> [`docs/B2W_Z1_TCP_TRAINING_PLAN.md`](docs/B2W_Z1_TCP_TRAINING_PLAN.md).

## Start here

### 2026-09-03 B2-W + Z1 implementation update

This update supersedes the older statements below that the Isaac asset still
has no wheels or that robot identity must be selected before model work. The
user selected **B2-W + Z1** and supplied hardware photographs of the 35 mm Z1
adapter and the real folded-arm pose.

A separate reproducible B2-W + Z1 asset now exists at:

```text
source/LeggedManip_Lab/LeggedManip_Lab/assets/b2w_z1/
```

It is derived from Unitree `unitree_ros` commit
`7d6075f7f58588b189b940130e3edab3c839b2df`, contains 23 movable joints
(12 legs, four wheels, six Z1 joints and one gripper joint), and keeps the
original footed B2 + Z1 asset unchanged. The Z1 mounting plane is 35 mm above
the B2-W deck. Mount XY, adapter mass/inertia and the exact TCP still require
measurement.

The corrected hardware-like stow is
`[0.00, 0.15, -0.45, 0.30, 0.00, 0.00] rad`. It replaces the unsafe all-zero
visual fold for generic resets. A separate raised training-ready pose remains
`[0.00, 1.35, -1.65, 0.30, 0.00, 0.00] rad`. Direct Isaac Sim users should
open `assets/b2w_z1/b2w_z1_stow.usda`.

Validation completed:

- one articulation and 23 movable joints;
- four wheel joints rotating about their expected Y axes;
- 35 mm adapter height;
- direct-open USD and Isaac Lab reset presets agree;
- approximately 109.8 mm vertical clearance between the limiting distal
  gripper-stator collision mesh and the modeled lidar top;
- 250 physics steps passed with base height 0.538 m and upright score 1.000.

The stow angles are reconstructed from a single photograph and must eventually
be replaced with encoder readback. Self-collisions remain disabled, `link1` and
`link5` have no collision geometry, and the simplified lidar/adapter envelopes
do not include cables or hardware tolerances.

The next work item is task/controller implementation followed by staged
training, not an immediate 4,096-environment run. The detailed current plan is
[`docs/B2W_Z1_TCP_TRAINING_PLAN.md`](docs/B2W_Z1_TCP_TRAINING_PLAN.md). In
particular, train on a nominal plane first; deployment-matched mild terrain is
a late curriculum stage after flat tracking and dynamics randomization pass.

This file records the state found on the native Ubuntu RTX 4090 experiment
machine. It is intentionally separate from `HANDOFF.md`: that document records
the WheelRL/MuJoCo research direction, while this document records the actual
Isaac installation, the existing `LeggedManip_Lab` baseline, and the robot
model gates that must be resolved before another training run.

The immediate task for the next Codex session is **not long training**. B2-W is
now selected and the first asset gate has passed. Next calibrate the TCP and
wheel conventions, implement the separate world-frame TCP task and prove it at
1/16/64 environments before starting a longer PPO run. Do not silently guess
remaining hardware measurements.

## Safety and repository rules

- Run `git status --short --branch` in every repository before changing it.
- Do not reset, clean, overwrite, or stash user files without explicit
  instruction.
- Do not mix the Isaac Conda environment with WheelRL's MuJoCo `.venv`.
- Do not upgrade Isaac Sim, Isaac Lab, PyTorch, CUDA, or the NVIDIA driver in
  place. The current stack can launch and step a headless simulation. If an
  upgrade is desired, make a separate environment.
- The current `origin` is the read-only upstream
  `zzzJie-Robot/LeggedManip_Lab` repository. Push project work to the user's
  fork on a feature branch and keep upstream as the fetch/reference remote.
- Preserve all checkpoints and local training logs.
- Do not claim that an existing exported policy is trained or suitable for the
  real robot. The only inspected B2-Z1 WBC run is a one-iteration smoke test.

## Confirmed machine resources

- OS: Ubuntu 24.04, native Linux.
- GPU: one NVIDIA GeForce RTX 4090, 24 GB VRAM, compute capability 8.9.
- NVIDIA driver: 570.207. Driver-reported CUDA capability: 12.8.
- CPU: Ryzen 9 3900X, 12 physical cores / 24 logical CPUs.
- RAM: approximately 64 GB total, approximately 54 GB available during audit.
- Storage: approximately 1.9 TB available on the working filesystem.
- No scheduler, container, CPU quota, or memory quota was detected.
- PyTorch CUDA was tested in the exact Isaac environment: CUDA was available,
  one RTX 4090 was found, and a CUDA tensor computation succeeded.

One Isaac startup warning reported PCIe link width `x4` while the device
supports `x16` (idle link generation was 1, reported maximum generation 3).
This is not yet proven to be a physical-slot problem, but the next session
should verify it under a representative workload and inspect the motherboard
slot/BIOS if it remains `x4`. It can reduce CPU-GPU transfer throughput.

## Conda and Isaac installation

Miniforge is installed under `$HOME/miniforge3`. A non-interactive SSH shell
does not put Conda on `PATH`, so activate it explicitly:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_isaaclab51
```

The relevant environment is `env_isaaclab51`:

- Python 3.11.16;
- Isaac Sim 5.1.0.0;
- Isaac Lab source version 2.3.2;
- PyTorch 2.7.0+cu128;
- torchvision 0.22.0+cu128;
- torchaudio 2.7.0;
- Warp 1.17.0;
- RSL-RL 5.0.1.

The Isaac Lab checkout is `$HOME/IsaacLab`, commit
`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`. It was clean during the audit but
is at a detached HEAD. Its `VERSION` is 2.3.2, and its main-series documentation
states compatibility with Isaac Sim 5.1. The Isaac Lab Python packages are
editable installs pointing at this checkout.

`LeggedManip_Lab 0.1.0` is also installed editable and points at
`$HOME/LeggedManip_Lab/source/LeggedManip_Lab`.

### Runtime checks completed

- `torch.cuda.is_available()` returned `True`.
- A CUDA tensor operation completed on the RTX 4090.
- Isaac Lab selected `cuda:0` and loaded the
  `isaaclab.python.headless.kit` experience.
- A `SimulationContext` reset successfully and completed four physics steps;
  `ISAAC_HEADLESS_SMOKE_OK` was printed.
- The application shutdown then remained alive for more than 60 seconds and
  had to be terminated. No test process was left running afterward. Treat the
  slow/hung shutdown as an unresolved issue and reproduce it with a maintained
  official example before running long jobs.
- A GUI smoke test was not performed over SSH. The SSH session had no X display,
  which is expected. Test GUI locally at the experiment machine.
- `vulkaninfo` is not installed, although the Isaac startup reported the RTX
  4090 active through Vulkan.

### Dependency warnings: do not repair blindly

`pip check` currently reports:

- `fastapi 0.115.7` expects `starlette < 0.46`, but 0.49.1 is installed;
- `isaacsim-kernel 5.1.0.0` pins `psutil 5.9.8`, but 7.2.2 is installed;
- `isaacsim-kernel 5.1.0.0` pins `typing_extensions 4.12.2`, but 4.16.0 is
  installed.

The headless physics smoke test nevertheless passed. Record these conflicts,
then reproduce the official example and task before changing any package.
Downgrading individual packages in place could break other installed tools.

## Existing repositories and protected local state

### LeggedManip_Lab

Checkout: `$HOME/LeggedManip_Lab`

- Branch: `master` at `4e12109d1eccb1d9da2e4a35f189fabea3f13677`.
- Several GO1/GO2 USD files are modified.
- `scripts/rsl_rl/teleop_b2_z1.py` is untracked.
- These edits are user work and must be preserved. Do not reset or overwrite
  them.
- The B2-Z1 USD files themselves did not appear as modified in `git status` at
  audit time.

Registered tasks include:

- `B2-Z1-Flat` and `B2-Z1-Flat-Play`;
- `B2-Z1-WBC` and `B2-Z1-WBC-Play`.

The main asset entry point is:

```text
source/LeggedManip_Lab/LeggedManip_Lab/assets/b2_z1/
  b2_z1_articulation_cfg.py
  b2_z1.usd
  configuration/b2_z1_{base,physics,robot,sensor}.usd
```

The active WBC task configuration is:

```text
source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/
  leggedmanip_lab/config/b2_z1/wbc_env_cfg.py
```

### Existing B2-Z1 run is only a smoke test

The inspected run is under:

```text
logs/rsl_rl/b2_z1_wbc/2026-08-31_13-49-01/
```

Its saved parameters specify 64 environments and `max_iterations: 1`. The
exported `.pt`/`.onnx` files and `model_0.pt` therefore prove only that the
pipeline could save a model. They are not a converged controller. Preserve the
run and its recorded Git diff.

### WheelRL

Checkout: `$HOME/WheelRL`

- Current machine-local branch:
  `backup/vbc-vision-hybrid-tcp-20260829` at `907993d`.
- Untracked and protected: `track_ppo_v3.zip`, `track_ppo_v4.zip`.
- The desired remote source branch is
  `feature/traditional-compliant-door-control` at `a4c0b4c`.
- The desired `feature/isaac-lab-tcp-tracker` branch does not yet exist.
- The machine-local backup branch and the desired remote branch have diverged
  after their common parent. Do not replace the current checkout in place.
  Use a separate worktree or clone after preserving/fetching refs.

WheelRL uses its own uv `.venv` for MuJoCo. It is not the Isaac environment.
Use WheelRL as the MuJoCo/control-requirements reference and use
LeggedManip_Lab as the currently functioning Isaac Lab baseline unless the user
chooses a different repository layout.

## Critical model mismatch findings

Training must remain blocked until the first group below is answered with the
user and the subsequent checks pass.

### 1. Robot identity: B2 versus B2-W

The existing Isaac/LeggedManip task is a **footed B2 + Z1**:

- 12 leg joints;
- 6 Z1 joints;
- 18 direct joint-position actions;
- no wheel joints and no wheel rolling constraints.

The historical WheelRL objective is **B2-W + Z1**, with four wheel joints. The
user previously considered beginning with the footed robot because it is
simpler, but that choice must be confirmed explicitly. Ask which physical
robot will be deployed and whether the first Isaac experiment should be B2 or
B2-W.

### 2. Three incompatible Z1 mounting transforms

Current files encode different assumptions:

- LeggedManip MuJoCo source: Z1 `link00` at approximately
  `(0.200, 0.000, 0.250)` relative to the B2 base.
- WheelRL combined MJCF: approximately `(0.211, 0.000, 0.105)`.
- WheelRL MPC URDF: approximately `(0.000, 0.000, 0.056)`.

None may be treated as real-hardware truth. Measure the installed bracket
transform and orientation on the actual robot. A few centimetres of error here
will invalidate TCP tracking and collision geometry.

### 3. Gripper and TCP are not equivalent

- The existing LeggedManip action space contains only the 12 legs and 6 arm
  joints. Its model contains Z1 gripper meshes, but the mover is fixed and no
  gripper joint/action is configured.
- WheelRL uses a separate one-DoF approximate gripper.
- WheelRL's MuJoCo TCP site is roughly 0.196 m beyond `link06`, while its MPC
  model tracks `arm_link06` itself.

Measure/declare the real tool flange-to-TCP transform, gripper type, opening
range, coupling, mass, collision pads and grasp reference point. Do not train
door grasping with the current fixed gripper.

### 4. Existing target is not a fixed world-frame 6D target

`UniformPoseWBCCommand` currently stores:

- X/Y relative to the moving Z1 `link0` frame;
- Z as absolute world height;
- orientation relative to `link0`/base.

The position reward uses the same mixed X/Y-link0 plus Z-world convention.
Base velocity is sampled as a separate command. Consequently, this task does
not teach the robot to relocate its base solely because a world-fixed TCP
target is outside arm reach. This is the central behavior required by the new
tracker.

Also audit the orientation implementation: command visualization constructs
world orientation from `link0`, while the reward currently multiplies by the
articulation root quaternion. Add frame-invariance and quaternion-order tests
before reward tuning.

### 5. Existing policy is direct joint RL, not the agreed precision stack

The baseline produces 18 joint-position targets directly. It has no frozen
deterministic SE(3) QP servo and no slow reachability/posture coordinator. It is
useful as an ablation and locomotion prior, but it should not automatically
replace the agreed architecture:

```text
world/door-frame 6D target
  -> reference manager
  -> slow learned reachability/posture coordinator
  -> constrained deterministic SE(3) QP/WBC
  -> robot joint controllers
```

### 6. Collision, actuator and contact assumptions are not deployment-ready

- The Isaac articulation config sets `enabled_self_collisions=False`.
- Solver velocity iterations are zero.
- Explicit Z1 effort limits in the actuator config are commented out. The
  effective limits may therefore be hidden in the binary USD drive/max-force
  attributes. Export them at runtime and compare each axis with the
  LeggedManip MuJoCo values `[40, 40, 20, 10, 5, 5]` and the real Z1.
- The configured delayed-PD gains, armatures, friction and 0--4-step delay are
  assumptions, not measured real values.
- The common plane material sets restitution to 1.0; verify the effective
  combined contact material rather than assuming realistic foot contact.
- WheelRL's MuJoCo arm/gripper collision masks effectively disable Z1-Z1
  self-collision, and its WBC scene additionally excludes one
  `base_link`/`z1_link02` pair.

Collision safety, contact behavior, torque/velocity/position limits and
infeasibility behavior must be explicit before sim-to-real work.

### 7. Camera and payload assumptions are unfinished

- The current Isaac B2-Z1 task has no configured head/wrist camera observations.
- WheelRL's head and wrist camera poses are simulation assumptions and require
  measured extrinsics/intrinsics.
- WheelRL camera housing geoms currently contribute approximately 0.054 kg each
  and can collide, despite being described as visual housings. Decide whether
  real cameras are modeled payloads or massless visualization; do not double
  count their mass.
- Measure all added payloads: Z1 bracket, gripper, cameras, compute, cabling and
  any lidar changes, then update total mass/COM/inertia.

### 8. WheelRL MPC URDF is not an Isaac import source

The WheelRL MPC URDF has 36 mesh references whose files are absent. Its mount
and TCP definitions also disagree with the combined MJCF. It is sufficient for
the current Pinocchio dynamics experiment but must not be converted directly
into the canonical Isaac USD.

The existing LeggedManip B2-Z1 USD is not a reproducible authoritative source
either: its conversion configuration has no source asset path, and no original
URDF is present in that repository. Before rebuilding or editing the robot,
obtain a traceable official/source URDF or CAD asset and record its URL or
commit, content hash, conversion tool version and conversion options.

### 9. Wheel geometry is internally inconsistent

If the physical platform is B2-W, do not inherit a wheel radius from the
current code without measurement. WheelRL's WBC, MPC and traditional door
controller use 0.10 m, while the RL door environment uses 0.12 m; the mesh
outer radius is approximately 0.113 m. Loaded effective rolling radius can be
different again. Measure radius and track width on the real robot, then use one
versioned value for kinematics, reward computation and deployment.

The footed LeggedManip rewards for air time and foot sliding are not valid
substitutes for wheel rolling, lateral-slip, drive and braking constraints.

### 10. There is no real-robot deployment bridge yet

No verified Unitree SDK/DDS/ROS low-level bridge was found in WheelRL. The
current work is simulation-only. Deployment still requires explicit
joint/order/sign mapping, state estimation, clock synchronization, command and
state latency measurement, emergency stop, torque/velocity watchdogs and a
safe fallback controller. Keep those concerns separate from the first Isaac
training smoke test, but do not label a policy sim-to-real ready without them.

## Real-robot information required from the user

Create a checked calibration record rather than editing numbers ad hoc:

- exact base model/revision: B2 or B2-W;
- photos/CAD/official URDF of the actual robot and Z1 bracket;
- base-to-Z1 mounting translation and rotation;
- exact leg, wheel, arm and gripper joint names, order, axes and signs;
- measured joint limits, torque/current limits, velocity limits and home pose;
- low-level command mode and control frequencies;
- motor/gear/PD behavior and measured command/state latency;
- total robot and payload masses, centers of mass and inertias where available;
- exact gripper model, controllable DoFs and flange-to-TCP transform;
- IMU frame/location and base/world coordinate conventions;
- foot geometry or wheel radius/contact direction and friction behavior;
- head/wrist camera intrinsics, resolution, frame rate and hand-eye extrinsics;
- known safe/unsafe self-collision pairs and real body clearances.

Use explicit units and frame names for every measurement. Photographs are good
for discovering attachments, but do not estimate a precision transform from a
single photograph.

## Recommended order for the next session

1. Read this file, `WheelRL/HANDOFF.md`, and
   `WheelRL/docs/research/ISAACLAB_TCP_TRACKING_HANDOFF_2026-09-02.md`.
2. Re-run `git status` in WheelRL, IsaacLab and LeggedManip_Lab; preserve all
   dirty/untracked files.
3. Ask the user to confirm B2 versus B2-W and enumerate/photograph the known
   real-model differences. Build a calibration table before editing USD.
4. Reproduce one maintained official headless example with a clean exit, then
   run a local GUI example. Investigate the observed slow shutdown.
5. Inspect the existing USD articulation at runtime: enumerate body/joint names,
   action mapping, limits, masses and fixed gripper state.
6. Select one canonical calibrated asset. Do not import the incomplete MPC URDF.
7. Compare canonical Isaac FK with the calibrated MuJoCo model at the home pose
   and randomized joint configurations.
8. Implement tests for left/right signs, quaternion convention, TCP definition,
   mount transform and a world target invariant under base translation/yaw/
   height/tilt.
9. Implement a minimal state-only fixed-world 6D target environment at 1, 16
   and 64 environments. Do not add cameras, doors or long training yet.
10. Only after the asset/frame gates pass, run a one-iteration save/load smoke
    test and benchmark 256/512/1024/2048/4096 environments.

## Go/no-go gates before real training

- The selected simulated robot is the same B2/B2-W configuration as the
  intended physical robot.
- All actuated joints, including wheels/gripper if applicable, are mapped by
  explicit name and tested for sign.
- Z1 mount and flange-to-TCP transforms are measured and unit-tested.
- A fixed world target remains unchanged while the base moves or tilts.
- Isaac and the canonical MuJoCo/kinematics model agree on TCP FK within a
  declared tolerance.
- Self/body/ground collision behavior is understood and safety constraints are
  active.
- The controller observes actual limits and has a bounded fallback for
  infeasibility or deadline misses.
- The official Isaac example and the project smoke test both exit cleanly.
- A checkpoint can be saved, loaded and evaluated deterministically.

Until these gates pass, do not start the 4096-environment training run and do
not interpret reward improvement as evidence of accurate TCP tracking.
