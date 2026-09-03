# B2-W + Z1 TCP tracking: implementation record and training plan

Last updated: 2026-09-03 (Europe/Berlin)

## Current outcome

The project now has a separate B2-W + Z1 Isaac Lab asset. It does not replace
the upstream footed B2 + Z1 asset or its existing tasks.

Completed work:

- Built a reproducible combined URDF from Unitree's `unitree_ros` repository at
  commit `7d6075f7f58588b189b940130e3edab3c839b2df`.
- Preserved the 12 leg joints, four continuous wheel joints, six Z1 joints and
  one gripper joint: 23 movable joints in total.
- Renamed the imported B2-W `*_foot` wheel links/joints to `*_wheel` so task
  selectors cannot confuse wheels with feet.
- Added the real-hardware 35 mm Z1 adapter height. The adapter XY footprint,
  mass, inertia and exact XY mounting position remain provisional.
- Added independent actuator groups for legs, wheels, arm and gripper.
- Added two explicit arm presets:
  - hardware-like stow: `[0.00, 0.15, -0.45, 0.30, 0.00, 0.00] rad`;
  - raised TCP-training ready pose:
    `[0.00, 1.35, -1.65, 0.30, 0.00, 0.00] rad`.
- Added `b2w_z1_stow.usda`, a direct-open Isaac Sim wrapper that authors the
  same hardware-like stow state without modifying the generated base USD.
- Added a validator for USD structure, wheel axes, mounting height, arm
  presets, lidar clearance and dynamic stability.

The stow pose was reconstructed from one side photograph of the real robot.
It produces the correct stacked Z-fold instead of folding the gripper through
the lidar. Exact transformed collision-mesh checking gives the limiting distal
gripper-stator vertical clearance above the modeled lidar as approximately
109.8 mm. The final 250-step Isaac smoke test passed with base height 0.538 m
and upright score 1.000.

This does not yet make the model sim-to-real calibrated. Encoder readback of
the real stow pose should replace the photo estimate. The real mount XY pose,
adapter mass/inertia, wheel radius under load, actuator parameters, payload and
flange-to-TCP transform must also be measured.

## What comes next

The next step is to implement the training task and its deterministic control
baseline, not to start a long PPO run immediately. The proposed task name is
`B2W-Z1-TCP`; the existing `B2-Z1-WBC` task must remain unchanged as a
comparison baseline.

No B2-W TCP task or trained B2-W policy exists yet. The completed work is the
asset and validation gate needed before that task can be implemented safely.

Create this task-specific structure:

```text
source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/
  leggedmanip_lab/
    config/b2w_z1_tcp/
      __init__.py
      env_cfg.py
      agents/
        __init__.py
        rsl_rl_ppo_cfg.py
    mdp/
      tcp_commands.py
      b2w_actions.py
```

The task must use `base_link`, `.*_wheel`, and a calibrated virtual TCP based
on `gripper_stator`. It must not reuse the old `base`, `.*_foot`, or
`end_effector` selectors.

## Target and controller definition

Each parallel environment stores one immutable TCP target in its environment
or world frame. Moving the base must not move the target. The reference manager
turns that final goal into a minimum-jerk or spline pose/twist trajectory.

Targets should not come from a manually chosen rectangular reach box. Instead:

1. sample a collision-free arm configuration;
2. compute its TCP pose with FK;
3. use that pose as a guaranteed reachable local target;
4. later apply a sampled virtual or "ghost-base" displacement to the target so
   whole-body relocation is required.

This produces valid targets from the robot's actual kinematics while still
teaching the policy to move the chassis for goals outside comfortable arm
reach.

The recommended precision architecture is:

```text
fixed world-frame 6D TCP goal
  -> smooth reference pose and twist
  -> slow learned reachability/posture coordinator
  -> frozen batched DIK/OSC or constrained WBC
  -> leg position, wheel velocity and arm commands
```

The learned coordinator should begin with the macro action
`[v_x, yaw_rate, body_height, body_pitch]`. B2-W should not receive a lateral
velocity command because it cannot produce lateral motion without wheel slip.
The arm controller tracks the TCP at every physics/control step and may later
accept a small bounded learned residual.

For a quick end-to-end ablation, also retain a direct 22-action baseline:

- 12 leg position residuals;
- four wheel velocity targets;
- six arm position residuals;
- fixed gripper during free-space tracking.

This baseline is useful for comparison, but the hybrid controller is the
preferred route for precise terminal tracking and explicit safety constraints.

## Observations

The deployable actor should receive only signals available on the real robot:

- base angular velocity and projected gravity;
- estimated base linear velocity;
- leg joint positions and velocities;
- wheel angular velocities, but not unbounded wheel angles;
- arm joint positions and velocities;
- TCP position error and SO(3) logarithmic orientation error;
- desired TCP linear and angular velocity;
- joint-limit/reachability margin;
- previous action and a short three-to-five-frame history.

The critic may additionally receive true base velocity, contact forces,
collision distance, payload, friction, randomized dynamics and delay. Actor
inputs must never depend on simulation-only values unavailable at deployment.

## Rewards, costs and terminations

Positive terms:

- TCP position and orientation tracking;
- desired TCP twist tracking during motion;
- continuous dwell inside the final pose/twist tolerance;
- upright posture and appropriate base height.

Penalties and hard constraints:

- lateral wheel slip and rolling/contact kinematic error;
- wheel lift and non-wheel ground contact;
- unnecessary chassis motion for arm-reachable or settled targets;
- joint-limit proximity, low manipulability and self/body collision;
- base vertical velocity, roll/pitch rate and settled jitter;
- separate leg, wheel and arm torque, acceleration, power and action-rate
  penalties;
- fall, illegal body contact, invalid state or controller infeasibility.

Foot air-time, gait-variance and foot-slide rewards must be removed rather than
renamed to wheels. A rolling wheel center is supposed to move while touching
the ground.

## Parallel training curriculum

Use 200 Hz physics/control (`dt=0.005`) and initially 50 Hz policy updates
(`decimation=4`). Promotion is based on held-out metrics and multiple seeds,
not on training reward alone.

| Stage | Distribution | Promotion gate |
| --- | --- | --- |
| 0 | One robot, nominal flat plane, no RL | Correct frames/signs; 2,000 stable steps; no unintended contacts |
| 1 | Static FK-derived local goals, arm dominant | At least 95% held-out success, 99% survival, negligible base motion |
| 2 | Smooth 6D pose/twist trajectories, base held | Nominal moving RMS at most 10 mm and 5 degrees |
| 3 | Ghost-base translation/yaw requiring relocation | Relocation works without continuous or unnecessary chassis motion |
| 4 | Safe reset-pose diversity and modest measured payload | Joint, collision and saturation gates pass |
| 5 | Mild dynamics, sensing and delay randomization on flat ground | At least 90% success; nominal regression below 15% |
| 6 | Measured randomization envelope and modest pushes | At least three seeds; saturation below 1% of steps |
| 7 | Deployment-matched mild terrain | Terrain passes while flat regression remains below 15% |
| 8 | Frozen held-out evaluation and sim-to-sim test | Select by the complete metric matrix, not a showcase episode |

Arm behavior in every parallel environment should be consistent:

- reset in the raised ready pose for ordinary tracking episodes;
- initialize the target at the current TCP, then move it smoothly;
- use the arm while a goal remains comfortably reachable;
- roll or change body posture only as reachability/joint margin deteriorates;
- counter-move the arm while the chassis moves so the world target remains
  fixed;
- decay wheel commands to zero and hold the TCP with damping during settling;
- terminate or enter a defined recovery/retract behavior after a fall or unsafe
  state;
- introduce stow-to-ready transitions only after ordinary tracking is stable.

Scale parallelism progressively on the RTX 4090:

1. 1 environment for GUI, coordinate and sign debugging;
2. 16 environments for reset/action/reward tests;
3. 64--256 environments for batched controller and numerical validation;
4. 512 environments for the first PPO overfit/debug run;
5. 2,048 environments for curriculum development;
6. up to 4,096 environments only after profiling physics, controller, PPO,
   GPU memory and wall-clock throughput;
7. a separate fixed set of approximately 256 environments for evaluation.

Record RMS/P95 TCP pose error, settling time, overshoot, settled jitter,
survival, base displacement, wheel slip/lift, joint margin, collisions,
controller fallback and torque/velocity saturation.

## Domain randomization

Do not enable the upstream task's full randomization at the start. First prove
the controller under nominal conditions so coordinate, reward and controller
errors remain diagnosable.

Suggested mild starting envelope after nominal tracking passes:

| Parameter | Mild stage | Later measured envelope |
| --- | ---: | ---: |
| Ground friction | 0.7--1.1 | approximately 0.4--1.2, then replace with measurements |
| Link mass/inertia | +/-5% | mass +/-10%, inertia +/-15% |
| Base COM | +/-1 cm XY | up to +/-3 cm XY |
| Motor/PD gains | +/-10% | +/-20% |
| Actuator/observation delay | 0--10 ms | 0--20 ms |
| Joint position noise | +/-0.005 rad | +/-0.01 rad |
| Joint velocity noise | +/-0.2 rad/s | +/-0.5 rad/s |
| TCP estimate noise | +/-2 mm, +/-0.5 degree | +/-5 mm, +/-1 degree |
| Payload | none initially | measured mass, COM and inertia distribution |
| Pushes | none initially | modest horizontal disturbances, added last |

Keep approximately 20% of environments nominal during the later randomized
stages to anchor precision. Wider randomization is not automatically better;
replace provisional ranges with real measurements and do not randomize the
same latency twice in different layers.

## Terrain: what to borrow from MLM

The MLM paper trains a footed Go2 + Airbot system with a reward-threshold
terrain curriculum over random undulations, slopes, discrete obstacles and
stairs. Its critic receives 187 terrain-height samples. It also uses broad
domain randomization, including link mass `0.8--1.2x`, payload `0--2 kg`,
payload-position offsets of +/-5 cm, friction `0.05--2.0`, motor strength
`0.8--1.2x`, and observation delay `0--20 ms`.

Source: [MLM: Learning Multi-task Loco-Manipulation Whole-Body Control for
Quadruped Robot with Arm](https://arxiv.org/html/2508.10538).

Do not copy that terrain distribution into the first B2-W training run. MLM's
robot is footed, while B2-W has non-steering wheels, different contacts and a
precision indoor TCP objective. Early random geometry would mix locomotion
failures with TCP/controller bugs and slow diagnosis.

Use a plane through the nominal and dynamics-randomization stages. Add friction
patches first. Once randomized-flat tracking passes, use this deployment-matched
starting mixture for an indoor robot. At 4,096 environments it corresponds to:

- 20% exact nominal plane: 819 environments;
- 50% randomized flat plane/friction patches: 2,048 environments;
- 20% low height variation, beginning near +/-5 mm and capped near +/-10 mm:
  819 environments;
- 10% slopes up to approximately +/-3 degrees: 410 environments.

Do not add stairs, rubble or discrete obstacles unless the deployment really
contains them. Add a real/deployable terrain observation such as a lidar height
map before terrain exceeds roughly 1--2 cm. Texture and lighting augmentation
has no effect on a state-only actor and belongs only to a later camera-based
policy.

## Immediate implementation checklist

1. Measure or declare a clearly labeled simulation-only
   `gripper_stator -> tcp` transform.
2. Verify all four wheel signs and effective rolling radius.
3. Add the immutable world-frame command generator and invariance tests.
4. Add a one-environment batched DIK/OSC arm-tracking baseline.
5. Register `B2W-Z1-TCP` and `B2W-Z1-TCP-Play` with B2-W-specific actions,
   observations, rewards, contacts and terminations.
6. Pass 1/16/64-environment smoke tests and deterministic checkpoint
   save/load.
7. Benchmark 256/512/1,024/2,048 environments before choosing the full PPO
   scale.
8. Begin Stage 1 training on a nominal plane. Terrain is intentionally later.
