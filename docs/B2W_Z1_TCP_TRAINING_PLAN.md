# B2-W + Z1 TCP tracking: implementation record and training plan

Last updated: 2026-09-04 (Europe/Berlin)

Chinese step-by-step explanation:
[`B2W_Z1_TCP_TRAINING_STEPS_ZH.md`](B2W_Z1_TCP_TRAINING_STEPS_ZH.md).
Completed deterministic controller gate and measurements:
[`B2W_Z1_TCP_DIK_GATE_ZH.md`](B2W_Z1_TCP_DIK_GATE_ZH.md).
Completed standalone world-frame harness and measurements:
[`B2W_Z1_WORLD_FRAME_GATE_ZH.md`](B2W_Z1_WORLD_FRAME_GATE_ZH.md).
Historical open-loop wheel diagnostic status:
[`B2W_Z1_WHEEL_GATE_ZH.md`](B2W_Z1_WHEEL_GATE_ZH.md).
Official-source investigation of real B2-W turning:
[`research/B2W_REAL_TURNING_OFFICIAL_ZH.md`](research/B2W_REAL_TURNING_OFFICIAL_ZH.md).
Frozen `robot_lab` B2-W policy integration and measured results:
[`B2W_Z1_ROBOT_LAB_FOUNDATION_ZH.md`](B2W_Z1_ROBOT_LAB_FOUNDATION_ZH.md).
Physical frozen-locomotion plus immutable-world TCP gate:
[`B2W_Z1_ROBOT_LAB_TCP_HOLD_GATE_ZH.md`](B2W_Z1_ROBOT_LAB_TCP_HOLD_GATE_ZH.md).
Trainable task MVP, smoke tests and PPO pilot:
[`B2W_Z1_TCP_TASK_MVP_ZH.md`](B2W_Z1_TCP_TASK_MVP_ZH.md).

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
- Tracked the user-provided installation photograph in the asset directory and
  documented what it can and cannot establish. The default mount is
  `(0.211, 0.0, 0.110) m`, identity RPY: front-half, centered and upright.
- Added a geometry-free nominal `tcp_frame` 0.145 m forward of
  `gripper_stator`, or 0.196 m forward of `link6`. It is suitable for
  simulation-only controller development and remains explicitly configurable.
- Added independent actuator groups for legs, wheels, arm and gripper.
- Added two explicit arm presets:
  - photo-matched provisional stow:
    `[0.00, 0.15, -0.45, 0.30, 0.00, 0.00] rad`;
  - raised TCP-training ready pose:
    `[0.00, 1.35, -1.65, 0.30, 0.00, 0.00] rad`.
- Added `b2w_z1_stow.usda`, a direct-open Isaac Sim wrapper that authors the
  same photo-matched provisional stow state without modifying the generated
  base USD.
- Added a validator for USD structure, wheel axes, mounting height and
  rotation, nominal TCP parent/pose/forward direction, arm presets, lidar
  clearance and dynamic stability.
- Added a reproducible fixed-base 6D TCP differential-IK gate. The final
  64-environment RTX 4090 run passed all five signed target families with a
  worst position RMS of 2.19 mm, orientation RMS of 0.24 degrees and terminal
  jitter of 1.63 mm. Gravity remained enabled and was handled with explicit
  Z1 gravity feed-forward; actuator delay was held at zero for this nominal
  gate.
- Corrected replicated reset semantics by adding `scene.env_origins` to every
  root state, then passed the standalone 64-environment floating-base
  world-frame harness. Across seven signed root excitations, the worst moving
  RMS was 8.71 mm / 2.02 degrees; the worst displaced-hold RMS was
  1.15 mm / 0.136 degrees and the worst held jitter was 0.654 mm. The immutable
  world target was never mutated.
- Established a usable nominal straight/reverse wheel baseline with a
  provisional simulation velocity gain of 10 N m/(rad/s), while preserving the
  official 20 N m effort and 50 rad/s velocity limits. The complete wheel gate
  remains failed because neither left nor right arc reaches the requested yaw.
- Pinned and checksum-verified the public `rl_sar` B2-W TorchScript checkpoint,
  reproduced its exact 57-observation/16-action contract and actuator preset,
  and added an explicit name-mapped adapter for the combined B2-W + Z1
  articulation. The frozen policy passed stand, forward, reverse and right-turn
  phases and produced both left and right physical turns on the combined USD.
  A 16-environment stand test also passed. Low-command left yaw still shows a
  strong dead zone/asymmetry and must be treated as a closed-loop plant
  limitation rather than hidden by relaxing the gate.
- Connected the frozen locomotion policy to the 200 Hz Z1 world-frame DIK
  without writing root state. After exposing the measured forward bias and yaw
  dead-zone compensation as explicit configuration, the coupled physical gate
  passed at 1, 16 and 64 environments. The 64-environment worst moving TCP RMS
  was 12.42 mm / 1.08 degrees and worst terminal RMS was
  1.55 mm / 0.196 degrees; base terminal errors were below
  3.4 mm / 0.39 degrees.

The world-frame result is a kinematic coordinate/controller harness, not a
production WBC: gravity was disabled, root pose and velocity were written
directly, and wheel/contact dynamics were not in the loop.

The stow pose was reconstructed from the tracked side photograph of the real robot.
It produces the correct stacked Z-fold instead of folding the gripper through
the lidar. Exact transformed collision-mesh checking gives the limiting distal
gripper-stator vertical clearance above the modeled lidar as approximately
109.8 mm. The final 250-step Isaac smoke test passed with base height 0.538 m
and upright score 1.000.

This does not yet make the model sim-to-real calibrated. Encoder readback of
the real stow pose should replace the photo estimate. The real mount pose,
adapter geometry/mass/inertia, wheel radius under load, actuator parameters,
payload and gripper-to-TCP transform must also be measured.

## What comes next

The fixed-base arm gate and standalone moving-root world-frame harness have
passed. The earlier open-loop fixed-leg wheel gate still cannot turn, but it is
no longer the production blocker: the public `robot_lab` B2-W locomotion policy
uses coordinated leg and wheel motion and has produced forward, reverse and
bidirectional physical yaw on the combined B2-W + Z1 USD. Its strict left-turn
contact/yaw-rate and some stopping margins are not all passed, so the high-level
controller must treat it as an asymmetric closed-loop plant.

Official model, SDK, manual and tutorial checks now establish that the real
B2-W has no steering joint and supports four-wheel-grounded turn-in-place. The
mechanism is necessarily differential/skid steering with lateral tire scrub;
the video does not show deliberate wheel lifting or large body lean. Unitree's
internal `wheeled_sport` mixing, slip compensation and possible small leg-load
corrections remain proprietary. The earlier fixed-leg open-loop approximately
two-degree response therefore diagnoses that controller/contact setup; the
frozen learned policy's later bidirectional turns show that the combined USD
itself is not locked against yaw.

The next sequence is:

```text
[done] complete the batched frozen-locomotion validation
  -> [done] combine physical B2-W movement with immutable-world TCP DIK
  -> [done] pass 1/16/64-environment coupled controller gates
  -> [done] integrate the B2W-Z1-TCP task around the frozen low-level policy
  -> [done] pass 1/16/64-environment task smoke and checkpoint save/load
  -> [done] run a 20-iteration end-to-end PPO integration pilot
  -> [done, diagnostic only] add an analytic coordinator comparison
  -> define directional relocation gates and tune the nominal objective/controller
  -> train and held-out evaluate the nominal coordinator
  -> add general collision-filtered FK targets, then measured randomization
```

The registered task name is `B2W-Z1-TCP`; the existing `B2-Z1-WBC` task remains
unchanged as a comparison baseline.

No final B2-W TCP policy exists yet. The trainable high-level task and a short
PPO integration pilot now exist, but that pilot did not outperform zero action.
A compatible pretrained B2-W locomotion policy supplies the low-level base
controller; the remaining work is to train and validate the slow TCP
reachability/posture coordinator, not locomotion from scratch.

The task-specific structure is:

```text
source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/
  leggedmanip_lab/
    config/b2w_z1/
      __init__.py
      tcp_env_cfg.py
      agents/
        __init__.py
        rsl_rl_ppo_cfg.py
    mdp/
      b2w_tcp_command.py
      b2w_tcp_action.py
      b2w_tcp_terms.py
```

The task must use `base_link`, `.*_wheel`, and the explicit `tcp_frame` (after
replacing its nominal transform with a calibrated value for sim-to-real). It
must not reuse the old `base`, `.*_foot`, or
`end_effector` selectors.

## Target and controller definition

Each parallel environment stores one immutable TCP target in its environment
or world frame. Moving the base must not move the target. The reference manager
turns that final goal into a minimum-jerk or spline pose/twist trajectory.

The implemented MVP first applies a sampled planar ghost-root transform to the
reset TCP pose. It gives each target a known whole-body solution without a
manually chosen rectangular reach box. The next general-6D target generator
should instead:

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
       -> [vx, yaw rate] -> frozen robot_lab B2-W policy -> leg position + wheel velocity
       -> desired TCP pose/twist -> batched DIK/OSC -> six Z1 joint targets
```

The implemented MVP coordinator action is `[v_x, yaw_rate]`. B2-W does not
receive a lateral velocity command because it cannot produce lateral motion
without wheel slip. After this two-action relocation policy is validated, a
controlled extension may add `[body_height, body_pitch]`; the arm controller
continues tracking the TCP at every physics step and may later accept a small
bounded learned residual.

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

The real wheel/contact task is now wired and has passed finite 1/16/64-environment
smoke tests plus a 20-iteration PPO integration pilot. Longer held-out nominal
runs, collision metrics and a 2,000-step stability gate remain before Stage 0
can be considered fully closed.

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

1. **Done for simulation:** declare the provisional
   `gripper_stator -> tcp` transform and validate the fixed-base DIK path.
2. **Done as a standalone regression:** verify immutable world targets across
   64 separated environments, both signs and non-zero initial yaw.
3. **Done as the production locomotion foundation:** pin the public B2-W
   checkpoint, reproduce its exact observation/action/actuator contract and
   demonstrate physical stand, translation and bidirectional yaw on the
   combined USD. Strict left-turn contact/yaw-rate and some stopping margins
   remain open; keep them and the failed open-loop fixed-leg gate visible as
   plant diagnostics.
4. **Done:** combine frozen physical locomotion with the immutable-world TCP
   controller without writing root pose or velocity.
5. **Done:** pass the coupled controller gate in 1/16/64 environments, including
   TCP precision, posture and command-asymmetry metrics. Contact/slip metrics
   still need to be carried into the production task.
6. **Done:** register `B2W-Z1-TCP` and `B2W-Z1-TCP-Play` with B2-W-specific
   actions, observations, rewards, contacts and terminations.
7. **Done:** pass 1/16/64-environment task smoke tests and checkpoint
   save/load/export. The target suite still produces measurable IK clamp events
   that must remain visible in evaluation.
8. **Integration pilot done, training incomplete:** a 256-environment,
   20-iteration PPO run was stable and exportable but did not beat the matched
   zero-action baseline. A first analytic comparison reduced arm-home error but
   also failed the base relocation accuracy objective, especially for lateral
   and yaw targets. Next freeze directional target suites and pass explicit
   accuracy gates, then tune/train on a nominal plane and benchmark
   512/1,024/2,048 environments. Terrain is intentionally later. See
   [`B2W_Z1_TCP_TASK_MVP_ZH.md`](B2W_Z1_TCP_TASK_MVP_ZH.md).
