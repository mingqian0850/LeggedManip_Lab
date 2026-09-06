# Quadruped posture and symmetry literature lookup — 2026-09-06

Scope: primary papers on left/right symmetry, natural stance, posture regulation,
whole-body loco-manipulation, and end-effector tracking for quadruped robots.

Search queries used:

- quadruped locomotion reinforcement learning left right symmetry equivariant policy mirror loss
- quadruped robot nominal joint posture reward natural stance symmetry
- quadruped loco manipulation whole body control posture regularization null space end effector tracking
- quadruped adaptive body posture reinforcement learning crouch tilt confined space

Primary sources retained after checking the method text:

1. Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot Policies" (2024)
   - https://arxiv.org/abs/2403.04359
   - Compares mirror loss and symmetry data augmentation on goal-conditioned robot tasks, including ANYmal climbing/pushing.
   - Warns that hard symmetry can be harmful because real robots and neutral states are not perfectly symmetric.

2. Yu, Turk, and Liu, "Learning Symmetric and Low-Energy Locomotion" (2018)
   - https://arxiv.org/abs/1801.08093
   - Adds an action-level mirror-symmetry auxiliary loss to PPO; includes a quadruped example.

3. Su et al., "Leveraging Symmetry in RL-based Legged Locomotion Control"
   - https://hybrid-robotics.berkeley.edu/publications/Symmetry_RL_LeggedLoco.pdf
   - Compares PPO, mirrored data augmentation, and hard equivariant actor/invariant critic networks on quadruped loco-manipulation and bipedal tasks.
   - Reports that hard equivariance works well for task-space symmetry, while augmentation can be more robust to real hardware asymmetry.

4. Bellicoso et al., "ALMA - Articulated Locomotion and Manipulation for a Torque-Controllable Robot" (ICRA 2019)
   - https://www.research-collection.ethz.ch/handle/20.500.11850/384187
   - Hierarchical optimization WBC with torso-posture optimization to enlarge the arm workspace.

5. Jiang et al., "Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator" (2024)
   - https://arxiv.org/abs/2412.03012
   - Uses nominal wheel-position deviation, all-feet-contact, collision, base-stability and smoothness terms inside a nonlinear reward-prioritization/fusion scheme.

6. Li et al., "Efficient Learning of A Unified Policy For Whole-body Manipulation and Locomotion Skills" (2025)
   - https://arxiv.org/abs/2507.04229
   - Uses an IK-derived physical-feasibility reward so low EE goals induce useful torso pitch/front-leg bending instead of being suppressed by balance learning.

7. Liu et al., "MLM: Learning Multi-task Loco-Manipulation Whole-Body Control for Quadruped Robot with Arm" (2025)
   - https://arxiv.org/abs/2508.10538
   - Uses an adversarial motion prior style reward, default-position action offsets, collision/smoothness constraints, and future TCP trajectory context.

8. Peng et al., "Learning Agile Robotic Locomotion Skills by Imitating Animals" (RSS 2020)
   - https://arxiv.org/abs/2004.00784
   - Retargets dog motion with a default-pose regularizer and trains joint/root/end-effector motion tracking for natural quadruped movement.
