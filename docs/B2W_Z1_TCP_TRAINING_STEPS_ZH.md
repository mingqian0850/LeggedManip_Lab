# B2-W + Z1 TCP Tracking 训练步骤中文说明

最后更新：2026-09-04（Europe/Berlin）

本文解释从当前 B2-W + Z1 Isaac 资产到并行 PPO 训练的每一步分别解决
什么问题。详细参数和英文工程计划见
[`B2W_Z1_TCP_TRAINING_PLAN.md`](B2W_Z1_TCP_TRAINING_PLAN.md)。
固定本体微分 IK 的实测结果和复现命令见
[`B2W_Z1_TCP_DIK_GATE_ZH.md`](B2W_Z1_TCP_DIK_GATE_ZH.md)。
固定世界坐标测试见
[`B2W_Z1_WORLD_FRAME_GATE_ZH.md`](B2W_Z1_WORLD_FRAME_GATE_ZH.md)，轮子执行层
状态见 [`B2W_Z1_WHEEL_GATE_ZH.md`](B2W_Z1_WHEEL_GATE_ZH.md)，真实 B2-W 转向
的官方资料核对见
[`research/B2W_REAL_TURNING_OFFICIAL_ZH.md`](research/B2W_REAL_TURNING_OFFICIAL_ZH.md)。
复用公开 B2-W locomotion policy 的接口与实测结果见
[`B2W_Z1_ROBOT_LAB_FOUNDATION_ZH.md`](B2W_Z1_ROBOT_LAB_FOUNDATION_ZH.md)。
真实底盘运动时的世界 TCP 保持结果见
[`B2W_Z1_ROBOT_LAB_TCP_HOLD_GATE_ZH.md`](B2W_Z1_ROBOT_LAB_TCP_HOLD_GATE_ZH.md)。
正式 task、并行冒烟测试和 PPO pilot 见
[`B2W_Z1_TCP_TASK_MVP_ZH.md`](B2W_Z1_TCP_TASK_MVP_ZH.md)。

## 总体结论

下一阶段还不是长时间训练。当前正确顺序是：

```text
校准名义 TCP
  -> 建立确定性机械臂控制器
  -> 验证固定世界目标
  -> 接入冻结的 B2-W locomotion policy
  -> 验证真实 wheel/contact 物理下的 TCP 保持
  -> 创建 B2W-Z1-TCP 任务
  -> 小规模并行冒烟测试
  -> 分阶段 PPO 训练
  -> 域随机化
  -> 最后加入部署匹配的轻微地形
```

前五步的目的是保证训练问题本身正确，之后才是正式学习策略。否则 PPO
可能只是学会补偿错误坐标、错误轮子方向或错误 TCP，而不是学会正确的全身
协调。

当前进度：固定本体、重力补偿微分 IK 已通过。修复多环境 reset 时 root state
漏加 `scene.env_origins` 后，64 环境固定世界目标 harness 也已通过；七类本体
运动的最坏动态 RMS 为 8.712 mm / 2.017°，最坏位移端保持 RMS 为
1.149 mm / 0.136°，世界目标没有被重新生成或拖动。

不过该 harness 关闭重力并直接写入本体轨迹，只验证坐标语义、floating-base
Jacobian 和机械臂反向补偿，不等于生产 WBC。早期固定腿开环轮速 gate 不能
转向，但它不是现在的生产路线。固定版本的公开 `robot_lab` B2-W policy 已经在
当前 B2-W + Z1 USD 上完成站立、前后移动并证明两个方向都能产生物理 yaw；严格
左转 contact/yaw-rate 与部分停车窗口仍未全部通过。16 环境 deployed arm 站立
通过。“learned locomotion 产生真实底盘运动时，Z1 保持 immutable
world TCP”的组合闭环随后也已在 1、16、64 环境通过。64 环境最坏 moving RMS
为 12.42 mm / 1.08°，最坏 terminal RMS 为 1.55 mm / 0.196°。下一缺口已经变成
正式 task wiring 与可达性协调器，而不是底盘/机械臂能否同时闭环。

## 六个实施步骤

| 步骤 | 代表什么 | 解决的问题 |
| --- | --- | --- |
| 1. 校准 `gripper_stator -> tcp_frame` | 当前已有仿真名义 TCP；真机前测量夹爪基座到实际抓取中心的平移和旋转 | 消除模型末端与真实接触点之间的固定误差 |
| 2. 验证 B2-W 执行层（生产基础可用） | 固定腿开环 wheel gate 仍是失败诊断；冻结 locomotion 已完成前后运动并产生双向 yaw，但严格左转接触/yaw-rate 与部分停车 margin 仍开放 | 防止轮子互相对抗，并确认真实 wheel/contact 闭环可用且不隐藏不对称性 |
| 3. 实现固定世界坐标目标（已接入 task） | 64 环境独立不变性和正式 command term 均已通过 | 让移动底盘真正帮助接近远处目标 |
| 4. 验证 DIK/OSC/WBC（MVP 内环已完成） | 固定 DIK、运动学补偿及真实底盘运动下的 200 Hz DIK 已通过；受约束 WBC 与碰撞安全仍未完成 | 独立检查 FK、Jacobian、坐标系和关节方向 |
| 5. 创建 `B2W-Z1-TCP` 任务（已完成） | 已定义 command、57 维 observation、2 维 action、reward 和 termination | 建立能被 Isaac Lab 与 RSL-RL 正确训练的新环境 |
| 6. PPO 训练（管线 pilot 已完成，策略未收敛） | 20 iterations 已验证保存、加载和导出；下一步是标称平地训练与固定目标评估 | 学会何时只动机械臂，何时移动轮子和身体 |

## 第 1 步：定义真实 TCP

`gripper_stator` 是 USD 中的刚体坐标系，但 TCP 应位于夹爪真正夹住物体的
中心位置。需要定义：

```text
T_gripper_stator_tcp = [translation, rotation]
```

当前官方派生资产已经包含一个明确的 `tcp_frame`。其仿真初值为：

```text
gripper_stator -> tcp_frame = [0.145, 0, 0] m，旋转为单位旋转
link6 -> tcp_frame = [0.196, 0, 0] m
```

这个坐标系靠近官方夹爪尖端中心，足以保证仿真中的所有控制器使用同一个末端
定义；它不是根据单张侧视照片测得的真机标定结果。生成脚本已经提供
`--tcp-x/y/z` 和 `--tcp-roll/pitch/yaw`，取得真机测量后直接替换这些参数。

如果该变换在 X 方向偏差 20 mm，即使仿真显示 TCP 误差为零，真实夹爪仍会
偏差 20 mm。在拿到真机测量前，可以使用明确标记为“simulation-only”的临时
TCP，但不能把它的结果称为真机精度。

完成标准：

- Isaac FK 与独立 FK 在多个随机关节姿态下结果一致；
- 四元数顺序、坐标轴方向和单位均有自动测试；
- 身体移动、升降或倾斜时，世界目标本身不发生变化。

## 第 2 步：验证 B2-W 轮子

对四个轮子分别发送低速正、负命令，并确认：

- 正速度对应的真实滚动方向；
- 左右和前后关节的名字、顺序与符号；
- 轮子是否围绕模型中的 Y 轴旋转；
- 承重后的有效滚动半径；
- 轮速和底盘速度是否近似满足：

```text
base_linear_velocity ~= wheel_angular_velocity * effective_radius
```

目前 USD 中的速度上限不是推荐训练动作幅值。第一版策略应使用保守轮速范围，
确认制动、接触和滑移正常后再扩大。

当前实测确认四轮符号和左右转向符号本身正确；官方 `20 Nm / 50 rad/s` 限制
保持不变。将仿真轮速误差增益从无力驱动的临时值 1.2 调整到 10 后，直行和
倒车能够稳定执行。这个 gain 只是仿真基线，不是实机电机环已标定值。

左右转向仍是红灯。官方轮胎碰撞已经是凸包，不是逐三角形接触；主要矛盾是
四个不可转向轮在各向同性摩擦下转弯必须横向侧滑。下一实验要在相同 gain、
摩擦和官方力矩限制下，先增加按关节名映射的纯正/负 yaw 原地旋转回归，再
依次测试轮胎碰撞表示、阻抗稳定站姿和接触模型，而不是靠无限增益或降低摩擦
制造一个表面通过结果。

官方 URDF/MuJoCo 模型确认轮子没有转向舵机，SDK 示例用
`Move(0,0,0.5)` 原地旋转；官方普通转向视频中四轮始终着地，没有明显抬轮或
大幅倾斜。因此不能先假设真机靠抬起或强制卸载内侧轮转向。真实平地 yaw 从
机械上属于左右差速并伴随横向轮胎 scrub；腿可能小幅稳定本体，但
`wheeled_sport` 的具体混控没有公开。

## 第 3 步：固定世界坐标 TCP 目标

每个 Isaac 并行环境都保存一个固定的最终目标：

```text
T_world_tcp_target
```

机器人本体运动时只能改变“当前 TCP”，不能重新生成或拖动最终目标。这样才会
出现正确行为：

- 目标在手臂舒适工作空间内：底盘不动，只用机械臂；
- 目标逐渐接近关节极限：底盘开始滚动或调整身体高度、俯仰；
- 底盘运动期间：机械臂反向补偿，继续保持世界 TCP；
- 到达目标后：轮速衰减到零，机械臂稳定保持目标。

目标不需要手工指定一个 XYZ 长方体。更可靠的方法是：

1. 采样安全的机械臂关节姿态；
2. 使用 FK 计算对应 TCP；
3. 将它作为保证可达的局部目标；
4. 后续给目标加入虚拟的 `ghost-base` 平移或旋转，使目标必须通过移动底盘才能
   到达。

修正环境原点后，64 环境独立测试已经证明这一语义成立。所有环境相互分开
2.5 m，最大 origin 误差仅 `5.96e-8 m`；前后、左右、升降、roll、pitch、yaw
和组合本体运动都通过，最坏终端抖动为 0.654 mm。完整结果见
[`B2W_Z1_WORLD_FRAME_GATE_ZH.md`](B2W_Z1_WORLD_FRAME_GATE_ZH.md)。

必须保留边界：该测试直接规定本体运动且不含重力/轮地动力学，因此只应作为
持续回归测试，不能代替正式 task 中由车轮和腿闭环产生的本体运动。

## 第 4 步：确定性 DIK、OSC 或 WBC 基线

- DIK（微分逆运动学）：把 TCP 速度/误差转换成关节速度或关节变化；
- OSC（操作空间控制）：直接在笛卡尔空间控制 TCP；
- WBC（全身控制）：同时考虑机械臂、身体、腿、轮子和约束。

推荐的精度架构是：

```text
固定世界 6D TCP 目标
  -> 平滑 pose/twist 参考
  -> RL 可达性与姿态协调器
  -> 冻结的批量 DIK/OSC 或受约束 WBC
  -> 腿、轮子和机械臂命令
```

当前已经实现的 MVP 协调器只输出：

```text
[v_x, yaw_rate]
```

含义分别是前后速度和转向速度。该 2 维策略验证 relocation 后，再做受控扩展加入
`[body_height, body_pitch]`。B2-W 不应直接要求横向速度 `v_y`，因为普通非转向轮
无法在不侧滑的情况下横移。

同时保留一个 22 维直接关节 RL 作为对照实验：12 个腿位置动作、4 个轮速
动作和 6 个机械臂位置动作。它适合验证端到端 PPO 是否能学习，但不是高精度
部署的首选方案。

## 第 5 步：创建独立 Isaac Lab 任务

已注册任务：

```text
B2W-Z1-TCP
B2W-Z1-TCP-Play
```

该任务必须使用 B2-W 对应的刚体和关节名称：

- 根刚体：`base_link`；
- 轮子：`.*_wheel`；
- 末端：显式刚体 `tcp_frame`，真机前把名义变换替换为测量值。

不能直接复制足式 B2 任务中的 `base`、`.*_foot`、`end_effector`，也不能把
足端腾空时间或足端滑动 reward 简单改名为轮子 reward。

Actor 的主要 observation：

- IMU 角速度和投影重力；
- 可在真机估计的底盘线速度；
- 腿和机械臂关节位置、速度；
- 轮子角速度，但不使用无限累积的轮子角度；
- TCP 位置误差和 SO(3) 姿态误差；
- 期望 TCP 线速度和角速度；
- 关节裕度或可达性指标；
- 上一步动作和三到五帧短历史。

Critic 可以额外看到仿真真值，例如接触力、真实底盘速度、摩擦、质量、负载、
延迟和碰撞距离。Actor 不能依赖真机上无法获得的特权信息。

Reward 需要鼓励：

- TCP 位置、姿态和运动速度跟踪；
- 在最终容差内连续稳定停留；
- 保持直立和合理身体高度。

同时惩罚或硬限制：

- 轮子侧滑和滚动速度不一致；
- 轮子离地、机身或非轮子部件触地；
- 目标可达或已经稳定时仍无意义移动底盘；
- 关节极限、奇异姿态和碰撞；
- 到点后的 TCP 抖动；
- 腿、轮子和机械臂各自的力矩、功率、加速度和动作变化；
- 跌倒、非法接触、控制器不可行或数值异常。

## 第 6 步：PPO 并行训练

建议使用 200 Hz 物理/控制频率和 50 Hz policy 频率，即
`dt=0.005`、`decimation=4`。环境数量逐步扩大：

| 环境数量 | 用途 |
| ---: | --- |
| 1 | GUI 检查、坐标系、轮子方向和碰撞调试 |
| 16 | 并行 reset、command、action、reward 冒烟测试 |
| 64--256 | 批量 DIK/OSC/WBC 数值稳定性和速度测试 |
| 512 | 第一轮 PPO overfit/debug，确认策略确实能够学习 |
| 2048 | 正式调 curriculum、reward 和 observation |
| 最多 4096 | 在显存、控制器耗时和 PPO 吞吐验证后进行完整训练 |
| 独立约 256 | 固定 held-out 目标评估，不参与训练 |

不能只看 PPO reward。每个 checkpoint 都应记录：

- TCP 位置和姿态的 RMS/P95 误差；
- 到达时间、超调和稳定后的抖动；
- episode 存活率；
- 不必要的底盘位移；
- 轮子滑移和离地比例；
- 关节裕度、碰撞和控制器 fallback；
- 力矩与速度饱和比例。

## 训练难度课程

| 阶段 | 学习内容 | 进入下一阶段的条件 |
| --- | --- | --- |
| 0 | 单机器人、标称平地、无 RL | 2000 步稳定，无错误方向、碰撞或 NaN |
| 1 | 固定身体、静态 FK 可达目标 | held-out 成功率至少 95%，存活率至少 99% |
| 2 | 固定身体、平滑 6D pose/twist 轨迹 | 标称动态 RMS 不高于约 10 mm / 5 度 |
| 3 | 加入必须移动底盘的 `ghost-base` 目标 | 能自然倒车/转向，而且不会持续无意义移动 |
| 4 | 安全初始姿态变化和少量真实负载 | 关节、碰撞和饱和检查通过 |
| 5 | 平地上的轻度质量、摩擦、噪声和延迟随机化 | 成功率至少 90%，标称性能退化小于 15% |
| 6 | 测量后的完整随机化范围和适度推力 | 至少三个随机种子通过，饱和低于 1% 步数 |
| 7 | 与部署环境匹配的轻微地形 | 地形通过且平地性能退化仍小于 15% |
| 8 | 冻结策略并进行 held-out 和 sim-to-sim 测试 | 根据完整指标矩阵选择 checkpoint |

这些阈值是首轮工程验收标准，不是已经实现的精度保证。

## 域随机化何时加入

域随机化用于覆盖仿真与真机的差异，但不应该在第一轮就全部打开。正确顺序：

1. 标称平地，无噪声、延迟和推力；
2. 证明静态、动态 TCP tracking 和底盘 relocation；
3. 加入轻度摩擦、质量、惯量、COM、增益、噪声和延迟随机化；
4. 用真机测量值逐渐扩大范围；
5. 最后增加适度外力和轻微地形。

完整随机化阶段仍保留约 20% 完全标称环境，防止鲁棒性训练破坏最终精度。
Z1 负载必须随机化质量、COM 和惯量，而不是只随机一个重量数值。

## 是否按照 MLM 加入随机地形

MLM 使用足式 Go2 + Airbot，在随机起伏、坡面、离散障碍物和楼梯之间进行
reward-threshold terrain curriculum；Critic 额外看到 187 个地形高度采样点。
论文还随机化质量、负载、摩擦、执行器强度和 0--20 ms observation delay。

来源：[MLM: Learning Multi-task Loco-Manipulation Whole-Body Control for
Quadruped Robot with Arm](https://arxiv.org/html/2508.10538)。

应该借用 MLM 的“根据 tracking 表现逐级增加难度”思想，但不要直接复制它的
地形。原因是 MLM 使用足式机器人，而 B2-W 使用非转向轮，接触和可通过地形
完全不同。论文也没有给出各类地形的准确比例、坡度、障碍高度或楼梯尺寸。

本项目的建议是先在平地完成 Stage 0--6，Stage 7 才加入室内环境匹配的地形。
4096 个最终并行环境可分配为：

- 819 个完全标称平地环境（20%）；
- 2048 个随机摩擦/动力学的平地环境（50%）；
- 819 个低起伏环境（20%），从 +/-5 mm 增加到最多 +/-10 mm；
- 410 个轻微坡面环境（10%），坡度不超过约 +/-3 度。

除非真实部署场景确实包含楼梯、碎石或明显障碍，否则不要加入这些地形。
当地形变化超过约 1--2 cm 时，应先加入真机能够获得的 lidar/height-map
观测，而不是要求策略盲目适应。纹理、光照和图像增强只对以后使用相机的
Actor 有意义，对当前 state-only controller 没有作用。

## 训练期间机械臂应该如何动作

- 普通 tracking episode 从 raised ready pose 开始，不从运输折叠姿态开始；
- Reset 后首先把当前 TCP 设为初始目标，避免目标突然跳变；
- 目标在舒适范围内时只移动机械臂，轮子保持静止；
- 关节裕度或可操作度变差时才允许底盘移动或调整身体；
- 底盘移动时，机械臂同步反向补偿，保持世界目标；
- 接近目标时减速，到达后轮速归零并使用阻尼稳定保持；
- 跌倒、危险碰撞或控制器失败时终止或进入明确的安全收回行为；
- free-space tracking 阶段固定夹爪，抓取动作以后单独训练；
- stow-to-ready 动作在普通 tracking 稳定之后再加入课程。

当前安全折叠姿态是
`[0.00, 0.15, -0.45, 0.30, 0.00, 0.00] rad`，普通训练 ready pose 是
`[0.00, 1.35, -1.65, 0.30, 0.00, 0.00] rad`。

## 紧接着应该实施什么

1. 保留 64 环境 world-frame harness，作为每次修改后的坐标回归测试；
2. 保留冻结 B2-W policy 的 stand/forward/reverse/left/right gate 及 checksum，
   继续把低速 yaw 不对称、轮子短暂卸载和停车尾部记录为 plant 特性；
3. **已完成：** 将该 policy 与 Z1 world-frame DIK 接入同一浮动底盘
   simulation loop，gravity 保持开启，且绝不直接写 root pose/velocity；
4. **已完成：** 通过 1、16、64 环境的组合 TCP 保持门槛；
5. **已完成：** 注册并集成 `B2W-Z1-TCP` 与 `B2W-Z1-TCP-Play`；
6. **已完成：** 通过正式 task 的 1、16、64 环境测试，以及 checkpoint
   保存、加载、TorchScript/ONNX 导出测试；
7. **已完成训练管线 pilot，但策略未收敛：** 256 环境、20 iterations 共采集
   327,680 transitions，运行稳定；相同 seed 的确定性评估没有超过 zero-action，
   所以不能称为最终 policy；
8. **首个解析 coordinator 诊断已完成：** 它改善 arm-home，但没有通过底盘
   relocation accuracy，left/lateral 与 yaw 尤其明显；下一步先固定分方向目标集和
   门槛、校准 reward/controller，再继续 512 环境标称训练与 held-out 评估；
9. 再建立经过碰撞、lidar clearance、joint margin 和 manipulability 过滤的 6D
   FK pose catalog。名义表现通过后才加入域随机化，terrain 最后加入。

Stage 3 的任务接口与 ghost-base 目标已经实现，但高层策略尚未学会有效
relocation。能够输出非零底盘动作不等于已经学会自然倒车/转向、释放机械臂
裕度，也不等于能在 TCP 到位后无振荡停住。
