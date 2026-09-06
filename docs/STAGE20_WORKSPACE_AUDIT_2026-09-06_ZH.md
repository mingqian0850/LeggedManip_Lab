# Stage20：B2W + Z1 EE 工作空间覆盖与审阅录像

日期：2026-09-06  
分支：`codex/b2w-z1-e2e-ee-wbc`  
策略：Stage18 `model_725.pt` + Stage19e reach-aware adaptive mirrored Filter35 action

## 目的

此前的动态基准以 TCP 前方小范围周期轨迹为主，平均误差不能回答“后方、侧方、极低位和复合姿态是否都自然”。Stage20 不训练模型，专门扩大确定性评估覆盖，并把 TCP 精度与本体姿态、四轮几何和载荷同时检查。

## 测试设计

新增两个互补任务：

- `B2W-Z1-EE-WBC-Stage20-Workspace-Grid-Adaptive-Mirrored-Filter35-Play-v0`：39 个环境并行测试 13 个命名目标，每个目标使用 0.85、1.00、1.15 三档幅度；一个目标失败不会阻止其他目标继续。
- `B2W-Z1-EE-WBC-Stage20-Workspace-Sweep-Adaptive-Mirrored-Filter35-Play-v0`：单机器人依次执行所有目标，用于人工视频审阅。

命令均相对落稳后的 TCP/本体坐标定义，使用 2.5 s minimum-jerk 过渡。批量任务随后保持 2.5 s；录像任务每个目标保持 1.5 s。

| 目标 | 基准位移 `(x,y,z)` m | 姿态 RPY | 检查重点 |
|---|---:|---:|---|
| front | `(0.40, 0, 0)` | 0 | 前进协同 |
| left/right | `(0.20, ±0.32, 0)` | 0 | 侧向移动、roll 与左右对称 |
| rear | `(-0.35, 0, 0)` | 0 | 是否自然倒车 |
| high | `(0.20, 0, 0.16)` | pitch −10° | 高位与姿态耦合 |
| low_front | `(0.30, 0, −0.16)` | pitch +10° | 低位俯身 |
| low_left/right | `(0.25, ±0.25, −0.12)` | roll ±8°、pitch +8°、yaw ±15° | 低位侧向复合姿态 |
| far_front | `(0.60, 0, 0)` | 0 | 超臂长目标和底盘协同 |
| pose_combo | `(0.20, 0.12, 0.05)` | `(15°, −10°, 25°)` | 训练边界外的 6D 组合 |
| rear_pose | `(−0.25, 0, 0.03)` | `(−10°, 10°, −25°)` | 后方 6D 组合 |

另含 home 与 return_home，用来检查零位偏差和动作可逆性。

## 批量结果

运行：seed 42，39 environments，430 steps（8.6 s），无随机扰动。

- 首回合存活：**39/39**。
- `low_base`、`bad_orientation`、`undesired_contact`：均为 **0**。
- 所有目标在 2.5 s 过渡结束时已经连续 0.25 s 满足 3 cm / 6°。
- 总体 TCP 位置误差：平均 **4.59 mm**，p95 **8.79 mm**，最大 **9.78 mm**。
- 总体 TCP 姿态误差：平均 **1.05°**，p95 **2.00°**，最大 **2.31°**。

| 目标 | 位置均值 mm | 姿态均值 ° | 最大本体 roll ° | 最大本体 pitch ° | 轮位镜像误差均值 cm | 接触载荷 CV 均值 | 最终底盘平移 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| front | 2.90 | 0.61 | 0.89 | 2.37 | 8.49 | 0.202 | 0.442 |
| far_front | 2.79 | 0.62 | 0.92 | 2.33 | 9.29 | 0.184 | 0.636 |
| rear | 3.22 | 0.72 | 0.89 | 1.79 | 6.25 | 0.166 | 0.324 |
| left | 5.04 | 0.34 | **8.03** | 1.45 | 9.71 | **0.570** | 0.243 |
| right | **8.75** | 1.36 | **10.86** | 2.68 | **12.65** | **0.576** | 0.376 |
| high | 8.26 | 0.89 | 1.39 | 2.37 | 3.08 | 0.174 | 0.276 |
| low_front | 4.40 | 0.81 | 1.46 | **5.99** | 11.53 | 0.216 | 0.308 |
| low_left | 2.08 | 0.94 | 3.78 | 4.88 | 9.76 | 0.353 | 0.268 |
| low_right | 6.61 | 1.12 | **5.58** | 4.98 | **15.19** | **0.532** | 0.346 |
| pose_combo | 4.94 | 1.87 | 2.25 | 1.90 | 3.55 | 0.178 | 0.262 |
| rear_pose | 3.72 | 1.75 | 4.06 | 2.37 | 8.99 | 0.225 | 0.158 |

底盘方向也符合命令：rear 的最终 x 平移均值为 −0.324 m；left/right 的 y 平移均值分别为 +0.122 m/−0.119 m；far_front 的 x 平移均值为 +0.635 m。因此“后方目标不倒车”和“左右符号反向”在这组确定性命令中没有复现。

完整 54 s 顺序轨迹又以无渲染模式复核一次：首回合 **1/1** 完整存活，无任何终止或自动 reset；连续跨目标切换时位置误差平均 8.35 mm、p95 18.69 mm、最大 24.52 mm，姿态误差平均 1.31°、p95 3.01°、最大 4.71°。因此录像中的运动是同一首回合连续完成，但跨越相距较远的目标时误差明显高于各目标从 home 独立出发的批量结果。

## 结论

Stage20 改变了对当前模型的评价：

1. **TCP tracking 本身已经可用。** 所有测试目标均达到 3 cm / 6°，实际平均误差仍在毫米和约 1°量级，前后大范围目标也会触发正确方向的底盘移动。
2. **全身姿态质量仍不合格。** 左右目标存在 8–11° 本体侧倾；右侧与右低位同时出现显著轮位不对称和接触载荷不均。视觉上四腿动作也不够协调。
3. **问题具有明显方向不对称性。** 前后方向比左右方向稳定；右侧通常比左侧更差。继续只优化全局平均 tracking reward 会掩盖这一点。
4. **当前模型不是最终 EE-WBC。** 它是可用于下一阶段消融和重新训练的基线，不能仅凭 39/39 存活率直接进入真机。

下一步应让训练分布显式包含 Stage20 的方向/高度/姿态分桶，并分别约束本体 roll、接触力分配和左右腿几何。更根本的候选方案是把腿部 raw joint residual 改成有物理语义的 base/stance action，再由四腿 IK 生成关节目标；Stage20 将作为两种方案共同的验收基准。

## 审阅视频

- 远端试验机上的带标签版本：`/home/mingqian/LeggedManip_Lab-e2e-ee-wbc/docs/validation/e2e_ee_wbc/stage20/stage20_workspace_sweep_model725_labeled.mp4`
- Windows/WSL 本地镜像：`implementation/e2e_ee_wbc/artifacts/stage20/stage20_workspace_sweep_model725_labeled.mp4`
- 仓库中的接触表：`docs/validation/e2e_ee_wbc/stage20/stage20_workspace_sweep_labeled_contact_sheet.jpg`
- 仓库中的原始数据：`docs/validation/e2e_ee_wbc/stage20/workspace_grid_model725_seed42.json`
- 顺序轨迹复核数据：`docs/validation/e2e_ee_wbc/stage20/workspace_sweep_model725_seed42.json`
- Windows/WSL 本地镜像另保留原始视频：`implementation/e2e_ee_wbc/artifacts/stage20/stage20_workspace_sweep_model725_raw.mp4`

视频未放入 Git 历史：仓库的 `*.mp4` 由 Git LFS 管理，而 GitHub 不允许当前 public fork 上传新的 LFS 对象。代码、报告、JSON 和接触表正常进入分支，视频保留在上述两台机器的明确路径中。

录像共 54 s、1280×720、50 fps。时间轴：0–2 s 落稳；之后每 4 s 依次为 home、front、left、right、rear、high、low_front、low_left、low_right、far_front、pose_combo、rear_pose、return_home。
