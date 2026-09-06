# Stage 18：B2W 自然腿部动作实验记录（2026-09-06）

## 1. 为什么原策略看起来“歪”

原 actor 直接输出 12 个彼此独立的腿关节残差。虽然已有动作变化、髋关节默认姿态和左右关节镜像奖励，但 PD 目标每 20 ms 都可能变化，策略也可以用一个不对称的静态支撑形态换取 TCP 精度。Z1、夹爪和机身传感器又构成非对称载荷，因此“强制几何完全对称”同样不是正确答案。

本轮把自然性拆成四个可以独立测量的目标：

1. 时间平顺：腿关节速度、加速度、actor 输出二阶差分；
2. 静态形态：髋关节默认姿态偏差、轮心相对 nominal 的偏差；
3. 左右形态：左右轮对的镜像误差；
4. 承载质量：四轮接触力变异系数（CV），避免外形整齐但某个轮子几乎不承重。

## 2. 实现

- 新增 `LowPassJointPositionAction`。actor 仍观察和记录未经滤波的原始动作，只对送给 PD 的腿部位置目标做一阶低通，避免破坏现有 checkpoint 的 observation/action 语义。
- 比较 `alpha=0.50` 和 `alpha=0.35`；选用 `alpha=0.35` 后从 Stage16d `model_700` 适配 25 轮。
- 动态 benchmark 新增腿部动作二阶差分、腿关节速度/加速度、轮心 nominal 偏差、左右镜像误差和四轮载荷 CV。
- 新增低位 Stage5、空间协同 Stage6、极端低位 Stage10 的 Filter35 回归任务。
- 增加近距离 6D 录像视角；无显示器的远端录像必须显式传入 `--enable_cameras --headless`。

## 3. 主要结果

均为 seed 42、0.20 Hz、128 环境确定性基准；数值为 mean（p95）。

| 模型 | TCP 位置误差 | TCP 姿态误差 | 腿加速度范数 | 轮心 nominal 偏差 | 左右镜像误差 | 接触力 CV |
|---|---:|---:|---:|---:|---:|---:|
| 原始 `model_700` | 4.81 (9.94) mm | 1.07 (2.61)° | 11.51 (37.53) rad/s² | 18.26 (19.71) cm | 13.75 (16.32) cm | 0.222 (0.313) |
| Filter35 `model_725` | 5.16 (10.16) mm | 0.95 (2.38)° | 10.29 (33.24) rad/s² | 11.65 (14.15) cm | 8.19 (9.91) cm | 0.225 (0.348) |
| 平衡姿态 `model_736` | 5.86 (11.18) mm | 0.79 (2.36)° | 10.11 (31.50) rad/s² | 9.58 (12.24) cm | 5.47 (7.66) cm | 0.265 (0.430) |
| 强对称 `model_749` | 6.76 (12.97) mm | 0.68 (2.32)° | 10.41 (33.37) rad/s² | 7.48 (10.58) cm | 4.30 (8.54) cm | 0.412 (0.578) |

落地/settling 最大髋偏差：原始 0.207 rad，`model_725` 0.136 rad，`model_736` 0.102 rad，强对称模型 0.073 rad。强对称模型虽然外形最整齐，但载荷 CV 和瞬时跟踪误差明显恶化，因此拒绝。

### 困难目标回归

- `model_725` Stage5：128/128，无碰撞；过程位置误差 6.25 mm，p95 10.98 mm。
- `model_725` Stage6：128/128，无碰撞；过程位置误差 5.76 mm，p95 10.94 mm。
- `model_725` Stage10，seeds 42/43/44：383/384；唯一失败为一次非期望接触。最终位置误差均值 4.60–4.92 mm。
- `model_736` Stage10 seed 42：128/128；过程位置误差 7.11 mm，最终位置误差 5.38 mm。

### 快速目标

`model_725` 在 0.35 Hz 下的位置误差为 9.41 mm（p95 23.36 mm），姿态误差 1.29°（p95 4.05°）。它能稳定运行，但快速轨迹精度仍显著低于 0.20 Hz，不应把 0.35 Hz 宣称为毫米级跟踪。

## 4. 被拒绝的实验

1. **只做零样本滤波**：落地形态改善，但原 actor 会补偿滤波延迟，关节加速度反而略增；必须在滤波 plant 上适配。
2. **标准 `joint_acc_l2` 权重 `-2.5e-7` 再训练 25 轮**：平均加速度仅再降约 5.6%，位置误差从 5.16 mm 增至 6.16 mm，Stage10 出现一次姿态失败；拒绝。
3. **髋权重 `-2`、左右镜像权重 `-2`**：几何显著改善，但接触力 CV 从 0.225 增至 0.412，且产生 64.6 mm 瞬时位置误差；拒绝。原因是载有 Z1 的真实系统需要小幅非对称支撑来平衡载荷。

## 5. 当前选模结论

- **精度/稳健默认：`model_725`**。适合批量评估、困难目标和后续 sim-to-real 基线。
- **自然姿态演示：`model_736`**。静态腿形和左右一致性更好，Stage10 seed 42 仍为 128/128，但平均位置误差比 `725` 高约 0.7 mm，载荷 CV 也略高。
- 两者都比 Stage16d `model_700` 更适合作为当前 B2W+Z1 动态 EE tracking 基线。

检查点在 4090 远端分别为：

```text
logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_11-49-23_stage18a_filter35_from700_50/model_725.pt
logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_12-27-55_stage18d_filter35_balanced_stance_from725_12/model_736.pt
```

Windows/WSL 镜像：

```text
implementation/e2e_ee_wbc/artifacts/stage18/model725_precision.pt
implementation/e2e_ee_wbc/artifacts/stage18/model736_natural_stance.pt
implementation/e2e_ee_wbc/artifacts/stage18/model725_filter35_sixd_close_12s.mp4
implementation/e2e_ee_wbc/artifacts/stage18/model736_filter35_balanced_sixd_close_12s.mp4
```

## 6. 复现命令

```bash
cd /home/mingqian/LeggedManip_Lab-e2e-ee-wbc
PY=/home/mingqian/miniforge3/envs/env_isaaclab51/bin/python
export PYTHONPATH="$PWD/source/LeggedManip_Lab:${PYTHONPATH:-}"

$PY scripts/rsl_rl/play.py \
  --task B2W-Z1-EE-WBC-Dynamic-Filter35-SixD-Play-v0 \
  --checkpoint logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_12-27-55_stage18d_filter35_balanced_stance_from725_12/model_736.pt \
  --num_envs 1 --device cuda:0

# 无显示器的录像
$PY scripts/rsl_rl/play.py \
  --task B2W-Z1-EE-WBC-Dynamic-Filter35-SixD-Play-v0 \
  --checkpoint logs/rsl_rl/b2w_z1_e2e_ee_wbc/2026-09-06_12-27-55_stage18d_filter35_balanced_stance_from725_12/model_736.pt \
  --num_envs 1 --device cuda:0 --video --video_length 600 \
  --enable_cameras --headless
```

## 7. 下一步

本轮的低风险改动已经接近 Pareto 前沿，继续单纯增大奖励权重会用稳定性换外观。下一阶段应建立新的实验任务，不直接覆盖当前 22 维策略：

1. 将腿部 12 个独立残差改成语义 body/stance action，例如机身高度、roll、pitch、前后重心偏置、前后轮距；
2. 用四腿解析/数值 IK 将语义 action 映射为关节参考，并只保留很小的 learned residual；
3. 同时优化 TCP tracking、四轮持续接地、载荷 CV、关节/自碰撞约束；
4. 与 `model_725` 和 `model_736` 在相同三 seed、速度和低位课程上比较；
5. 通过后再加入 actuator delay、摩擦、质量和 COM 随机化。

这一步比继续堆“看起来对称”的奖励更有希望得到可解释、可迁移且姿态合理的 whole-body controller。
