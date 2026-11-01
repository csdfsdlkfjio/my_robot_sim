# my_robot_sim

基于 **ROS 2 Humble + Gazebo Classic 11** 的两轮差速驱动机器人仿真示例：一个 URDF 小车模型、一个差速驱动插件配置，以及一个让小车走正方形的演示节点。

本项目刻意保持"最小但完整"，适合作为学习 URDF 建模、`gazebo_ros_diff_drive` 配置和差速底盘参数调试的起点。

---

## 项目简介

小车采用 **前二后一** 的布局：

- 车头两个驱动轮（左右各一，由差速驱动插件控制）
- 车尾一个万向轮（caster，球体，零摩擦，仅提供支撑）

模型通过 `spawn_entity.py` 从 `robot_description` 话题直接生成到 Gazebo 中，不依赖额外的模型数据库。

### 模型结构

| Link | 几何 | 质量 | 相对 `base_link` 位置 |
| --- | --- | --- | --- |
| `base_link` | box `0.4 × 0.25 × 0.05 m` | 1.0 kg | 原点 |
| `left_wheel` | cylinder `r=0.1 m, l=0.05 m` | 0.5 kg | `(0.10, +0.15, -0.10)` |
| `right_wheel` | cylinder `r=0.1 m, l=0.05 m` | 0.5 kg | `(0.10, -0.15, -0.10)` |
| `caster_link` | sphere `r=0.05 m` | 0.1 kg | `(-0.18, 0, -0.15)` |

- 三个着地点同在 `z = -0.20`，静止时 `base_link` 离地 **0.20 m**。
- 驱动轮绕 `y` 轴旋转（`axis xyz="0 1 0"`），轮距 0.30 m，轮径 0.20 m。
- 万向轮为 `fixed` 关节，球体 `mu1 = mu2 = 0`，可在地面上自由滑动。

### 差速驱动插件参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `update_rate` | 10 Hz | 与物理时钟对齐 |
| `wheel_separation` | 0.30 m | 左右轮中心距 |
| `wheel_diameter` | 0.20 m | 轮径 |
| `max_wheel_torque` | 50 N·m | 最大轮扭矩 |
| `max_wheel_acceleration` | 1.0 rad/s² | 最大轮角加速度 |
| `publish_odom` / `publish_odom_tf` | `true` | **ROS 2 中默认是 `false`，必须显式打开** |
| `odometry_frame` | `odom` | 里程计坐标系 |
| `robot_base_frame` | `base_link` | 机器人基座坐标系 |

---

## 目录结构

```
my_robot_sim/
├── launch/
│   └── sim.launch.py        # 启动 Gazebo + robot_state_publisher + 生成小车
├── urdf/
│   └── my_car.urdf          # 小车模型与 gazebo_ros_diff_drive 插件配置
├── my_robot_sim/
│   ├── __init__.py
│   └── square_mover.py      # 发布 /cmd_vel，让小车走正方形
├── resource/
├── test/                    # ament 代码风格测试
├── package.xml
├── setup.py
├── setup.cfg
└── .gitignore
```

---

## 依赖环境

| 依赖 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 LTS |
| ROS 2 | Humble Hawksbill |
| 仿真器 | Gazebo Classic 11 |
| Python | 3.10+ |

安装依赖：

```bash
sudo apt update
sudo apt install -y \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-robot-state-publisher \
  ros-humble-xacro \
  python3-colcon-common-extensions
```

> `gazebo_ros_diff_drive` 来自 `gazebo_ros_pkgs`，它同时提供 `spawn_entity.py`。

---

## 编译

本包是 `ament_python` 包，需要在 colcon 工作空间根目录编译：

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select my_robot_sim
source install/setup.bash
```

`--symlink-install` 会让 `install/` 中的 `urdf/`、`launch/` 直接指向 `src/`，修改模型后**无需重新编译**，重启 launch 即可生效。

---

## 运行

终端 1 —— 启动仿真：

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch my_robot_sim sim.launch.py
```

终端 2 —— 让小车走正方形：

```bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/my_robot_sim/my_robot_sim/square_mover.py
```

> `square_mover` 目前没有在 `setup.py` 中注册为 `console_scripts`，因此用 `python3` 直接运行；若要支持 `ros2 run my_robot_sim square_mover`，在 `setup.py` 的 `entry_points` 中加入对应条目后重新编译即可。

### 查看运行状态

```bash
ros2 topic echo /odom                 # 里程计
ros2 topic echo /joint_states         # 轮子关节状态
ros2 run tf2_tools view_frames        # 检查 odom -> base_link 是否存在
gz model -m my_car -p                 # 直接读取 Gazebo 中的位姿（x y z roll pitch yaw）
```

---

## 核心排错记录

以下是本项目从"能启动"到"能稳定走完正方形"过程中实际遇到并修复的三个问题，按时间顺序记录。

### 1. `/odom` 和 `odom → base_link` TF 不发布

**现象**

- `ros2 topic echo /odom` 收不到任何消息；
- RViz 中 TF 树缺少 `odom → base_link`；
- 依赖里程计的节点报 `No transform from [base_link] to [odom]`。

**原因**

`gazebo_ros_diff_drive` 在 **ROS 2 版本** 中，把 `publish_odom` 和 `publish_odom_tf` 的默认值都改成了 `false`（ROS 1 版本默认是 `true`）。插件本身会正常驱动轮子，但不会创建 `/odom` 话题，也不会广播 `odom → base_link` 的 TF。

**解决**

在 URDF 的插件配置中显式打开：

```xml
<plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
  <!-- ... -->
  <publish_odom>true</publish_odom>
  <publish_odom_tf>true</publish_odom_tf>
  <odometry_frame>odom</odometry_frame>
  <robot_base_frame>base_link</robot_base_frame>
</plugin>
```

**验证**

```bash
ros2 topic echo /odom          # 有数据
ros2 run tf2_tools view_frames # 出现 odom -> base_link
```

---

### 2. 出生高度过高，落地后姿态歪掉、万向轮着不了地

**现象**

`spawn_entity.py` 的 `-z` 设为 `0.5` 时，小车自由落体着地后车身是歪的，万向轮悬空，之后再也站不平。

**原因**

- 小车静止时 `base_link` 离地仅 **0.20 m**，从 0.5 m 出生意味着要掉 **0.30 m**；
- 下落高度大，着地瞬间冲击大、容易弹跳，三点支撑（两轮 + 万向轮）在落地瞬间被打乱；
- 由于当时重心余量极小（见下一条），一旦万向轮离地，模型就没有任何恢复水平姿态的能力。

**解决**

`launch/sim.launch.py` 中把出生高度降到 **0.25 m**（轮底距地 5 cm，只掉 5 cm）：

```python
arguments=['-topic', 'robot_description', '-entity', 'my_car', '-z', '0.25']
```

同时配合第 3 条的重心修复，让模型即使落地有扰动也能自行回正。

**验证**

重启后 `gz model -m my_car -p` 读到的 `pitch`、`roll` 均在 `1e-6 rad` 量级。

---

### 3. 重心压在轮轴线上，行驶中 48° 前倾栽倒

**现象**

让小车走正方形时，俯仰角反复冲到 **0.73 ~ 0.90 rad（42° ~ 51°）**，最后彻底停不住：

```
pitch = 0.846 rad ≈ 48.5°，z = 0.166 m
```

此时车头前沿撑在地上、驱动轮离地、车尾万向轮翘起约 0.2 m —— 也就是"车头扎地、屁股朝天"的姿态。

**原因分析**

原始模型把两个驱动轮的关节放在 `x = 0`，即底盘**正中间**，万向轮在 `x = -0.18`。按各 link 的质量计算整车重心：

```
总质量 M = 1.0 + 0.5 + 0.5 + 0.1 = 2.1 kg
x_com = (1.0·0 + 0.5·0 + 0.5·0 + 0.1·(-0.18)) / 2.1 ≈ -8.6 mm
```

- 重心只在轮轴后方 **8.6 mm**，几乎正压在轮轴线上；
- 万向轮只承担 **4.8%** 的重量（约 0.98 N），几乎是个"两轮平衡车"；
- 支撑多边形退化成一条边。只要刹车、转弯产生一点向前的惯性力，重心就越过轮轴，车头开始下跌；一旦下跌，万向轮离地，重心进一步前移，**没有任何恢复力矩**，最终栽倒。
- 按 1 m/s² 减速估算，惯性等效把重心前移约 5.6 mm，已经吃掉 8.6 mm 余量的大半。

**解决**

把两个驱动轮关节从 `x = 0` 前移到 `x = +0.10`（符合"前二后一"的实际布局），并按几何重新计算惯量：

```
轮子（圆柱，m=0.5, r=0.1, l=0.05）：
  I_轴  = ½·m·r²                      = 0.0025
  I_径向 = m·(3r² + l²)/12            = 0.001354

底盘（box，m=1.0, 0.4×0.25×0.05）：
  Ixx = (0.25² + 0.05²)/12            = 0.005417
  Iyy = (0.4²  + 0.05²)/12            = 0.013542
  Izz = (0.4²  + 0.25²)/12            = 0.018542
```

**修复后的重心**

```
x_com = (1.1·(-0.0164) + 1.0·0.10) / 2.1 ≈ +39 mm
```

重心位于轮轴后方 **61 mm**，万向轮承担约 **22%** 的重量，支撑余量从 8.6 mm 提升到 61 mm。

**验证**

在独立 Gazebo 实例中，用同一个 `square_mover` 完整跑完正方形并采样位姿：

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 运动中最大俯仰角 | **51°** | **0.19°** |
| 结束时俯仰角 | **48.5°**（栽倒） | **≈ 0°** |
| 结束时 `base_link` 高度 | 0.166 m（车头撑地） | **0.2000 m**（正常） |
| 横滚角 | 0° | 0° |

---

## 已知限制

- 使用 Gazebo **Classic 11**（`gazebo_ros`），不是 Gazebo Sim（`ros_gz`），迁移到新版本需改写插件标签。
- `square_mover` 用固定时长开环控制（直行 2 s、转弯 1 s），不读取 `/odom` 做闭环，实际轨迹会随地面摩擦和负载略有偏差。
- 万向轮为零摩擦球体，转向时靠滑动，不适合高精度路径跟踪场景。

> **WSL2 提示**：在 WSL2 中运行 Gazebo Classic 若出现黑屏/闪退，可尝试 `export LIBGL_ALWAYS_SOFTWARE=1` 使用软件渲染（帧率会明显下降）。
