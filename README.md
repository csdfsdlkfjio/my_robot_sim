# my_robot_sim

基于 **ROS 2 Humble + Gazebo Classic 11** 的两轮差速驱动机器人仿真示例：一个带 2D 激光雷达的 URDF 小车模型、差速驱动与射线雷达插件配置，以及走正方形和激光避障两个演示节点。

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
| `lidar_link` | cylinder `r=0.05 m, l=0.05 m` | 0.1 kg | `(0.10, 0, +0.05)` |

- 三个着地点同在 `z = -0.20`，静止时 `base_link` 离地 **0.20 m**。
- 驱动轮绕 `y` 轴旋转（`axis xyz="0 1 0"`），轮距 0.30 m，轮径 0.20 m。
- 万向轮为 `fixed` 关节，球体 `mu1 = mu2 = 0`，可在地面上自由滑动。
- 雷达装在车头 `x = +0.10`、底盘顶面之上，扫描平面离地 **0.25 m**（`base_link` 0.20 m + 雷达 0.05 m），不会扫到自身车体。

### 差速驱动插件参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `update_rate` | 10 Hz | 与物理时钟对齐 |
| `wheel_separation` | 0.30 m | 左右轮中心距 |
| `wheel_diameter` | 0.20 m | 轮径 |
| `max_wheel_torque` | 50 N·m | 最大轮扭矩 |
| `max_wheel_acceleration` | 1.0 rad/s² | 最大轮角加速度 |
| `publish_odom` / `publish_odom_tf` | `true` | **ROS 2 中默认是 `false`，必须显式打开** |
| `odometry_source` | `1` | `0` = 轮式积分，`1` = Gazebo 真实位姿。**这里必须是整数**：写成字符串 `world` 会被插件当 0 解析，退回会打滑的编码器里程计 |
| `odometry_frame` | `odom` | 里程计坐标系 |
| `robot_base_frame` | `base_link` | 机器人基座坐标系 |

### 激光雷达参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `sensor type` | `ray` | CPU 射线雷达，无需 GPU |
| `update_rate` | 10 Hz | 与差速插件同频 |
| `samples` | 360 | 每圈 360 个点，分辨率 1° |
| `min_angle` / `max_angle` | `-3.14` / `3.14` | 360° 全向扫描 |
| `range min` / `max` | 0.1 / 5.0 m | **超出范围的点返回 `inf`** |
| `noise stddev` | 0.01 m | 高斯噪声 |
| 输出话题 | `/scan`（`sensor_msgs/LaserScan`） | 由 `libgazebo_ros_ray_sensor.so` 发布，`frame_id = lidar_link` |

> **索引约定**：`angle_min = -π`，所以 `ranges[0]` 是**正后方**，正前方在索引 `len/2` 附近。按索引 0 找障碍物的代码一定会失效，必须按 `angle_min + i · angle_increment` 换算。

---

## 目录结构

```
my_robot_sim/
├── launch/
│   ├── sim.launch.py        # 启动 Gazebo + robot_state_publisher + 生成小车
│   ├── slam.launch.py       # slam_toolbox + 带配置的 RViz
│   └── nav2.launch.py       # Nav2 导航（map_server + AMCL + 规划/控制）
├── rviz/
│   └── slam.rviz            # LaserScan + Map + TF 显示配置
├── config/
│   └── nav2_params.yaml     # Nav2 参数（AMCL 的 base_frame_id 改为 base_link）
├── maps/
│   └── my_map.yaml/.pgm     # 由 demo.world 精确几何生成、与世界对齐的地图
├── worlds/
│   └── demo.world           # 8×8 m 演示房间（四面墙 + 两个障碍物）
├── urdf/
│   └── my_car.urdf          # 小车模型与 gazebo_ros_diff_drive 插件配置
├── my_robot_sim/
│   ├── __init__.py
│   ├── square_mover.py      # 发布 /cmd_vel，让小车走正方形
│   └── avoid_obstacle.py    # 订阅 /scan，激光雷达避障
├── resource/
├── test/                    # ament 代码风格测试 + 避障逻辑单元测试
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

终端 2 —— 激光雷达避障：

```bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/my_robot_sim/my_robot_sim/avoid_obstacle.py
```

> 避障节点和 `square_mover` 都会向 `/cmd_vel` 发指令，**不要同时运行**，否则两个节点会互相抢控制权。

### 避障节点行为与参数

节点订阅 `/scan`，取正前方 ±30° 扇区内**第 3 小**的有效距离（过滤单点坏值），按带迟滞的规则输出 `/cmd_vel`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `FRONT_HALF_ANGLE` | 30° | 正前方安全走廊半角 |
| `OBSTACLE_DISTANCE` | 0.6 m | 小于该距离开始原地转向 |
| `CLEAR_DISTANCE` | 1.0 m | 转向后必须大于该距离才恢复直行（迟滞） |
| `FORWARD_SPEED` | 0.3 m/s | 直行速度 |
| `TURN_SPEED` | 1.0 rad/s | 原地左转角速度 |
| `SCAN_TIMEOUT` | 1.0 s | 超过该时间没收到 `/scan` 就发零速度停车 |

终端 3 —— SLAM 建图 + RViz 可视化：

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch my_robot_sim slam.launch.py
```

`slam.launch.py` 会同时启动 `slam_toolbox` 和 RViz，RViz 使用项目自带的 `rviz/slam.rviz`：Fixed Frame = `map`，LaserScan 已绑定 `/scan`，并显示 `/map` 与 TF。

> 先关掉手动打开的 RViz 再运行，避免两个窗口同时订阅。手动建图时让小车慢慢走一圈即可，`slam_toolbox` 会用 `/scan` 和 `odom → base_link` 生成地图。

### 用演示世界建图（重要）

`sim.launch.py` 默认加载 Gazebo 的**空世界**，里面没有任何障碍物，雷达 360 个点全是 `inf`，SLAM 只会存出一张全 `unknown` 的空白地图。建图请指定 `worlds/demo.world`：

```bash
source ~/ros2_ws/install/setup.bash
DEMO=$(ros2 pkg prefix my_robot_sim)/share/my_robot_sim/worlds/demo.world
ros2 launch my_robot_sim sim.launch.py world:=$DEMO
```

演示世界是一个 8×8 m 的封闭房间，四面有墙；房内有一张 2.0 m（沿 y）× 0.6 m（沿 x）× 0.30 m 高的长桌（`grasp_table`，中心 `(1.45, 0, 0.15)`），桌子两端各有一个箱子；雷达从中心就能看到 4 m 外的墙。红方块 `grasp_box` 由 launch 在运行时生成，默认放在桌面中心 `(1.45, 0, 0.325)`。

建完图保存到源码目录：

```bash
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/src/my_robot_sim/maps/my_map
```

> `--symlink-install` 只对**编译时已存在**的文件建软链，新保存的 `my_map.*` 不会自动出现在 `install/` 里，需要再执行一次 `colcon build --symlink-install --packages-select my_robot_sim`。

仓库里自带的 `maps/my_map.*` 不是 SLAM 输出，而是**按 `demo.world` 的精确几何生成**的占据栅格（174×174 栅格、0.05 m/格，origin `[-4.35, -4.35]`，occupied 2004 / free 24240 / unknown 4032）。它和 Gazebo 世界坐标对齐：机器人出生在 `(0, 0)`、朝向 +x，所以 AMCL 的正确初值就是 `(0, 0, 0)`。长桌和两个箱子都在栅格里占位；红方块是运行时生成的，所以地图里没有它。

> 换成长桌后重新跑了 SLAM（`slam_toolbox` 在线建图 + `map_saver_cli`），两次结果都不理想：机器人在起点原地转一圈时前端雷达贴墙、回环约束少，存出来的图要么被裁掉一个角，要么相对世界坐标偏转几度到十几度。这种图会让 AMCL 初值不再是 `(0, 0, 0)`，搜索阶段的方块估计和 Nav2 的目标点都会跟着歪。最终提交的地图直接按 `demo.world` 的几何规则生成，房间和长桌的位置与世界坐标严格一致。自己重新建图时请先让车沿墙走完整一圈再保存；如果新地图的坐标系不是世界坐标系，记得到 `nav2_params.yaml` 里改 `initial_pose`。

> 旧版本里那张 223×267 的 SLAM 地图把房间相对世界旋转了约 52°，而且有两套重叠的墙线。原地搜索旋转时 AMCL 会在两个假设之间跳（实测 180° 跳变），Nav2 就会朝错误方向开。这一版地图去掉了那两处问题，`nav2_params.yaml` 里的 AMCL 初值也直接设成了 `(0, 0, 0)`。

### Nav2 导航

先停掉 `slam.launch.py`（建图和导航不要同时跑），再：

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch my_robot_sim nav2.launch.py
# 也可以指定别的地图：
# ros2 launch my_robot_sim nav2.launch.py map:=/绝对路径/xxx.yaml
```

`config/nav2_params.yaml` 里已经写好了 `set_initial_pose: true` 和 `initial_pose: (0, 0, 0)`，与仓库地图配套，启动后 AMCL 立即发布 `map → odom`。如果你换了地图（或启动 Nav2 前把机器人挪走了），要用 RViz 的 **2D Pose Estimate** 重新给初值，否则 Nav2 的 `global_costmap` 会一直等 `map → base_link`，`planner_server` 停在 inactive。

初值给好后，用 **Nav2 Goal** 发送目标点。

导航参数来自本包的 `config/nav2_params.yaml`（`nav2_bringup` 默认参数的一份副本，改动集中在 AMCL）：

- `base_frame_id` 用 `base_link`（本车没有 `base_footprint`）；
- 运动噪声 `alpha*` 收到 0.01、`max_beams` 提到 360——方形房间在旋转下几乎对称，默认参数会让粒子云在搜索旋转时被错误的 90° 假设吸走；
- `laser_model_type: "beam"`：长桌有 0.6 m 厚，似然场模型会把落在桌面内部的雷达端点也算成"完美命中"，粒子云因此滑到偏东 0.45 m 的伪解上（视觉把 1.45 m 处的方块算成 1.72 m）。换成 beam 模型后，横向抓取测试全程 AMCL 与 Gazebo 真值相差 2–5 cm、约 3° 以内；
- 进度检查放宽到 `required_movement_radius: 0.1`、`movement_time_allowance: 15.0`：默认要求 10 s 内移动 0.5 m，而观察位离起点只有 1 m 左右，最后一段常常是在长桌旁原地转向，`base_link` 位置不变，默认参数会把转向误判成卡死（实测在 y=+0.6 m 触发 `Failed to make progress`）。

另外 URDF 里的 `odometry_source` 必须是整数 `1`（Gazebo 真值）：插件用 `Get<int>` 解析这个标签，写字符串 `world` 会被读成 `0`，退回会打滑的轮式编码器里程计，AMCL 会持续被错误的先验拖偏。

需要调参时改这个文件，也可以这样换一份：

```bash
ros2 launch my_robot_sim nav2.launch.py params_file:=/绝对路径/nav2_params.yaml
```

### 自主搜索与移动抓取（search_and_grasp）

`my_robot_sim/search_and_grasp.py` 是移动操作的总控状态机，直接复用已经验证过的 `visual_grasp` 流程：

1. **搜索**：腕部相机移到前视位姿，底盘原地每步转 30°，共 12 步（360°）；每一步做红色分割 + 顶面反投影，得到方块在 `panda_link0` 下的点，再用 TF 换到 `map`。
   这里有两个和"转了 360° 也没看到红色方块"直接相关的细节：
   - **只信新帧**：WSL2 下 `/camera/image_raw` 实测只有 0.3–0.6 Hz（帧间隔最长 4.9 s），底盘转完时 `color_image` 里常常还是转之前、甚至机械臂还在运动时的旧帧。搜索会记下每步旋转前的图像时间戳，只用**晚于它**的帧检测（`wait_for_fresh_detection`：每步最多等 5 s，至少检查两帧新图）。开始搜索前还会先等 `/camera/image_raw` 出图，没有就立即报错并提示检查相机插件，而不是空转一圈。
   - **俯角避开夹爪**：相机装在夹爪正后方 6 cm，深度图在画面中心读到的就是夹爪自己（0.06 m，而房间在 1 m 以外），所以夹爪会占据画面中央。前视位姿的俯角因此加到 34°（`SEARCH_CAMERA_TILT`），让 1–3 m 处的桌面和方块投影到夹爪上方。实测从起点重启后**第 1 步**就能看到方块（像素 (364, 110)），不需要转动。
2. **导航**：调用 Nav2 的 `NavigateToPose`，走到离方块 `standoff`（默认 0.63 m）的位置，并转身面对方块。这个半径被两个约束挤出来：抓取在 `panda_link0` 下超过 `GRASP_REACH_MAX = 0.78 m` 就拒绝（Panda 实际能伸到约 0.8 m），而 `panda_link0` 在 `base_link` 后面约 4 cm，所以车体中心离方块最多约 0.74 m；本地代价地图按 0.22 m 车体半径膨胀长桌，整条 x ≥ 0.875 的带子都是内切障碍（代价 99），目标格子落在那里时 DWB 会报 "No valid trajectories" 并卡死，因此目标要留在 x ≤ 0.85。0.63 m 同时满足两边：目标 x ≈ 0.84，`panda_link0` 距离约 0.67 m。
   发送目标前用 `map_server` 的栅格做两级检查：以目标点为圆心、先按 `NAV_GOAL_SAFE_CLEARANCE = 0.30 m`（比内切带多留一格）检查，全部候选都不行再退回 `NAV_GOAL_CLEARANCE = 0.24 m`（车体外接半径 + 2 cm 余量）。距离按点到栅格矩形的**精确**距离算，不按格子取整，避免斜向目标被误拒。候选顺序是：标称目标 → 沿"车到方块"视线后退 2 cm（`NAV_GOAL_RETREAT_STEPS`，距离变远但朝向仍正对方块）→ 最后才试同一半径上 ±15°、±30°、±45° 的方位（`NAV_GOAL_BEARING_OFFSETS`）。**先退后绕**是实测教训：在长桌左侧 y=+0.6 m 时，守卫把标称目标 (0.86, 0.36) 直接换成 -15° 方位的 (0.82, 0.52)，这个点已经落到方块的侧后方，Nav2 原地打转后停在 (0.19, -0.42)，完全够不到。沿视线后退只把 `panda_link0` 距离增加 2 cm（上限 0.78 m 仍有余量），却不会破坏"车正对方块"的几何；绕方位会改变观察角，是最后手段。只有全部候选都不行才拒绝发送并报错，避免视觉定位异常时把目标点算到墙里或地图外。
   末端不再死等：剩余距离连续 20 s 都在 0.20 m 内却拿不到 `SUCCEEDED`（日志里常见的 0.00–0.16 m 反复横跳），就取消目标并进入"补完最后一段"。**但"到点"必须由 `/odom` 复核**：Nav2 在规划失败、只剩恢复行为时会反馈没有意义的剩余距离（非有限值或负数），旧代码把它当成"已在 0.20 m 内"；左侧 y=+0.6 m 实测车根本没动、`/odom` 离目标还有 0.99 m，20 s 后却被判成到点，接着"补完"因太远被拒，状态机还是继续向前挪了两次。现在 `_on_nav_feedback()` 直接忽略非有限/负的反馈；20 s 窗口到期时再用 `/odom` 量一次到目标的距离，只有不超过 `FINISH_DISTANCE_LIMIT = 0.25 m` 才按 `near` 处理，否则报 "剩余距离…无效反馈" 并返回失败。Nav2 报 `SUCCEEDED` 也不代表车停在观察位上：goal checker 在 xy 容差边缘（5 cm）就会锁定并返回成功，而无刹车的底盘随后还会继续漂移（实测沿 -x 漂了 5.7 cm，最后离方块 0.81 m，超出抓取范围）。所以 `finish_standoff()` 对两种结果都会执行：用 `/odom`（本仿真里等于 Gazebo 真值）低速补完最后不超过 25 cm 的直线段和朝向误差，再按搜索时估计的方块位置重新对准腕部相机——相机视野只有约 ±8 cm，14° 的底盘朝向误差就足以让方块出画。**`near` 和 `SUCCEEDED` 两条路径都必须检查 `finish_standoff()` 的返回值**：它拒绝（离目标超过 25 cm）时立即停车报错，不再进入视觉对准和抓取。真正到不了的目标仍受 180 s 硬超时保护。除此之外还有一层"原地打转"看门狗：剩余距离 60 s 内没有改善 5 cm（`NAV_STALL_TIMEOUT` / `NAV_STALL_PROGRESS`），就取消目标、停车并返回失败，而不是让 DWB 一直转到 180 s。2 cm 的来回抖动不算进展；一次掉头约 31 s（贴桌时实测约 0.1 rad/s），所以 60 s 不会误伤正常的转向。
   **关于 (0.82, 0.52) 和膨胀层**：这个 -15° 方位候选离 SLAM 地图里的桌边（占据格从 x = 1.15 m 开始）0.325 m，大于代价地图的内切半径 0.22 m，所以目标点本身**没有**被膨胀成致命格，`inflation_radius = 0.55 m` 只是在它周围抬高代价、不会让格子不可通行。这次车没动不是"目标被吞"：日志里没有出现任何实际行驶，20 s 后却被判为到点，符合"无路径时的无效剩余距离触发了看门狗"（见上一段）。如果 Nav2 终端里同时有 `Failed to create plan` / `No valid trajectories`，那才是规划器或 DWB 的问题，需要先把那份输出发出来。
   **"为什么导航停在离目标 8 cm 的地方"**：不是膨胀层把目标点吞了——日志里剩余距离多次降到 0.02 m，目标点本身在代价地图里是可达的，只是贴桌一侧的膨胀代价让最后的原地转向很慢（轮子打滑，实测约 0.1 rad/s），20 s 内拿不到 `SUCCEEDED`。真正造成 8 cm 缺口的是 goal checker 的 5 cm xy 容差**锁定**，加上无刹车底盘在锁定后继续后漂约 5.7 cm；8 cm 正好把 standoff 的几何放大：0.63 m（standoff）+ 0.04 m（`panda_link0` 在 `base_link` 后）+ 0.08 m（停车缺口）≈ 0.75 m，这就是左侧 y=+0.6 m 实测到的 `panda_link0` 距离。`finish_standoff()` 能补回大部分，但视觉估计和轮子打滑都会让残差留在几厘米量级，所以不能只靠导航把车停到精确位置。
   **够不到就向前挪（只在一步之遥时）**：`finish_standoff()` 和重新对准完成后，如果方块在 `panda_link0` 下仍然超出 `GRASP_REACH_MAX = 0.78 m`（正前方，横向 |y| ≤ 0.25 m），`align_camera_over_cube()` 会自动发 `/cmd_vel` 以 0.05 m/s 向前挪 10 cm（用 `/odom` 量距离，最多 2 次），停稳后重新对准腕部相机，再由正常的视觉抓取重新检测——不再直接报"超出可抓范围"退出。**前挪有上限 `NUDGE_FORWARD_MAX_DISTANCE = 1.0 m`**：两次 10 cm 只能补回"刚超出臂展"的距离。左侧 y=+0.6 m 实测 Nav2 没开动、方块还在 `panda_link0` 下 1.42 m，旧代码照样向前挪了两次，既没意义又把底盘挪离 standoff；现在这种情况直接报 `超过前挪能补回的上限 … Nav2 没有把底盘送到位` 并停车退出，让 Nav2 先把车开过去。方块太近、或偏到侧面时向前开同样没有意义，仍会拒绝并报出具体坐标。如果底盘是**停歪**而不是停远（朝向误差在 10°–90° 之间），`align_camera_over_cube()` 会先调用 `face_the_cube()` 原地转正（限幅 `NUDGE_ALIGN_MAX = 90°`），再向前挪；方块在底盘正后方超过 90° 时不做无意义的转弯，直接报错。两次前挪后仍对不准，就输出 `无法停到可抓范围内：停止并退出，不再原地调整。` 并停车返回失败，不会在方块旁边反复转圈。
3. **抓取**：底盘停稳后执行 `grasp_after_arrival()`（视觉测量 → 预抓取 → 下探 → 预闭合 → 10 N 夹紧 → 焊点固定 → 抬升 10 cm）。腕部相机装在手指上方，观察位姿下手指会挡住顶面靠近相机的一半；顶面点云改为从远离相机的那条真实棱边重建中心（见 `highest_face_center`），实测测量误差从 17 mm 降到 1 mm 左右。
   夹爪由两个控制器分阶段驱动：`panda_hand_controller`（位置 JTC）负责张开和预闭合，`panda_hand_effort_controller`（常力）负责夹紧。两者不能同时占用手指的位置/力接口，所以预闭合前会先查 `controller_manager/list_controllers`，确认 `panda_hand_controller` 是 `active`、effort 控制器是 `inactive`，必要时先切回来；夹紧前再切到 effort。这个检查是必需的：`ensure_gripper_open()` 在两指看起来已经张开时会直接返回，如果上一次夹紧失败后 effort 控制器仍占着接口，预闭合轨迹发给未激活的 JTC 会被接受、甚至报成功，但手指一动不动（实测两指停在 0.0392/0.0395 m，目标 0.0295 m，重发多次无效）。位置 JTC 的每关节目标容差从 10 mm 收紧到 **5 mm**，这个值要同时满足两边：旧值刚好覆盖整段预闭合行程（0.040 − 0.0295 ≈ 10.5 mm），控制器会把"张开限位"当成"已到目标"而立即返回成功；但也不能太紧——实测预闭合后两指停在 (0.0321, 0.0300) m、目标 0.0299 m，差 2.2 mm，这是夹爪真实的落座误差，剩下的由 10 N 夹紧补上。曾把容差收到 1.5 mm，结果在左侧 y=+0.6 m 时 JTC 以 `status=6 (ABORTED)` 中止了一条已经快夹到方块的预闭合轨迹。5 mm 既高于 2.2 mm 的落座误差，又远小于 11 mm 的最小行程（仍强制至少 6 mm 的真实移动）。轨迹失败时日志现在还会打印 JTC 的 `error_code` / `error_string` 和两指的实际位置，便于区分"指令没到夹爪"和"只差几毫米落座"。
   **深度测高的兜底**：方块始终放在桌面上，所以顶面高度是已知的（桌面 `table_height = 0.30 m` + 方块边长 `size = 0.05 m` = **0.35 m**；方块中心是 0.325 m）。视觉反投影的结果现在要过三道检查：① 深度小于 `MIN_VALID_DEPTH = 0.15 m` 的采样点直接丢弃——相机装在夹爪后 6 cm，近距离时画面中心读到的是夹爪自己（0.06 m），不能当方块；② `measure_target()` / `refine_target()` 只信机械臂停稳后拍到的帧，颜色和深度分别记录 header 时间戳，深度帧过期就退回"像素射线 × 已知高度平面"，不再拿旧帧配当前 TF；③ 顶面 z 与已知 0.35 m 相差超过 `SURFACE_HEIGHT_TOLERANCE = 0.03 m` 时，保留视觉测到的 x/y，把 z 换成已知高度并在日志里报出偏差。实测左侧 y=+0.6 m 时反投影把顶面算成世界 z=0.2131 m（比桌面低 8.7 cm），MoveIt 目标被带到 z=0.128 m，机械臂直接撞上长桌边缘；这类错误现在会被第 ③ 条拦住，即使深度图本身有问题也不会再让机械臂往桌子里扎。
4. **重试**：视觉测量或抓取失败时，先收回机械臂，再后退 20 cm，然后让 Nav2 重新开到方块前的 standoff 位置重试（默认 2 次）。后退后的距离上腕部相机只能看到方块的窄边，直接重抓不可靠；不先收臂的话，夹爪会带着方块一起后退，把方块拖倒（实测方块被甩出 2 m，下一次测量拿到错误高度）。

**前提**：`mobile_manipulation.launch.py`（Gazebo + 机械臂 + 相机）和 `nav2.launch.py` 都已在跑，且 AMCL 已定位（本仓库地图的初值已在参数里设好）。`mobile_manipulation.launch.py` 默认加载 `worlds/demo.world`：8×8 m 的封闭房间，四面墙（0.1 m 厚、1 m 高，四角互相搭接），底盘开不出去。要换回 Gazebo 空世界可以传 `world:=$(ros2 pkg prefix --share gazebo_ros)/worlds/empty.world`，但那里没有墙，不要用来跑导航。完整步骤：

```bash
source ~/ros2_ws/install/setup.bash
# 终端 1：仿真 + 机械臂（默认就是带墙的 demo.world）
ros2 launch my_robot_sim mobile_manipulation.launch.py
# 终端 2：Nav2（地图和初值都用仓库配置）
ros2 launch my_robot_sim nav2.launch.py
# 终端 3：总控
ros2 run my_robot_sim search_and_grasp
```

地图与墙壁的对齐关系：`maps/my_map.yaml` 的 174×174 栅格覆盖 `[-4.35, 4.35]`；墙内表面在 `±3.95`、外表面在 `±4.05`，在栅格里正好各占两列/两行，墙外 0.3 m 是 unknown。`test/test_map_assets.py` 会逐条检查这层对齐关系。

想确认墙真的挡住了底盘：`gz model -m wall_north -p` 应能读到墙体；朝任一方向持续给 `/cmd_vel` 直到贴墙，`gz model -m mobile_manipulator -p` 会停在墙内侧约一个车体半径处（约 `±3.7`），z 仍在 0.20 m 附近，不会掉进虚空。

如果启动 Nav2 时机器人不在世界原点，或者换了地图，给总控节点传初值：

```bash
ros2 run my_robot_sim search_and_grasp --ros-args \
  -p publish_initial_pose:=true -p initial_x:=0.0 -p initial_y:=0.0 -p initial_yaw:=0.0
```

**成功时终端里的关键输出**（数字来自实测）：

```
搜索：第 1 步看到红色区域，像素 (364, 110)，面积 751 px²，开始反投影...
搜索成功：方块在 panda_link0 下为 (1.5061, -0.0026, 0.1160)（来源：顶面点云），地图坐标为 (1.4537, -0.0027)。
导航：方块在 (1.45, -0.00)，机器人在 (-0.01, 0.00)，目标停在离方块 0.63 m 的 (0.82, -0.00)，朝向 -0° ...
方块上表面的世界坐标为 (1.4499, -0.0012, 0.3525)；...
下探后夹爪中心与方块中心的水平偏差为 1.2 mm（沿手指方向 0.9 mm，允许 12.0 mm）。
预闭合后两指停在 (0.0321, 0.0300) m，与目标 0.0299 m 相差 2.2 mm；这段距离会由随后的夹紧补上。
夹爪已用 10.0 N 的力夹住箱子，关节位置 (0.0330, 0.0112) m。
Gazebo 已把 grasp_box 固定到 panda_hand 上
找-走-抓完成：方块已被夹爪提起。
```

方块默认真实位置是 `(1.45, 0, 0.325)`（桌面上表面 z = 0.30 + 方块半高 0.025，见 `config/grasp_box.yaml`）。搜索阶段的地图坐标允许几厘米误差（AMCL 估计），最终抓取用的是腕部相机测得的世界坐标，所以抓取精度不受搜索误差影响。

#### 搜索看不到红色方块时的排查顺序

以下命令都在 `source ~/ros2_ws/install/setup.bash` 之后执行。如果日志里出现 `搜索：第 N 步没等到新图像`，说明相机话题已经停发，先按第 1、2 条查。

1. **图像话题到底有没有在发**
   ```bash
   ros2 topic info -v /camera/image_raw      # publisher 应为 arm_camera_controller，RELIABLE
   ros2 topic hz /camera/image_raw           # WSL2 下正常也只有 0.3-0.6 Hz
   ros2 topic echo --once /camera/image_raw --field header     # stamp 应在走
   ros2 topic echo --once /camera/image_raw --field encoding   # 应为 rgb8
   ```
   `ros2 topic hz` 对这么低的帧率要等十几秒才打印第一行，没立刻出数不等于没有图像。

2. **相机插件是不是加载失败了、Gazebo 是不是还活着**
   ```bash
   ros2 node list | grep camera      # /arm_camera_controller
   gz topic -l | grep camera         # /gazebo/default/mobile_manipulator/panda_link7/arm_camera/image
   ls -t ~/.ros/log | head -3        # 最近几次 launch 的日志目录
   dmesg | tail -40                  # WSL2 下 Gazebo 崩溃会留下 gzserver ... signal 6 / CaptureCrash
   ```
   没有 `arm_camera_controller` 节点就去看 launch 终端或 `~/.ros/log/latest/`：`spawn_entity` 失败、`gzserver` 提前退出都会让相机话题消失。本机诊断时遇到过 WSL2 图形栈把 `gzserver` 弄成 SIGABRT，表现就是仿真卡死、话题全部消失。

3. **图像里有没有红色像素**：插件发的是 `rgb8`，直接当 BGR 读会把红蓝对调（实测会把蓝色 `box_1` 统计成"红色"）。用 RViz（`ros2 run rviz2 rviz2`，加一个 Image display 选 `/camera/image_raw`）最直观；要用命令行，跑下面这段（Ctrl-C 结束）：
   ```bash
   python3 - <<'EOF'
   import cv2, numpy as np, rclpy
   from rclpy.node import Node
   from rclpy.qos import qos_profile_sensor_data
   from sensor_msgs.msg import Image
   class Check(Node):
       def __init__(self):
           super().__init__('red_check')
           self.create_subscription(Image, '/camera/image_raw', self.cb,
                                    qos_profile_sensor_data)
       def cb(self, msg):
           img = np.frombuffer(msg.data, np.uint8).reshape(
               msg.height, msg.step)[:, :msg.width * 3].reshape(
                   msg.height, msg.width, 3)
           if msg.encoding == 'rgb8':
               img = img[:, :, ::-1]
           hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
           mask = cv2.inRange(hsv, (0, 100, 60), (10, 255, 255)) | \
               cv2.inRange(hsv, (170, 100, 60), (180, 255, 255))
           print('red_pixels=%d' % int(mask.sum() / 255), flush=True)
   rclpy.init()
   rclpy.spin(Check())
   EOF
   ```

4. **是不是被夹爪挡住或超出视野**：画面中心读到的深度只有 0.06 m 时，那是相机正后方的夹爪自己（房间在 1 m 以外）。如果方块就正对车头、画面中央却没有红色，先怀疑夹爪遮挡——正常搜索位姿的俯角已经为此调到 34°，方块在画面里应出现在夹爪上方。方块放在桌上 `y ∈ [-1, 1]` 都在搜索覆盖范围内；放在桌外或离车不足约 0.8 m 时搜索位姿可能看不到（那种距离直接跑视觉抓取即可）。

### 手动测试长桌左 / 中 / 右抓取

长桌沿 y 轴摆放，测试点在桌中心 `(1.45, 0)` 以及左右各 0.6 m。约定：机器人从 `(0, 0)` 朝 +x 看桌子，**+y 是车的左边**。

1. 按上一节启动 `mobile_manipulation.launch.py` 和 `nav2.launch.py`，等 AMCL 的 `map → base_link` 出现（`ros2 run tf2_ros tf2_echo map base_link` 能打印即可）。

2. 在 Gazebo 里移动方块。可以直接用命令行设置模型位姿（等价于在 GUI 里拖动，但坐标精确）：

   ```bash
   # 左端
   gz model -m grasp_box -x 1.45 -y 0.6 -z 0.325
   # 中间
   gz model -m grasp_box -x 1.45 -y 0.0 -z 0.325
   # 右端
   gz model -m grasp_box -x 1.45 -y -0.6 -z 0.325
   ```

   `-z 0.325` 必须带上，否则方块会落到桌面以下。移动后确认：

   ```bash
   gz model -m grasp_box -p   # 应回读 x=1.45、y=±0.6/0、z≈0.325
   ```

3. 在第三个终端运行总控：

   ```bash
   source ~/ros2_ws/install/setup.bash
   ros2 run my_robot_sim search_and_grasp
   ```

   看日志确认三段：`搜索成功：方块在 ... 地图坐标为 (1.4x, ±0.6x)`、`导航：... 目标停在离方块 0.63 m 的 ...`、最后 `找-走-抓完成：方块已被夹爪提起。` 如果标称目标贴桌太近，日志会先出现一条 `导航目标 ... 不可用：...；沿视线后退 0.02 m，改用 ...` 的警告；只有沿视线后退仍不行时才会看到 `绕方块转 ±15°，改用 ...`。这两种都是候选切换，不是失败。若看到 `导航 ... 秒没有进展 ... 取消目标并停车` 或 `无法停到可抓范围内：停止并退出`，则是底盘确实到不了安全抓取点，状态机会停车报错而不是原地打转。

4. 测下一个位置前**重启整个仿真栈**（关掉 `mobile_manipulation.launch.py` 再重开）。成功抓取后 Gazebo 把方块焊到了 `panda_hand` 上，焊点会一直保留到本次仿真结束；此时用 `gz model` 挪方块只会让夹爪带着它一起动，下一次测量必然失败。失败重试（默认 2 次）和观察位自动后退（5/10 cm）都在同一次运行里自动完成，不需要手动干预。

5. 建议的范围与预期：

   | 方块 y | 预期 |
   | --- | --- |
   | `+0.6` / `0.0` / `-0.6` | 正常抓取；斜向观察位的搜索估计可能偏东几厘米，总控会自动换一个方位（±15/30/45°）或后退 2 cm 后重试目标，随后抓取成功（三个位置均已实测通过） |
   | `\|y\| > 0.6` | 观察位开始贴近桌沿或落到桌内，`NAV_GOAL_CLEARANCE` 检查会拒绝；方块也接近 Panda 0.85 m 臂展的边缘，不建议作为测试点 |

   `gz model -m grasp_box -p` 读到的位姿是生成时刻的 Gazebo 真值；移动方块后若想确认雷达/AMCL 没有因此跑偏，可以再看 `ros2 topic echo /amcl_pose` 的协方差是否正常。

### 查看运行状态

```bash
ros2 topic echo /odom                 # 里程计
ros2 topic echo /joint_states         # 轮子关节状态
ros2 run tf2_tools view_frames        # 检查 odom -> base_link 是否存在
gz model -m my_car -p                 # 直接读取 Gazebo 中的位姿（x y z roll pitch yaw）
ros2 topic info /scan                 # 雷达有没有发布者（Publisher count 应为 1）
ros2 topic hz /scan                   # 雷达频率（应接近 10 Hz）
ros2 topic echo /cmd_vel              # 避障节点最终发出的速度指令
```

---

## 核心排错记录

以下是本项目从"能启动"到"能稳定走正方形、能避障、能建图、能导航"过程中实际遇到并修复的十个问题，按时间顺序记录。

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

### 4. 避障节点启动了，但小车不动

**现象**

避障节点正常打印 `避障节点已启动`，但小车一动不动；`ros2 topic info /scan` 显示：

```
Publisher count: 0
```

**原因（四个问题叠加）**

1. **正在运行的 URDF 里没有雷达**：Gazebo 中没有 `ray` 传感器，`/scan` 没有任何发布者。
2. **指令只写在回调里**：节点只在 `LaserScan` 回调中发布 `/cmd_vel`，没有雷达数据就永远不发布，日志只剩启动那一行。
3. **URDF 本身是坏的**：`<scan>` 块用 `</ray>` 闭合，XML 解析直接失败；同时 `lidar_link` / `lidar_joint` 重复定义。
4. **索引假设错误**：代码用 `ranges[0]` 当正前方。实际 `angle_min = -π`，索引 0 是**正后方**，正前方在 `len/2` 附近。（即使雷达正常，这套判断也会朝错误方向避障。）

**解决**

- URDF：修正标签闭合、删除重复定义，雷达固定在 `(0.10, 0, 0.05)`，扫描平面离地 0.25 m（高于底盘顶面，不会扫到自身）；
- 插件：`~/out:=scan` 重映射，`output_type = sensor_msgs/LaserScan`；
- 节点：按 `angle_min + i · angle_increment` 计算每个点的真实角度来筛选前方扇区，不再依赖索引；
- 增加看门狗：超过 1 s 没有 `/scan` 就发布零速度，避免失控。

**验证**

在独立 Gazebo 实例中（前方 1.0 m 处放一堵墙）实测：

| 检查项 | 结果 |
| --- | --- |
| `/scan` 发布者 | `Publisher count: 1`，`frame_id = lidar_link`，360 点 |
| `/scan` 频率 | 约 10 Hz（受实时因子影响） |
| `/cmd_vel` 是否发布 | 10 Hz，先直行后转向 |
| 小车是否移动 | 20 s 内位移超过 1.4 m |
| 行驶姿态 | `pitch / roll ≈ 1e-6 rad`，`z = 0.200 m` |

---

### 5. 避障时贴着墙一路蹭，最后把车顶翻

**现象**

第一版避障确实会转向，但转向后车仍旧朝墙逼近，反复"转一点 → 以为安全 → 再前进"，日志像这样跳变：

```
危险！前方障碍物 0.59 m，紧急左转
危险！前方障碍物 0.10 m，紧急左转   ← 单帧坏值
危险！前方障碍物 0.53 m，紧急左转
前方安全，最近距离 0.63 m，继续前进   ← 刚转一点就判定安全
```

长时间运行后车贴在墙上摩擦，最终被顶翻（`roll = π`）。

**原因**

1. **扇区太窄**：只看 ±10°，车头转 10° 墙就滑出扇区，读数瞬间变大；
2. **没有迟滞**：读数一旦超过 0.6 m 就恢复直行，而车身仍然斜着朝墙逼近；
3. **被单点坏值带偏**：取 `min()`，Gazebo 射线雷达偶发的 `range_min` 级坏值（0.10 m）一帧就能触发转向；
4. **`inf` 的歧义**：障碍物近于 `range_min`（0.1 m）时雷达返回 `inf`，节点把它当"前方空旷"，于是继续顶墙。

**解决**

| 修改 | 作用 |
| --- | --- |
| 扇区加宽到 **±30°** | 覆盖车身宽度，转向后不会立刻"看不到"障碍物 |
| 迟滞：转向后必须 **> 1.0 m** 才恢复直行 | 不会转一点点就继续前进 |
| 取扇区内**第 3 小**距离 | 过滤单点坏值，真障碍物占满扇区不受影响 |

**验证**

同一堵墙、同一初始位置重跑 20 s：

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 转向次数 | 反复来回（117 帧转向 / 80 帧前进） | **1 次**（1.8 s，约 104°） |
| 是否接触墙面 | 是，贴墙摩擦直至顶翻 | **否**（车头最远 x≈0.64 m，墙面 0.95 m） |
| 行驶姿态 | 贴墙时 roll 涨到 0.04 rad | **pitch / roll ≈ 1e-6 rad** |
| `base_link` 高度 | 0.204 m（被顶起） | **0.200 m** |

---

### 6. RViz 里选不到 `/scan`，提示 `Error subscribing: Empty topic name`

**现象**

RViz 中添加 LaserScan 显示后，Topic 下拉框为空或选不到 `/scan`，显示项状态栏报：

```
Error subscribing: Empty topic name
```

**排查**

```bash
ros2 topic list -t | grep scan      # /scan [sensor_msgs/msg/LaserScan]
ros2 topic info /scan -v            # Publisher count: 1（gazebo_ros_laser）
ros2 topic hz /scan                 # 约 10 Hz
ros2 topic echo /scan --once        # frame_id: lidar_link，ranges 有数据
ros2 run tf2_ros tf2_echo base_link lidar_link
```

排查结果：**话题存在、有数据、TF 也正常**，雷达和插件都没有问题——故障在 RViz 配置侧。

**原因**

- `Error subscribing: Empty topic name` 是显示项 **Topic 为空** 时的状态提示，不是"订阅失败"；RViz 只有选中话题后才会真正创建订阅；
- 手动添加 LaserScan 显示后，Topic 默认是空的，必须自己选；
- WSL2 软件渲染下 RViz 的话题下拉框可能弹不出来或显示为空；
- Fixed Frame 必须是 TF 树里真实存在的帧（`map` 或 `odom`），否则即使订阅成功也画不出点。

**解决**

1. 使用项目自带的配置启动，显示项、话题和 Fixed Frame 都已配好：

   ```bash
   ros2 launch my_robot_sim slam.launch.py
   ```

2. 手动添加时：在 LaserScan 的 **Topic 输入框里直接键入 `/scan`**（下拉框为空也能手输），Fixed Frame 选 `map`，Reliability 选 `Best Effort`。
3. 若列表始终为空：确认 `ros2 topic list` 能看到 `/scan` 后重启 RViz，或检查添加的是不是 LaserScan 显示（PointCloud2 显示的列表里不会出现 `LaserScan` 类型的话题）。

**验证**

RViz 启动后：

```bash
ros2 node info /rviz
```

能看到订阅 `/scan: sensor_msgs/msg/LaserScan` 与 `/map`，RViz 中显示红色的激光扫描点。

---

### 7. Nav2 启动后 RViz 全黑，Global Status 报 Error

**现象**

`ros2 launch my_robot_sim nav2.launch.py` 后 RViz 一片漆黑，无论怎么缩放移动都看不到地图，Global Status 显示 Error。地图文件就放在包目录下：

```
~/ros2_ws/src/my_robot_sim/my_map.yaml
~/ros2_ws/src/my_robot_sim/my_map.pgm
```

**排查**

```bash
ros2 lifecycle get /map_server   # unconfigured（正常应为 active）
ros2 lifecycle get /amcl         # unconfigured
ros2 topic info /map -v          # Publisher count: 0
```

`map_server` 节点起来了，但连配置都没通过；`/map` 没有任何发布者，AMCL 也就无法激活、不发布 `map → odom`，RViz 既没有地图也没有 TF。

**原因 1：地图文件没有被安装到 `share/`**

`nav2.launch.py` 里写的是：

```python
map_file = os.path.join(
    get_package_share_directory('my_robot_sim'), 'my_map.yaml')
```

`get_package_share_directory()` 返回的是 `install/.../share/my_robot_sim`，而 `setup.py` 只安装了 `urdf/`、`launch/`、`rviz/`——**`my_map.*` 从来没有被装进去**，`map_server` 读 yaml 时文件不存在，configure 失败，lifecycle manager 放弃启动，后面所有节点都停在 `unconfigured`。

**原因 2：这张地图本身是空白的**

即使把路径修好，这张图也没有任何可用信息。直接解 PGM 就能确认：

```
P5 / 280 45 / 255
12600 个像素，全部是 205（unknown）
```

同时检查实时雷达：

```bash
ros2 topic echo /scan --once    # 360 个 ranges，有效回波 0 个
```

世界是空的（`empty.world`），雷达看到的所有东西都超出 5 m 量程返回 `inf`，SLAM 没有任何 occupied/free 栅格可写，保存出来的自然是一张全 `unknown` 的图。

**解决**

1. 地图移到 `maps/` 目录，`setup.py` 增加安装规则：

   ```python
   (os.path.join('share', package_name, 'maps'), glob('maps/*')),
   (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
   ```

2. `nav2.launch.py` 改用 `share/maps/my_map.yaml`，并支持 `map:=` 覆盖；重新编译后 `map_server` 可以正常加载。
3. 新增 `worlds/demo.world`（8×8 m 房间 + 障碍物），`sim.launch.py` 支持 `world:=` 参数，让 SLAM 有东西可画。

**验证**

| 检查 | 修复前 | 修复后 |
| --- | --- | --- |
| `map_server` 状态 | `unconfigured` | **`active`** |
| `/map` 发布者 | 0 | **1** |
| 地图加载 | 文件不存在 | `280 X 45 map @ 0.05 m/cell` |
| 演示世界 SLAM 结果 | 全 `unknown`（空白图） | **occupied 396 / free 11690** |

> 判断一张已保存的地图是否为空：PGM 里所有像素都是 205（灰度 205 = unknown）。只要出现 0（occupied）或 254（free），就说明建图时确实采到了障碍物。
>
> 仓库当前自带的 `maps/my_map.*` 是按 `demo.world` 的精确几何生成的（174×174，occupied 1532 / free 24720），不是这张空白图；早期那版 SLAM 地图（223×267，occupied 2656 / free 25183）已被替换，原因见“用演示世界建图”一节。

### 8. RViz 的 RobotModel 报 `No transform from [left_wheel]`

**现象**

RViz 里机器人只显示底盘和雷达，左右轮消失，RobotModel 对它们报 `No transform from [left_wheel]` / `[right_wheel]`；`base_link`、`lidar_link` 却一切正常。

**排查**

```bash
ros2 topic info /joint_states      # Publisher count: 0
ros2 topic echo /tf_static --once  # 只有 caster_link、lidar_link
```

TF 树里固定关节（`caster_joint`、`lidar_joint`）由 `robot_state_publisher` 直接发布；但两个轮子是 `continuous` 关节，`robot_state_publisher` 必须先收到 `/joint_states` 才知道它们的角度，才会广播 `base_link → left_wheel`。

**原因**

`gazebo_ros_diff_drive` 只负责驱动轮子、发布 `/odom`，**不会发布 `/joint_states`**。URDF 里只配置了差速驱动和雷达两个插件，因此队列里没有任何关节状态，轮子永远没有 TF。

**解决**

在 `urdf/my_car.urdf` 末尾追加 Gazebo 关节状态发布插件：

```xml
<gazebo>
  <plugin name="joint_state_publisher"
          filename="libgazebo_ros_joint_state_publisher.so">
    <ros>
      <remapping>~/out:=joint_states</remapping>
    </ros>
    <update_rate>30</update_rate>
    <joint_name>left_wheel_joint</joint_name>
    <joint_name>right_wheel_joint</joint_name>
  </plugin>
</gazebo>
```

**验证**

| 检查 | 修复前 | 修复后 |
| --- | --- | --- |
| `/joint_states` 发布者 | 0 | **1** |
| `base_link → left_wheel` TF | 不存在 | **(0.1, 0.15, -0.1)** |
| `base_link → right_wheel` TF | 不存在 | **(0.1, -0.15, -0.1)** |
| 以 0.2 m/s 直行时轮速 | — | **2.0 rad/s**（轮半径 0.1 m） |

> 修改 URDF 后必须**重启 Gazebo 并重新 spawn 模型**——已经在运行的 gzserver 里加载的是旧插件，只重启 RViz 是没用的。

### 9. 一打开 RViz 就报 `Global Status: Error`

**现象**

`ros2 launch my_robot_sim nav2.launch.py` 后，RViz 左上角立刻是红色 Error，地图一片黑。终端里能看到：

```
[amcl]: Couldn't transform from lidar_link to base_footprint ...
[global_costmap.global_costmap]: Timed out waiting for transform from base_link
         to map to become available, tf error: Invalid frame ID "map"
```

**原因**

RViz 的 Fixed Frame 是 `map`，而 `map → odom` 只能由 AMCL 发布。本车 URDF 的根 link 是 `base_link`，**没有 `base_footprint`**；但 Nav2 自带的 `nav2_params.yaml` 里 AMCL 的 `base_frame_id` 默认就是 `base_footprint`。于是 AMCL 的每次激光回调都在 TF 查询处失败，初始位姿也应用不上，`map → odom` 永远不会出现，Fixed Frame `map` 不存在，Global Status 报 Error。

这种情况下点 **2D Pose Estimate** 也没用——AMCL 在激光回调里就已经返回了。

**解决**

把 AMCL 的 base frame 改成本车实际存在的 `base_link`。在 `config/nav2_params.yaml` 中：

```yaml
amcl:
  ros__parameters:
    base_frame_id: "base_link"
```

`nav2.launch.py` 通过 `params_file:=` 传入这份参数。

**验证**

| 检查 | 修复前 | 修复后 |
| --- | --- | --- |
| AMCL 日志 | `Couldn't transform ... base_footprint` 刷屏 | 无该错误 |
| `map → odom` TF | 不存在 | 给出初始位姿后存在 |
| RViz Fixed Frame `map` | 不存在，Global Status Error | 正常 |

---

### 10. `/odom` 和 Gazebo 真值相差十几米，底盘停在墙外

**现象**

连续测试左 / 中 / 右三个位置时，导航最后卡在离目标 8 cm 左右，20 s 后
由近目标分支接管；随后出现：

```
方块偏在底盘 80° 方向，先转正再向前挪 ...
[ERROR] 方块在 panda_link0 下为 (-7.00, -12.29)，超出可抓范围 ...
```

同时 Gazebo 里能看到**另一个**底盘停在房间墙外。

**原因**

上一次的 Gazebo/launch 没有被完全关掉，或者新一次启动把机器人重复
spawn 进了仍在运行的 Gazebo。此时同一个 ROS 域里有两个机器人：

- `/odom` 和 `/ground_truth/odom` 各有两个发布者，消息交替到达；
- 两个 gzserver 时 `/clock` 也有两个发布者；
- 同一个 `/cmd_vel` 同时驱动两个底盘。

于是 `/odom` 可能说离目标 0.11 m，而下一条真值来自墙外的旧底盘，把
方块算到 14 m 外——8 x 8 m 的房间里，单个机器人不可能出现这个距离。
`turn_by()` 读到的偏航也在两个机器人之间跳变，原地转正因此会拖很久。

**解决**

在每个终端按 Ctrl-C 停掉任务、Nav2 和仿真，确认没有残留：

```bash
pgrep -af 'gzserver|gzclient|search_and_grasp|nav2|ros2 launch'
```

没有输出后，再按顺序重新启动。启动后先确认三个话题都只有一个发布者：

```bash
ros2 topic info /odom                 # Publisher count: 1
ros2 topic info /ground_truth/odom    # Publisher count: 1
ros2 topic info /clock                # Publisher count: 1
```

**预防**

`search_and_grasp` 启动时调用 `check_single_robot()`：上述三个话题任一
发布者数量大于 1 就直接报错退出，不再让机械臂或底盘动作。日志会写明
是哪个话题、有几个发布者。

---

## 已知限制

- 使用 Gazebo **Classic 11**（`gazebo_ros`），不是 Gazebo Sim（`ros_gz`），迁移到新版本需改写插件标签。
- `square_mover` 用固定时长开环控制（直行 2 s、转弯 1 s），不读取 `/odom` 做闭环，实际轨迹会随地面摩擦和负载略有偏差。
- 万向轮为零摩擦球体，转向时靠滑动，不适合高精度路径跟踪场景。
- 避障采用的是**反应式规则**（前方扇区 + 迟滞），没有建图和路径规划：遇到 U 型障碍、狭窄通道或死角时会在原地反复转向，不会主动"退出来"。

> **WSL2 提示**：在 WSL2 中运行 Gazebo Classic 若出现黑屏/闪退，可尝试 `export LIBGL_ALWAYS_SOFTWARE=1` 使用软件渲染（帧率会明显下降）。
