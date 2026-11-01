# CLAUDE.md

本文件是本项目的开发约定。Claude Code 在本仓库内工作时必须遵守以下规则。

## 规则 1：每次改动完成后都必须创建对应的 git commit

- 任何代码、配置、URDF、launch 或文档改动完成后，**立即创建一个对应的 commit**，便于后续追踪和回退。
- 一个 commit 只包含一个逻辑改动；提交信息要说明"改了什么、为什么"。
- 提交前先 `git status` 检查是否有遗漏或误加的文件；提交后用 `git log --oneline -1` 确认结果。
- 提交信息末尾保留署名行：
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`
- 除非用户明确要求，否则**不要自动 `git push`**；推送属于对外操作，需要用户确认。

## 规则 2：每次改动都必须编写或更新测试，并确保全部通过后才能交付

- 新增或修改功能时，必须同时编写或更新对应的测试，不允许"只写实现、不写测试"。
- 交付给用户之前，必须确认**所有测试与验证全部通过**；任何一项失败都要先修复，不得把失败状态当作完成交付。
- 如果某项验证因环境原因无法执行（例如 WSL2 下 Gazebo 无法启动、没有显示器），必须明确告诉用户：哪一项没有验证、为什么、有什么风险。

### 本项目的验证命令

```bash
# 1. 编译
cd ~/ros2_ws
colcon build --symlink-install --packages-select my_robot_sim
source install/setup.bash

# 2. 代码风格 / 版权 / 文档测试
colcon test --packages-select my_robot_sim
colcon test-result --verbose

# 3. URDF 语法与 SDF 转换检查
check_urdf src/my_robot_sim/urdf/my_car.urdf
gz sdf -p src/my_robot_sim/urdf/my_car.urdf > /dev/null

# 4. 运行时验证（需要 Gazebo）
ros2 launch my_robot_sim sim.launch.py
python3 src/my_robot_sim/my_robot_sim/square_mover.py
gz model -m my_car -p    # 位姿检查：pitch / roll 应接近 0，z 应保持 0.20
```

### 模型/物理参数改动时的额外要求

修改 `urdf/my_car.urdf` 中的质量、惯量、轮子位置等物理参数后，除上述检查外还必须：

1. 重新计算整车重心，确认重心位于支撑多边形内（轮轴与万向轮之间），且万向轮承载不为零；
2. 跑一次 `square_mover` 并采样 `gz model -m my_car -p`，确认行驶和停止后 `pitch / roll` 均接近 0。

## 测试失败时的处理

1. 先读完整报错，不要凭猜测修改；
2. 修复后重新运行**全部**测试，而不只是失败的那一项；
3. 如果确认是环境问题（依赖缺失、Gazebo 在 WSL2 下崩溃），在交付说明中如实标注，不要谎称通过。
