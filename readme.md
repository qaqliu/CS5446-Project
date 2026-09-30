# CS5446 Project — Multi-Agent Reinforcement Learning for Railway Traffic Coordination

[中文](#chinese) | [English](#english)

<a id="chinese"></a>

## 项目介绍

这是 CS5446 课程项目的团队仓库。项目使用 Flatland 铁路仿真环境，研究多列车共享轨道时的交通协调问题，比较独立学习和协作学习在不同地图规模、列车数量下的表现，重点关注到站率、旅行时间、延误和死锁情况。

仓库目标是搭建一个可配置的训练和评估框架，让团队成员按统一接口实现策略网络、价值网络和奖励组件。

项目问题背景参考 [Flatland 官方竞赛页面](https://www.aicrowd.com/challenges/flatland)。原竞赛的介绍与研究结果见 [Flatland Competition 2020: MAPF and MARL for Efficient Train Coordination on a Grid World](https://proceedings.mlr.press/v133/laurent21a.html)。环境使用方法见 [Flatland 官方文档](https://flatland-association.github.io/flatland-book/)。

## 环境安装

安装 Conda 后，在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate cs5446-project
```

环境使用 Python 3.11，包含 PyTorch、TorchRL、TensorDict、Flatland、官方展平观测需要的 Ray RLlib，以及绘图、日志和 notebook 工具。`environment.yml` 为 Linux、Windows 配置 CUDA 12.8 版 PyTorch，为 macOS 配置无 CUDA 的版本。使用 CUDA 需要兼容的 NVIDIA 驱动，PyTorch 安装包提供所需的 CUDA 运行时。

已经创建环境后，同步后续依赖修改时执行：

```bash
conda env update -n cs5446-project -f environment.yml
```

可以用下面的命令检查主要库导入和 CUDA 可用情况：

```bash
python -c "import torch, torchrl, tensordict, flatland; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

在仓库根目录运行脚本或 notebook，通过 `src` 导入项目模块。环境使用和模块接口说明位于 [technical_guide](technical_guide/)。

## 开发状态

**代码尚未完成，项目仍在开发中。** 当前已实现 Actor、Critic 的基础接口及独立、共享、中央实现。核心策略网络、价值网络、奖励组件、正式采样与训练管线、配置运行入口和评估流程仍需继续实现。完整的配置驱动训练流程将在后续开发中提供。

---

<a id="english"></a>

## About the Project

This is the team repository for the CS5446 course project. The project uses the Flatland railway simulation environment to study traffic coordination when multiple trains share tracks. It compares independent and cooperative learning across different map sizes and train counts, focusing on completion rate, travel time, delay, and deadlocks.

The repository aims to provide a configurable training and evaluation framework so that team members can implement policy networks, value networks, and reward components through common interfaces.

The problem setting is based on the [official Flatland competition](https://www.aicrowd.com/challenges/flatland). For the original competition's introduction and research results, see [Flatland Competition 2020: MAPF and MARL for Efficient Train Coordination on a Grid World](https://proceedings.mlr.press/v133/laurent21a.html). Environment usage is described in the [official Flatland documentation](https://flatland-association.github.io/flatland-book/).

## Environment Setup

Install Conda, then run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate cs5446-project
```

The environment uses Python 3.11 and includes PyTorch, TorchRL, TensorDict, Flatland, Ray RLlib for the official flattened observation builders, and plotting, logging, and notebook tools. `environment.yml` configures CUDA 12.8 PyTorch wheels for Linux and Windows and a version without CUDA for macOS. CUDA execution requires a compatible NVIDIA driver; the PyTorch wheels provide the required CUDA runtime.

To synchronize later dependency changes with an existing environment, run:

```bash
conda env update -n cs5446-project -f environment.yml
```

Check the main library imports and CUDA availability with:

```bash
python -c "import torch, torchrl, tensordict, flatland; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Run scripts or notebooks from the repository root and import project modules through `src`. Environment guides and module interface specifications are available in [technical_guide](technical_guide/).

## Development Status

**The code is incomplete and the project is under development.** Actor and Critic interfaces, including independent, shared, and centralized implementations, are available. Core policy and value networks, reward components, the permanent sampling and training pipeline, configuration entry points, and evaluation workflows still need implementation. A complete configuration-driven training workflow will be provided as development continues.
