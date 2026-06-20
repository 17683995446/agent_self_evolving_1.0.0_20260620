# Self-Evolving Agent

自驱永续进化Agent系统 — 基于第一性原理的自进化智能体。

## 简介

Self-Evolving Agent 是一个仿生架构的自驱永续进化智能体系统。它模拟生物体的"感知→评估→行动→学习"核心循环，具备目标驱动、技能习得、多agent协作、长任务编排、自主技能进化、群体自组织等能力。

## 快速开始

### 安装

```bash
git clone <repo-url>
cd self_evolving_agent
pip install -e .
```

安装后可直接使用 `self-evolving-agent` 命令，或通过 `python -m agent_core` 调用。

### 启动

```bash
# 使用默认配置启动
self-evolving-agent start

# 使用生产环境配置
self-evolving-agent start --env prod

# 使用自定义配置文件
self-evolving-agent start --config my_config.yaml
```

### 常用命令

```bash
self-evolving-agent start      # 启动Agent
self-evolving-agent stop       # 停止运行中的Agent
self-evolving-agent status     # 查询运行状态
self-evolving-agent health     # 健康检查
self-evolving-agent log        # 查看最近日志
self-evolving-agent dashboard  # 启动Dashboard
self-evolving-agent version    # 显示版本
```

### 健康检查

```bash
self-evolving-agent health --env dev
```

## CLI 用法

```
usage: agent [-h] [--version] {start,stop,status,health,dashboard,log,version} ...

Self-Evolving Agent - 自驱永续进化智能体

命令:
  start       启动Agent
  stop        停止运行中的Agent
  status      查询Agent运行状态
  health      健康检查
  dashboard   启动独立Dashboard服务
  log         查看最近日志
  version     显示版本信息

选项:
  -h, --help     显示帮助信息
  --version, -V  显示版本信息并退出
```

### start 选项

```
--env {dev,prod,test}        环境 (默认: dev)
--config PATH                配置文件路径 (覆盖--env)
--log-level LEVEL            日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
--online-interval SECONDS    在线循环间隔(秒)
--offline-interval SECONDS   离线循环间隔(秒)
--dry-run                    仅验证配置不启动Agent
```

## 配置

配置文件位于 `config/` 目录，采用 YAML 格式，支持多环境：

| 文件 | 说明 |
|------|------|
| `config/base.yaml` | 基础配置（所有环境共享） |
| `config/dev.yaml` | 开发环境覆盖配置 |
| `config/prod.yaml` | 生产环境覆盖配置 |
| `config/test.yaml` | 测试环境覆盖配置 |

通过 `--env` 选择环境，通过 `--config` 指定自定义配置文件。

### 零配置启动

不提供任何配置文件时，系统使用内置默认值启动，无需预先配置。

## 运行测试

```bash
# 运行全部测试
pytest

# 运行特定测试
pytest tests/agent_core/common/

# 运行根目录测试（同步架构）
python test_phase1_integration.py
```

## 项目结构

```
self_evolving_agent/
├── src/agent_core/          # 主包（异步架构）
│   ├── cli.py               # 统一CLI入口
│   ├── __main__.py          # python -m agent_core 入口
│   ├── common/              # 公共基础设施
│   ├── memory/              # 记忆系统与Agent主类
│   ├── perception/          # 感知模块
│   ├── predictor/           # 预测模型
│   ├── drive/               # 驱力引擎
│   ├── task/                # 任务执行
│   ├── monitoring/          # 监控与Dashboard
│   ├── security/            # 安全模块
│   ├── distributed/         # 分布式协作
│   └── energy/              # 能量管理
├── config/                  # 配置文件
├── tests/                   # 测试套件
├── docs/                    # 文档
├── scripts/                 # 辅助脚本
├── pyproject.toml           # 项目配置
└── requirements.txt         # 依赖清单
```

## 依赖

- Python >= 3.9
- numpy >= 1.24.0
- psutil >= 5.9.0
- pyyaml >= 6.0

## License

MIT
