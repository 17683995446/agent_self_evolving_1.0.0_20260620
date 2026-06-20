# 快速开始指南

## 目录

- [安装](#安装)
- [启动Agent](#启动agent)
- [常用命令](#常用命令)
- [配置](#配置)
- [运行测试](#运行测试)
- [故障排查](#故障排查)

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd self_evolving_agent

# 安装（开发模式，推荐）
pip install -e .

# 或仅安装依赖
pip install -r requirements.txt
```

安装后可直接使用 `self-evolving-agent` 命令。

## 启动Agent

```bash
# 零配置启动（使用内置默认值）
self-evolving-agent start

# 指定环境
self-evolving-agent start --env prod

# 指定配置文件
self-evolving-agent start --config my_config.yaml

# 自定义日志级别
self-evolving-agent start --log-level DEBUG
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `self-evolving-agent start` | 启动Agent |
| `self-evolving-agent stop` | 停止运行中的Agent |
| `self-evolving-agent status` | 查询运行状态 |
| `self-evolving-agent health` | 健康检查 |
| `self-evolving-agent dashboard` | 启动Dashboard |
| `self-evolving-agent version` | 显示版本信息 |
| `self-evolving-agent --version` | 快速查看版本 |

## 配置

配置文件位于 `config/` 目录，采用 YAML 格式，支持多环境：

| 文件 | 说明 |
|------|------|
| `config/base.yaml` | 基础配置（所有环境共享） |
| `config/dev.yaml` | 开发环境覆盖配置 |
| `config/prod.yaml` | 生产环境覆盖配置 |
| `config/test.yaml` | 测试环境覆盖配置 |

### 零配置启动

不提供任何配置文件时，系统使用内置默认值启动，无需预先配置。

### 配置优先级

从高到低：
1. CLI 参数（`--log-level` 等）
2. `--config` 指定的配置文件
3. `--env` 环境配置（`config/{env}.yaml`）
4. `config/base.yaml`
5. 内置默认值

## 运行测试

```bash
# 运行根目录测试套件（同步架构）
python test_phase1_integration.py
python test_goal_driven.py
python test_skill_library.py

# 运行全部测试
pytest

# 运行特定测试
pytest tests/agent_core/common/
```

## 故障排查

### 启动失败

```bash
# 检查健康状态
self-evolving-agent health

# 使用DEBUG日志排查
self-evolving-agent start --log-level DEBUG
```

### 端口被占用

```bash
# 停止已有实例
self-evolving-agent stop

# 或使用不同端口启动Dashboard
self-evolving-agent dashboard --port 9090
```

### 配置错误

配置校验失败时，CLI会输出具体错误信息和修复建议。常见问题：

- **YAML格式错误**：检查缩进（使用空格，非Tab）
- **字段类型错误**：参考 `config/base.yaml` 中的字段类型
- **缺少必填字段**：使用 `--config` 指定完整配置文件，或依赖零配置默认值
